from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from bike.cem import CEMPlanner
from bike.dynamics import MAX_STEER
from bike.environment import BikeEnv
from bike.metrics import TrainingMetrics
from bike.models import (
    FullStateModel,
    MODEL_DIFF_DT,
    SigRegFullStateModel,
    SimpleLeanHeadingModel,
    SubsetModel,
)
from bike.state import S
from bike.visualization import (
    plot_metrics_bar,
    plot_state_traces,
    plot_torque_traces,
    plot_training_curves,
    plot_trajectory_comparison,
)

MODEL_REGISTRY = {
    "1": (SimpleLeanHeadingModel, "Model 1"),
    "2": (FullStateModel, "Model 2"),
    "3": (SigRegFullStateModel, "Model 3"),
}


def random_policy(_state: np.ndarray) -> np.ndarray:
    return np.random.uniform(-MAX_STEER, MAX_STEER, size=(1,)).astype(np.float32)


def _angle_diff(a1: float, a2: float) -> float:
    tmp = a1 - a2
    if abs(tmp) < np.pi:
        return float(tmp)
    if tmp > np.pi:
        return float(-(2 * np.pi - tmp))
    return float(2 * np.pi + tmp)


class StickyGaussianTorquePolicy:
    """Stateful exploration policy with temporally correlated torque.

    Per decision step: with probability p_new sample torque ~ N(0, sigma),
    otherwise keep previous torque. The first step of each episode uses 0 torque.
    """

    def __init__(self, p_new: float = 0.01, sigma: float = 10.0) -> None:
        self.p_new = float(p_new)
        self.sigma = float(sigma)
        self.prev_torque = 0.0
        self._first_step = True

    def reset_episode(self) -> None:
        self.prev_torque = 0.0
        self._first_step = True

    def __call__(self, _state: np.ndarray) -> np.ndarray:
        if self._first_step:
            self._first_step = False
            return np.array([0.0], dtype=np.float32)

        if np.random.rand() < self.p_new:
            self.prev_torque = float(np.random.normal(0.0, self.sigma))

        self.prev_torque = float(np.clip(self.prev_torque, -MAX_STEER, MAX_STEER))
        return np.array([self.prev_torque], dtype=np.float32)


class ManualControllerPolicy:
    """Reference-style heading-to-lean controller for data collection."""

    def __init__(
        self,
        desired_heading: float | None = 0.0,
        c1: float = -1.0,
        c2: float = 100.0,
        c3: float = 100.0,
        torque_diversion_factor: float = 10,
    ) -> None:
        self.desired_heading = desired_heading
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = float(c3)
        self.divert_steps = 1000
        self.correct_steps = 500
        self.torque_diversion_factor = float(torque_diversion_factor)
        self.torque_diversion = self.torque_diversion_factor

    def reset_episode(self) -> None:
        return None

    def __call__(self, state: np.ndarray) -> np.ndarray:
        leaning = float(state[S.lean])
        heading = float(state[S.heading])
        leaning_dot = float(state[S.lean_dot])

        if self.desired_heading is not None:
            heading_diff = _angle_diff(self.desired_heading, heading)
            desired_lean = self.c1 * heading_diff
            desired_lean = 1.0 / (1.0 + np.exp(-desired_lean)) - 0.5
        else:
            desired_lean = 0.0

        torque = self.c2 * (desired_lean - leaning) - self.c3 * leaning_dot

        # divert for x, then correct for x
        if self.divert_steps > 0:
            self.divert_steps -= 1
            torque += self.torque_diversion
        elif self.correct_steps > 0:
            self.correct_steps -= 1
        else:
            # both 0
            self.divert_steps = 800
            self.correct_steps = 500
            self.torque_diversion = self.torque_diversion_factor

        torque = float(np.clip(torque, -MAX_STEER, MAX_STEER))
        return np.array([torque], dtype=np.float32)


def collect_episode(
    env: BikeEnv,
    policy,
    episode_seed: int | None = None,
    control_interval_steps: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], bool]:
    states, actions, next_states = [], [], []
    if hasattr(policy, "reset_episode"):
        policy.reset_episode()
    state, _ = env.reset(seed=episode_seed)

    while True:
        state_start = state.copy()
        action = np.asarray(policy(state), dtype=np.float32).flatten()
        terminated = False
        truncated = False
        for _ in range(max(1, int(control_interval_steps))):
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state

            if terminated or truncated:
                break

        # Keep one transition per control interval (or shorter final partial interval)
        # so dataset dt matches MODEL_DIFF_DT used in build_tensors.
        states.append(state_start)
        actions.append(action.copy())
        next_states.append(state.copy())

        if terminated or truncated:
            return states, actions, next_states, bool(terminated)


def collect_dataset(
    env: BikeEnv,
    policy,
    n_episodes: int,
    metrics: TrainingMetrics | None = None,
    base_seed: int | None = None,
    control_interval_steps: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    all_s, all_a, all_ns = [], [], []
    for episode_idx in range(n_episodes):
        episode_seed = None if base_seed is None else base_seed + episode_idx
        s, a, ns, fell = collect_episode(
            env,
            policy,
            episode_seed=episode_seed,
            control_interval_steps=control_interval_steps,
        )
        all_s.extend(s)
        all_a.extend(a)
        all_ns.extend(ns)
        if metrics is not None:
            metrics.add_episode(s, fell, actions=a, keep_trajectory=False)
    return all_s, all_a, all_ns


def _wrap_to_pi_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def build_tensors(
    states: list[np.ndarray],
    actions: list[np.ndarray],
    next_states: list[np.ndarray],
    model_dt: float = MODEL_DIFF_DT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_t = torch.tensor(np.array(states), dtype=torch.float32)
    a_t = torch.tensor(np.array(actions), dtype=torch.float32)
    next_t = torch.tensor(np.array(next_states), dtype=torch.float32)
    # Model outputs are state increments over the model step (0.02s).
    d_t = next_t - s_t
    d_t[:, S.heading] = torch.tensor(
        _wrap_to_pi_np((next_t[:, S.heading] - s_t[:, S.heading]).cpu().numpy()),
        dtype=torch.float32,
    )
    return s_t, a_t, d_t


def split_train_val(
    s_t: torch.Tensor,
    a_t: torch.Tensor,
    d_t: torch.Tensor,
    val_ratio: float = 0.2,
    seed: int | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    n = s_t.shape[0]
    if seed is None:
        idx = torch.randperm(n)
    else:
        gen = torch.Generator()
        gen.manual_seed(seed)
        idx = torch.randperm(n, generator=gen)
    n_val = max(1, int(n * val_ratio))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return (
        s_t[tr_idx],
        a_t[tr_idx],
        d_t[tr_idx],
    ), (
        s_t[val_idx],
        a_t[val_idx],
        d_t[val_idx],
    )


def train_world_model(
    model: SubsetModel,
    shared_data,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[dict[str, list[float]], float]:
    print("\n[Phase 1] Training for {epochs} epochs using shared random dataset...")
    print(f"  Random policy: {shared_data['random_summary']}")
    print(f"  Dataset size : {shared_data['num_transitions']} transitions")

    s_t = shared_data["s_tr"]
    a_t = shared_data["a_tr"]
    delta_t = shared_data["d_tr"]
    model.to(device).train()
    loader = DataLoader(
        TensorDataset(s_t.to(device), a_t.to(device), delta_t.to(device)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "total_loss": [],
        "pred_loss": [],
        "regularization_loss": [],
    }

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        pred_loss_acc = 0.0
        regularization_loss_acc = 0.0

        for s_b, a_b, d_b in loader:
            pred = model.predict_delta(s_b, a_b)
            pred_loss, reg_loss = model.loss(pred, d_b)
            loss = pred_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred_loss_acc += pred_loss.item()
            regularization_loss_acc += float(reg_loss.item())

        denom = max(1, len(loader))
        history["total_loss"].append(total_loss / denom)
        history["pred_loss"].append(pred_loss_acc / denom)
        history["regularization_loss"].append(regularization_loss_acc / denom)

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"    epoch {epoch:>4d}/{epochs} "
                f"total={history['total_loss'][-1]:.5f} "
                f"pred={history['pred_loss'][-1]:.5f} "
                f"sig={history['regularization_loss'][-1]:.5f}"
            )

    model.eval()

    val_mse = evaluate_prediction_mse(
        model, shared_data["s_val"], shared_data["a_val"], shared_data["d_val"], device
    )
    print(f"Training done;  Held-out prediction MSE: {val_mse:.6f}")
    print(model.pretty_print(1e-3))
    return history, val_mse


@torch.no_grad()
def evaluate_prediction_mse(
    model: SubsetModel,
    s_t: torch.Tensor,
    a_t: torch.Tensor,
    d_t: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    pred = model.predict_delta(s_t.to(device), a_t.to(device))
    return float(model.loss(pred, d_t.to(device))[0].item())


def cem_episode(
    env: BikeEnv,
    planner: CEMPlanner,
    metrics: TrainingMetrics | None = None,
    keep_trajectory: bool = False,
    episode_seed: int | None = None,
    control_interval_steps: int = 1,
    on_step: Callable[[], None] | None = None,
) -> dict[str, Any]:
    state, _ = env.reset(seed=episode_seed)
    planner.reset()
    states = [state.copy()]
    actions: list[np.ndarray] = []

    while True:
        action = planner.plan(state)
        for _ in range(max(1, int(control_interval_steps))):
            actions.append(action.copy())
            state, _, terminated, truncated, _ = env.step(action)
            states.append(state.copy())
            if on_step is not None:
                on_step()
            if terminated or truncated:
                fell = bool(terminated)
                if metrics is not None:
                    metrics.add_episode(
                        states, fell, actions=actions, keep_trajectory=keep_trajectory
                    )
                return {
                    "states": np.asarray(states, dtype=np.float32),
                    "actions": np.asarray(actions, dtype=np.float32),
                    "fell": fell,
                    "duration": len(states),
                }


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_world_model_cem(
    model,
    history,
    model_name: str,
    model_key: str,
    *,
    eval_episodes: int,
    seed: int,
    results_dir: Path,
    eval_max_steps: int = 10_000,
    wind_std: float = 0.01,
    cem_loss: str = "direction_lean",
    val_mse,
) -> dict[str, Any]:

    print(f"\n[Phase 3] CEM evaluation ({eval_episodes} episodes)...")
    sep = "=" * 60
    print(f"\n{sep}\n  {model_name}\n{sep}")

    eval_env = BikeEnv(max_steps=eval_max_steps, wind_std=wind_std)

    model_dir = results_dir / f"model_{model_key}"
    data_dir = results_dir / "data"
    metrics_dir = results_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    control_interval_steps = max(1, int(round(MODEL_DIFF_DT / eval_env.dt)))
    planner = CEMPlanner(
        model,
        target_heading=0.0,
        dt=MODEL_DIFF_DT,
        horizon=200,
        loss_mode=cem_loss,
    )
    eval_metrics = TrainingMetrics()
    eval_rollouts: list[dict[str, Any]] = []
    total_eval_steps = eval_episodes * eval_max_steps
    with tqdm(range(eval_episodes), desc=f"eval-{model_key}-episodes") as ep_bar:
        with tqdm(total=total_eval_steps, desc=f"eval-{model_key}-steps") as step_bar:

            def _on_eval_step() -> None:
                step_bar.update(1)

            for episode_idx in ep_bar:
                rollout = cem_episode(
                    eval_env,
                    planner,
                    metrics=eval_metrics,
                    keep_trajectory=True,
                    episode_seed=seed + int(model_key) * 100_000 + episode_idx,
                    control_interval_steps=control_interval_steps,
                    on_step=_on_eval_step,
                )
                eval_rollouts.append(rollout)

    print(f"  CEM result: {eval_metrics.summary()}")

    ckpt_path = model_dir / "checkpoint.pt"
    torch.save(model.state_dict(), ckpt_path)

    history_path = model_dir / "loss_history.json"
    _save_json(history_path, history)

    heldout_path = model_dir / "heldout_pred_mse.json"
    _save_json(heldout_path, {"heldout_pred_mse": val_mse})

    eval_npz = data_dir / f"model_{model_key}_eval_rollouts.npz"
    np.savez_compressed(
        eval_npz,
        states=np.array([r["states"] for r in eval_rollouts], dtype=object),
        actions=np.array([r["actions"] for r in eval_rollouts], dtype=object),
        fell=np.array([r["fell"] for r in eval_rollouts], dtype=bool),
    )

    summary_path = metrics_dir / f"model_{model_key}_summary.json"
    summary_payload = {
        "model": model_name,
        "summary": eval_metrics.to_summary_dict(),
        "episodes": eval_metrics.episodes_as_dicts(),
    }
    _save_json(summary_path, summary_payload)

    eval_env.close()

    return {
        "model": model,
        "model_name": model_name,
        "model_key": model_key,
        "checkpoint": ckpt_path,
        "history": history,
        "heldout_pred_mse": val_mse,
        "eval_metrics": eval_metrics,
        "eval_rollouts": eval_rollouts,
        "summary_path": summary_path,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train bike dynamics models and persist full observability artifacts."
        )
    )
    p.add_argument("--model", choices=["1", "2", "3", "all"], default="all")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument(
        "--eval-max-steps",
        type=int,
        default=10_000,
        help="Max steps per CEM eval episode (independent of data collection).",
    )
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--manual-data-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of data-collection episodes generated by manual controller [0,1]."
        ),
    )
    p.add_argument(
        "--cem-loss",
        choices=["direction_lean", "lean_only"],
        default="direction_lean",
        help="CEM objective: heading+lean or pure lean stabilization.",
    )
    p.add_argument(
        "--wind-std",
        type=float,
        default=0.01,
        help="Wind noise standard deviation for data collection and evaluation.",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def collect_data(args) -> dict[str, Any]:

    # Collect one shared random dataset and one shared train/val split
    # so every selected model is trained and evaluated on the same data.
    pre_env = BikeEnv(max_steps=15000, wind_std=args.wind_std)
    data_control_interval_steps = max(1, int(round(MODEL_DIFF_DT / pre_env.dt)))
    dataset_policy = StickyGaussianTorquePolicy(p_new=0.1, sigma=20.0)
    manual_policy = ManualControllerPolicy(desired_heading=0.0)

    n_manual = int(round(args.num_episodes * args.manual_data_fraction))
    n_sticky = max(0, args.num_episodes - n_manual)

    print(
        "Dataset mix: "
        f"sticky={n_sticky} episodes, manual={n_manual} episodes "
        f"(manual fraction={args.manual_data_fraction:.2f})"
    )

    pre_metrics = TrainingMetrics()
    all_s, all_a, all_ns = [], [], []

    if n_sticky > 0:
        s_sticky, a_sticky, ns_sticky = collect_dataset(
            pre_env,
            dataset_policy,
            n_sticky,
            metrics=pre_metrics,
            base_seed=args.seed,
            control_interval_steps=data_control_interval_steps,
        )
        all_s.extend(s_sticky)
        all_a.extend(a_sticky)
        all_ns.extend(ns_sticky)

    if n_manual > 0:
        s_manual, a_manual, ns_manual = collect_dataset(
            pre_env,
            manual_policy,
            n_manual,
            metrics=pre_metrics,
            base_seed=args.seed + n_sticky,
            control_interval_steps=data_control_interval_steps,
        )
        all_s.extend(s_manual)
        all_a.extend(a_manual)
        all_ns.extend(ns_manual)

    # Keep transition order randomized after mixing policies.
    mix_idx = np.random.permutation(len(all_s))
    all_s = [all_s[i] for i in mix_idx]
    all_a = [all_a[i] for i in mix_idx]
    all_ns = [all_ns[i] for i in mix_idx]

    pre_env.close()

    s_t, a_t, d_t = build_tensors(all_s, all_a, all_ns, model_dt=MODEL_DIFF_DT)
    (s_tr, a_tr, d_tr), (s_val, a_val, d_val) = split_train_val(
        s_t, a_t, d_t, val_ratio=0.2, seed=args.seed
    )

    return {
        "random_summary": pre_metrics.summary(),
        "num_transitions": len(all_s),
        "s_tr": s_tr,
        "a_tr": a_tr,
        "d_tr": d_tr,
        "s_val": s_val,
        "a_val": a_val,
        "d_val": d_val,
    }


def make_plots(outputs: dict[str, dict[str, Any]], figure_dir: Path, model_summaries):
    figure_dir.mkdir(parents=True, exist_ok=True)
    loss_histories = {name: out["history"] for name, out in outputs.items()}
    episodes_by_model = {
        name: [
            {
                "states": r["states"].tolist(),
                "actions": r["actions"].tolist(),
                "fell": r["fell"],
                "duration": r["duration"],
            }
            for r in out["eval_rollouts"]
        ]
        for name, out in outputs.items()
    }

    plot_training_curves(loss_histories, figure_dir / "training_curves.png")
    plot_trajectory_comparison(
        episodes_by_model, figure_dir / "trajectory_comparison.png"
    )
    plot_state_traces(episodes_by_model, figure_dir / "state_traces.png")
    plot_torque_traces(episodes_by_model, figure_dir / "torque_traces.png")
    plot_metrics_bar(model_summaries, figure_dir / "model_metrics.png")


def main() -> None:
    args = parse_args()
    args.manual_data_fraction = float(np.clip(args.manual_data_fraction, 0.0, 1.0))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(
        f"GPU    : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}"
    )

    results_dir = Path(args.results_dir)

    selected = (
        MODEL_REGISTRY
        if args.model == "all"
        else {args.model: MODEL_REGISTRY[args.model]}
    )

    shared_data = collect_data(args)

    outputs: dict[str, dict[str, Any]] = {}
    for model_key, (model_cls, model_name) in selected.items():
        model = model_cls()
        history, mse = train_world_model(
            model,
            shared_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        outputs[model_name] = validate_world_model_cem(
            model,
            history,
            model_name,
            model_key,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            results_dir=results_dir,
            eval_max_steps=args.eval_max_steps,
            wind_std=args.wind_std,
            cem_loss=args.cem_loss,
            val_mse=mse,
        )

    model_summaries = {
        name: out["eval_metrics"].to_summary_dict() for name, out in outputs.items()
    }
    _save_json(results_dir / "metrics" / "all_models_summary.json", model_summaries)

    if not args.skip_plots and len(outputs) >= 1:
        make_plots(outputs, results_dir / "figures", model_summaries)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for name, out in outputs.items():
        print(f"  {name}")
        print(
            f"    {out['eval_metrics'].summary()} |"
            f" heldout_mse={out['heldout_pred_mse']:.6f}"
        )


if __name__ == "__main__":
    main()
