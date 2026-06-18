from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bike.cem import CEMPlanner
from bike.environment import BikeEnv
from bike.metrics import TrainingMetrics
from bike.models import (
    FullStateModel,
    MODEL_DIFF_DT,
    SigRegFullStateModel,
    SimpleLeanHeadingModel,
)
from bike.visualization import (
    animate_episode_pybullet,
    plot_metrics_bar,
    plot_sigreg_distribution,
    plot_state_traces,
    plot_training_curves,
    plot_trajectory_comparison,
    plot_wind_sweep,
)
from evaluate.list_outputs import (
    choose_top3,
    gather_outputs,
    print_inventory,
    write_results_readme,
)

MODEL_REGISTRY = {
    "1": (SimpleLeanHeadingModel, "Model 1"),
    "2": (FullStateModel, "Model 2"),
    "3": (SigRegFullStateModel, "Model 3"),
}
WIND_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(model_key: str, results_dir: Path, device: torch.device):
    model_cls, model_name = MODEL_REGISTRY[model_key]
    model = model_cls().to(device)
    ckpt = results_dir / f"model_{model_key}" / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, model_name, ckpt


def run_eval_episode(
    env: BikeEnv,
    planner: CEMPlanner,
    episode_seed: int | None = None,
    control_interval_steps: int = 1,
) -> dict[str, Any]:
    state, _ = env.reset(seed=episode_seed)
    planner.reset()
    states = [state.copy()]
    actions = []

    while True:
        action = planner.plan(state)
        for _ in range(max(1, int(control_interval_steps))):
            actions.append(action.copy())
            state, _, terminated, truncated, _ = env.step(action)
            states.append(state.copy())
            if terminated or truncated:
                return {
                    "states": np.asarray(states, dtype=np.float32),
                    "actions": np.asarray(actions, dtype=np.float32),
                    "fell": bool(terminated),
                    "duration": len(states),
                }


def evaluate_model(
    model_key: str,
    results_dir: Path,
    eval_episodes: int,
    device: torch.device,
    wind_std: float = 0.02,
    seed: int = 7,
) -> dict[str, Any]:
    model, model_name, _ = load_model(model_key, results_dir, device)
    env = BikeEnv(max_steps=15000, wind_std=wind_std)
    planner = CEMPlanner(model, target_heading=0.0, dt=MODEL_DIFF_DT)
    control_interval_steps = max(1, int(round(MODEL_DIFF_DT / env.dt)))

    metrics = TrainingMetrics()
    episodes = []
    for episode_idx in range(eval_episodes):
        episode = run_eval_episode(
            env,
            planner,
            episode_seed=seed + int(model_key) * 100_000 + episode_idx,
            control_interval_steps=control_interval_steps,
        )
        episodes.append(episode)
        metrics.add_episode(
            [s for s in episode["states"]],
            episode["fell"],
            actions=[a for a in episode["actions"]],
            keep_trajectory=True,
        )

    env.close()

    return {
        "model_key": model_key,
        "model_name": model_name,
        "model": model,
        "metrics": metrics,
        "episodes": episodes,
    }


def sigreg_projection_values(
    model: SigRegFullStateModel, episodes: list[dict[str, Any]], device: torch.device
) -> np.ndarray:
    states = np.concatenate([ep["states"][:-1] for ep in episodes], axis=0)
    actions = np.concatenate([ep["actions"] for ep in episodes], axis=0)

    s_t = torch.tensor(states, dtype=torch.float32, device=device)
    with torch.no_grad():
        h2 = model.encode_state(s_t).cpu().numpy()

    rng = np.random.default_rng(7)
    direction = rng.normal(size=(h2.shape[1],))
    direction /= np.linalg.norm(direction) + 1e-8
    proj = h2 @ direction
    proj = (proj - proj.mean()) / (proj.std() + 1e-8)
    return proj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate trained bike models and generate observability artifacts."
    )
    p.add_argument("--model", choices=["1", "2", "3", "all"], default="all")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--wind-sweep", action="store_true")
    p.add_argument("--wind-sweep-episodes", type=int, default=20)
    p.add_argument("--list-outputs", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    videos_dir = results_dir / "videos"
    metrics_dir = results_dir / "metrics"
    data_dir = results_dir / "data"
    for d in [figures_dir, videos_dir, metrics_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_keys = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    # Load training losses if available
    loss_histories = {}
    for k in selected_keys:
        history_path = results_dir / f"model_{k}" / "loss_history.json"
        if history_path.exists():
            loss_histories[MODEL_REGISTRY[k][1]] = _load_json(history_path)

    outputs: dict[str, dict[str, Any]] = {}
    for k in selected_keys:
        out = evaluate_model(k, results_dir, args.eval_episodes, device, seed=args.seed)
        outputs[out["model_name"]] = out

        np.savez_compressed(
            data_dir / f"model_{k}_eval_rollouts_latest.npz",
            states=np.array([e["states"] for e in out["episodes"]], dtype=object),
            actions=np.array([e["actions"] for e in out["episodes"]], dtype=object),
            fell=np.array([e["fell"] for e in out["episodes"]], dtype=bool),
        )

        _save_json(
            metrics_dir / f"model_{k}_evaluation_summary.json",
            {
                "model": out["model_name"],
                "summary": out["metrics"].to_summary_dict(),
                "episodes": out["metrics"].episodes_as_dicts(),
            },
        )

        # representative video: longest surviving episode
        rep = max(out["episodes"], key=lambda e: e["duration"])
        animate_episode_pybullet(
            rep["states"][0],
            rep["actions"],
            videos_dir / f"model_{k}_representative.mp4",
        )

    model_summaries = {
        name: out["metrics"].to_summary_dict() for name, out in outputs.items()
    }
    _save_json(metrics_dir / "evaluation_all_models_summary.json", model_summaries)

    episodes_by_model = {
        name: [
            {
                "states": ep["states"].tolist(),
                "actions": ep["actions"].tolist(),
                "fell": ep["fell"],
                "duration": ep["duration"],
            }
            for ep in out["episodes"]
        ]
        for name, out in outputs.items()
    }

    if loss_histories:
        plot_training_curves(loss_histories, figures_dir / "training_curves.png")
    plot_trajectory_comparison(
        episodes_by_model, figures_dir / "trajectory_comparison.png"
    )
    plot_state_traces(episodes_by_model, figures_dir / "state_traces.png")
    plot_metrics_bar(model_summaries, figures_dir / "model_metrics.png")

    # SIGReg distribution for model 3 if selected
    for name, out in outputs.items():
        if name != "Model 3":
            continue
        proj = sigreg_projection_values(out["model"], out["episodes"], device)
        stats = plot_sigreg_distribution(
            name,
            proj,
            figures_dir / "sigreg_distribution_model3.png",
            ks_stats_path=metrics_dir / "sigreg_distribution_stats_model3.json",
        )
        _save_json(metrics_dir / "sigreg_ks_model3.json", stats)

    if args.wind_sweep:
        wind_curve: dict[str, dict[str, float]] = {
            MODEL_REGISTRY[k][1]: {} for k in selected_keys
        }
        for k in selected_keys:
            model_name = MODEL_REGISTRY[k][1]
            for w in WIND_LEVELS:
                out = evaluate_model(
                    k,
                    results_dir,
                    eval_episodes=args.wind_sweep_episodes,
                    device=device,
                    wind_std=w,
                )
                wind_curve[model_name][f"{w:.2f}"] = out["metrics"].success_rate
        _save_json(metrics_dir / "wind_sweep_summary.json", wind_curve)
        plot_wind_sweep(wind_curve, figures_dir / "wind_sweep.png")

    categories = gather_outputs(results_dir)
    top3 = choose_top3(categories)
    readme = write_results_readme(results_dir, categories, top3)

    print("\nTop 3 files to inspect first:")
    for p in top3:
        print(f"- {p.resolve()}")
    print(f"\nInspection guide written to: {readme.resolve()}")

    if args.list_outputs:
        print("\nFull output inventory:")
        print_inventory(categories)


if __name__ == "__main__":
    main()
