"""Train bicycle dynamics models via supervised next-state prediction,
then evaluate using CEM-based model-predictive control.

Training pipeline (per model):

  Phase 1 — Supervised training
      Collect ``--num-episodes`` episodes with a random steering policy
      (wind enabled).  Fit the dynamics model to (s, a) → Δs data using
      mean-squared error.  For Model 3 the SIGReg regularisation on the
      layer-2 activations is added to the loss.

  Phase 2 — CEM evaluation
      Run ``--eval-episodes`` episodes guided by CEM + the trained model.
      Report success rate (no fall) and mean distance.

Usage examples::

    # Train all three models
    python train.py

    # Quick smoke test with Model 1
    python train.py --model 1 --num-episodes 20 --epochs 50 --eval-episodes 5

    # Train Model 3 and save checkpoints to a custom dir
    python train.py --model 3 --save-dir checkpoints/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from bike.environment import BikeEnv
from bike.models import SimpleLeanHeadingModel, FullStateModel, SigRegFullStateModel
from bike.cem import CEMPlanner
from bike.metrics import TrainingMetrics

StateList = List[np.ndarray]
ActionList = List[np.ndarray]


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def random_policy(_state: np.ndarray) -> np.ndarray:
    return np.random.uniform(-0.3, 0.3, size=(1,)).astype(np.float32)


def collect_episode(
    env: BikeEnv,
    policy,
) -> Tuple[StateList, ActionList, StateList, bool]:
    """Run one episode; return (states, actions, next_states, fell)."""
    states, actions, next_states = [], [], []
    state, _ = env.reset()
    fell = False

    while True:
        action = np.asarray(policy(state), dtype=np.float32).flatten()
        next_state, _, terminated, truncated, _ = env.step(action)

        states.append(state.copy())
        actions.append(action.copy())
        next_states.append(next_state.copy())

        state = next_state
        if terminated or truncated:
            fell = terminated
            break

    return states, actions, next_states, fell


def collect_dataset(
    env: BikeEnv,
    policy,
    n_episodes: int,
    metrics: TrainingMetrics | None = None,
) -> Tuple[StateList, ActionList, StateList]:
    """Collect ``n_episodes`` and concatenate all transitions."""
    all_s, all_a, all_ns = [], [], []
    for _ in range(n_episodes):
        s, a, ns, fell = collect_episode(env, policy)
        all_s.extend(s)
        all_a.extend(a)
        all_ns.extend(ns)
        if metrics is not None:
            metrics.add_episode([np.array(x) for x in s], fell)
    return all_s, all_a, all_ns


# ───────────────────────────────────────────────────
# Supervised training
# ─────────────────────────────────────────────────────────────────────────────

def build_tensors(
    states: StateList,
    actions: ActionList,
    next_states: StateList,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (states, actions, deltas) float32 tensors."""
    s_t = torch.tensor(np.array(states), dtype=torch.float32)
    a_t = torch.tensor(np.array(actions), dtype=torch.float32)
    d_t = torch.tensor(np.array(next_states), dtype=torch.float32) - s_t
    return s_t, a_t, d_t


def train_model(
    model: nn.Module,
    s_t: torch.Tensor,
    a_t: torch.Tensor,
    delta_t: torch.Tensor,
    *,
    epochs: int,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device,
) -> None:
    """Supervised training of a dynamics model on (s, a) → Δs data.

    For SigRegFullStateModel the SIGReg loss on layer-2 activations is
    added automatically via ``model.forward_with_sigreg``.
    """
    model.to(device).train()

    loader = DataLoader(
        TensorDataset(s_t.to(device), a_t.to(device), delta_t.to(device)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    is_sigreg = isinstance(model, SigRegFullStateModel)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for s_b, a_b, d_b in loader:
            if is_sigreg:
                pred, sigreg_loss = model.forward_with_sigreg(s_b, a_b)
            else:
                pred = model.predict_delta(s_b, a_b)
                sigreg_loss = torch.tensor(0.0, device=device)

            # Align target to model output dimension:
            # Model 1 predicts only [Δlean, Δheading] (indices 2, 3)
            target = d_b[:, 2:4] if pred.shape[-1] == 2 else d_b

            loss = nn.functional.mse_loss(pred, target) + sigreg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            avg = epoch_loss / max(len(loader), 1)
            sigreg_str = "" if not is_sigreg else "  (incl. SIGReg)"
            print(f"    epoch {epoch:>4d}/{epochs}  loss={avg:.5f}{sigreg_str}")

    model.eval()


# ────────────────────────────────────
# CEM evaluation
# ───────────────────────────────────────────────────────────────────

def cem_episode(
    env: BikeEnv,
    planner: CEMPlanner,
    metrics: TrainingMetrics | None = None,
) -> bool:
    """Run one CEM-guided episode; return fell flag."""
    state, _ = env.reset()
    planner.reset()
    episode_states = [state.copy()]
    fell = False

    while True:
        action = planner.plan(state)
        state, _, terminated, truncated, _ = env.step(action)
        episode_states.append(state.copy())
        if terminated or truncated:
            fell = terminated
            break

    if metrics is not None:
        metrics.add_episode(episode_states, fell)
    return fell


# ──────────────
# Per-model training pipeline
# ────────────────────────────────

def run_training(
    model_cls,
    model_name: str,
    *,
    num_episodes: int,
    epochs: int,
    eval_episodes: int,
    device: torch.device,
    save_path: Path | None = None,
) -> Tuple[nn.Module, TrainingMetrics]:
    """Full training and evaluation pipeline for one dynamics model."""
    sep = "=" * 60
    print(f"\n{sep}\n  {model_name}\n{sep}")

    env = BikeEnv(max_steps=500, wind_std=0.02)
    model = model_cls()

    # ── Phase 1: data collection with random policy ─────────────────────
    print(f"\n[Phase 1] Collecting {num_episodes} random episodes (with wind)…")
    random_metrics = TrainingMetrics()
    all_s, all_a, all_ns = collect_dataset(
        env, random_policy, num_episodes, metrics=random_metrics
    )
    print(f"  Random policy: {random_metrics.summary()}")
    print(f"  Dataset size : {len(all_s)} transitions")

    # ── Phase 2: supervised training ────────────────────────────────────
    print(f"\n[Phase 2] Supervised training ({epochs} epochs)…")
    s_t, a_t, d_t = build_tensors(all_s, all_a, all_ns)
    train_model(model, s_t, a_t, d_t, epochs=epochs, device=device)

    # ── Phase 3: CEM evaluation ─────────────────────────────────────────
    print(f"\n[Phase 3] CEM evaluation ({eval_episodes} episodes)…")
    planner = CEMPlanner(model, target_heading=0.0)
    eval_metrics = TrainingMetrics()
    for _ in tqdm(range(eval_episodes), desc="eval"):
        cem_episode(env, planner, metrics=eval_metrics)

    print(f"  CEM result: {eval_metrics.summary()}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  Saved → {save_path}")

    env.close()
    return model, eval_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────

MODEL_REGISTRY = {
    "1": (SimpleLeanHeadingModel, "Model 1 — SimpleLeanHeadingModel  (lean + heading only)"),
    "2": (FullStateModel,         "Model 2 — FullStateModel          (full state)"),
    "3": (SigRegFullStateModel,   "Model 3 — SigRegFullStateModel    (SIGReg on layer-2 activations)"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train bike dynamics models (supervised) then evaluate with CEM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()) + ["all"], default="all",
        help="Which model(s) to train.",
    )
    p.add_argument("--num-episodes", type=int, default=50,
                   help="Random episodes for supervised training data.")
    p.add_argument("--epochs", type=int, default=200,
                   help="Supervised training epochs.")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="CEM evaluation episodes per model.")
    p.add_argument("--save-dir", type=str, default=".",
                   help="Directory for saved model checkpoints (.pt files).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"GPU    : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}")

    to_train = (
        MODEL_REGISTRY
        if args.model == "all"
        else {args.model: MODEL_REGISTRY[args.model]}
    )

    save_dir = Path(args.save_dir)
    results: dict[str, TrainingMetrics] = {}

    for key, (model_cls, name) in to_train.items():
        _, metrics = run_training(
            model_cls,
            name,
            num_episodes=args.num_episodes,
            epochs=args.epochs,
            eval_episodes=args.eval_episodes,
            device=device,
            save_path=save_dir / f"model_{key}.pt",
        )
        results[name] = metrics

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for name, m in results.items():
        print(f"  {name}")
        print(f"    {m.summary()}")


if __name__ == "__main__":
    main()
