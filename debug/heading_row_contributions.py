"""Why does the heading row weight lean_dot above steer_angle?

Raw linear weights are not comparable across input channels because the rate
channels (lean_dot, steer_rate) are pre-scaled by DIFF_SCALE_TO_MODEL=0.02 in
predict_delta while steer_angle/lean enter raw. The meaningful measure of how
much an input drives Δheading is |weight| * std(input as the model sees it).

This script loads the trained Model 1, collects a dataset, scales the inputs
exactly as predict_delta does, and reports per-input:
  - the learned weight in the heading row
  - the std of that input in model-space
  - the contribution = |weight| * std (units of Δheading)
"""

from __future__ import annotations

import logging

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from bike.environment import BikeEnv
from bike.models import (
    MODEL_DIFF_DT,
    SimpleLeanHeadingModel,
    _scale_action_for_model,
    _scale_state_differentials_for_model,
)
from bike.state import S
from train import StickyGaussianTorquePolicy, build_tensors, collect_dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    env = BikeEnv(max_steps=15000, wind_std=0.0)
    cis = max(1, round(MODEL_DIFF_DT / env.dt))
    policy = StickyGaussianTorquePolicy(p_new=0.1, sigma=20.0)
    s, a, ns = collect_dataset(
        env, policy, n_episodes=20, base_seed=0, control_interval_steps=cis
    )
    env.close()

    s_t, a_t, d_t = build_tensors(s, a, ns)

    # Train a fresh Model 1 (single linear layer) on this data.
    model = SimpleLeanHeadingModel()
    loader = DataLoader(TensorDataset(s_t, a_t, d_t), batch_size=256, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(60):
        for s_b, a_b, d_b in loader:
            pred = model.predict_delta(s_b, a_b)
            pred_loss, reg_loss = model.loss(pred, d_b)
            opt.zero_grad()
            (pred_loss + reg_loss).backward()
            opt.step()
    model.eval()
    # Reproduce exactly the input vector predict_delta feeds the linear layer.
    scaled_state = _scale_state_differentials_for_model(s_t)
    sel = scaled_state[..., model.INPUT_INDICES]
    act = _scale_action_for_model(a_t)
    model_inputs = torch.cat([act, sel], dim=-1)  # [action, *INPUT_INDICES]

    input_names = ["action"] + [S(int(i)).name for i in model.INPUT_INDICES]
    heading_row = model.OUTPUT_INDICES.index(S.heading)
    weights = model.net[0].weight.detach()[heading_row]  # type: ignore[index]
    input_std = model_inputs.std(dim=0)

    logger.info("heading-row contributions (|w| * std of model-space input):")
    logger.info("  %-12s %10s %12s %12s", "input", "weight", "input_std", "contrib")
    rows = []
    for i, name in enumerate(input_names):
        w = float(weights[i])
        sd = float(input_std[i])
        rows.append((name, w, sd, abs(w) * sd))
    for name, w, sd, contrib in sorted(rows, key=lambda r: -r[3]):
        logger.info("  %-12s %+10.5f %12.5f %12.6f", name, w, sd, contrib)


if __name__ == "__main__":
    main()
