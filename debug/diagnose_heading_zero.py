"""Diagnose why Model 1 predicts Δheading≈0.

Collects a small dataset the same way train.py does, builds the model-facing
tensors, and reports for each delta channel:
  - target std (how much signal there is to fit)
  - correlation of the target delta with each candidate predictor

This reveals (a) that Δheading/Δlean are tiny vs Δlean_dot/Δsteer_rate and
(b) that Δheading is explained by heading_dot (NOT in Model 1 inputs), not by
steer_angle.
"""

from __future__ import annotations

import logging

import numpy as np

from bike.dynamics import MAX_STEER
from bike.environment import BikeEnv
from bike.state import S
from train import (
    StickyGaussianTorquePolicy,
    build_tensors,
    collect_dataset,
)
from bike.models import MODEL_DIFF_DT

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
    s_np = s_t.numpy()
    a_np = a_t.numpy().reshape(-1)
    d_np = d_t.numpy()

    logger.info("dataset: %d transitions, control_interval_steps=%d", len(s), cis)
    logger.info("\n--- target delta std per channel ---")
    for idx in [
        S.lean,
        S.heading,
        S.lean_dot,
        S.heading_dot,
        S.steer_angle,
        S.steer_rate,
    ]:
        logger.info("  Δ%-12s std=%.5f", S(int(idx)).name, d_np[:, idx].std())

    # candidate predictors for Δheading
    predictors = {
        "steer_angle": s_np[:, S.steer_angle],
        "heading_dot": s_np[:, S.heading_dot],
        "lean": s_np[:, S.lean],
        "action": a_np,
    }
    logger.info("\n--- corr(Δheading, predictor) ---")
    dh = d_np[:, S.heading]
    for name, x in predictors.items():
        r = np.corrcoef(dh, x)[0, 1]
        logger.info("  corr(Δheading, %-12s) = %+.3f", name, r)

    # best linear fit Δheading = c * heading_dot, recover c and compare to dt
    hd = s_np[:, S.heading_dot]
    c = float(np.dot(hd, dh) / np.dot(hd, hd))
    logger.info(
        "\nbest-fit Δheading = %.5f * heading_dot  (dt per step=%.5f)", c, MODEL_DIFF_DT
    )
    logger.info("max|action|=%.1f (MAX_STEER=%.0f)", np.abs(a_np).max(), MAX_STEER)


if __name__ == "__main__":
    main()
