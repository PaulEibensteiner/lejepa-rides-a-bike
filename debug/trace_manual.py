"""Instrument the manual controller in BikeEnv to find why the bike falls.

Logs ground speed, lean, lean_dot, steer angle and applied torque over time so we
can see what actually diverges, and compares the simulator's physics timestep
against the reference main.py (which never calls setTimeStep -> default 1/240 s).
"""

from __future__ import annotations

import logging

import numpy as np

from bike.environment import BikeEnv
from bike.dynamics import BIKE_SPEED, WHEEL_TARGET_RAD_S, WHEEL_RADIUS_M
from bike.state import S
from train import ManualControllerPolicy

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run(dt: float, max_steps: int = 20000) -> None:
    """Roll out the manual policy and log diagnostics every 250 steps."""
    logger.info(
        "=== dt=%.6f s (%.1f Hz) | BIKE_SPEED=%.3f | back_wheel=%.3f rad/s "
        "(=%.3f m/s ground) ===",
        dt,
        1.0 / dt,
        BIKE_SPEED,
        WHEEL_TARGET_RAD_S,
        WHEEL_TARGET_RAD_S * WHEEL_RADIUS_M,
    )
    env = BikeEnv(max_steps=max_steps, wind_std=0.0, dt=dt)
    policy = ManualControllerPolicy(desired_heading=0.0, torque_diversion_std=0.0)
    policy.reset_episode()
    state, _ = env.reset(seed=7)

    for step in range(max_steps):
        action = policy(state)
        state, _, terminated, truncated, _ = env.step(action)
        if step % 250 == 0 or terminated:
            speed = float(np.hypot(state[S.x_dot], state[S.y_dot]))
            logger.info(
                "step %4d | speed=%.3f | lean=%+.4f | lean_dot=%+.4f | "
                "heading=%+.4f | steer=%+.4f | torque=%+.2f",
                step,
                speed,
                float(state[S.lean]),
                float(state[S.lean_dot]),
                float(state[S.heading]),
                float(state[S.steer_angle]),
                float(action[0]),
            )
        if terminated:
            logger.info(">>> FELL at step %d", step)
            break
        if truncated:
            logger.info(">>> SURVIVED to step %d", step)
            break
    env.close()


if __name__ == "__main__":
    run(dt=0.001)
