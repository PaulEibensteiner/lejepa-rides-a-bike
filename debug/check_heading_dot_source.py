"""Quantify what drives heading_dot: steering geometry vs. wheel slip.

Measures the bike wheelbase, then during a high-torque (sticky-like) rollout
compares the observed heading_dot against the kinematic-bicycle prediction
    heading_dot_kin = v * tan(steer_angle) / wheelbase
and reports how often wheels are off the ground (contact lost) -> slip regime.
"""

from __future__ import annotations

import logging

import numpy as np
import pybullet as p

from bike.dynamics import _STEER_JOINT, _FRONT_WHEEL_JOINT, _BACK_WHEEL_JOINT
from bike.environment import BikeEnv
from bike.state import S

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _wheelbase(env: BikeEnv) -> float:
    """Distance between front and back wheel joint frames in the bike base frame."""
    cid = env._dyn.client_id
    bike = env._dyn.bike_id
    front = np.array(p.getJointInfo(bike, _FRONT_WHEEL_JOINT, physicsClientId=cid)[14])
    back = np.array(p.getJointInfo(bike, _BACK_WHEEL_JOINT, physicsClientId=cid)[14])
    steer = np.array(p.getJointInfo(bike, _STEER_JOINT, physicsClientId=cid)[14])
    # Front wheel hangs off the steer link; sum parent-relative offsets along x.
    return float(abs((front[0] + steer[0]) - back[0]))


def run(dt: float = 0.001, max_steps: int = 4000) -> None:
    env = BikeEnv(max_steps=max_steps, wind_std=0.0, dt=dt, detect_falls=True)
    state, _ = env.reset(seed=3)
    cid = env._dyn.client_id
    bike = env._dyn.bike_id
    L = _wheelbase(env)
    logger.info("measured wheelbase L = %.4f m", L)

    rng = np.random.default_rng(0)
    torque = 0.0
    for step in range(max_steps):
        if rng.random() < 0.02:
            torque = float(rng.normal(0.0, 4.0))
        state, _, term, trunc, _ = env.step(np.array([torque], dtype=np.float32))

        v = float(np.hypot(state[S.x_dot], state[S.y_dot]))
        steer = float(state[S.steer_angle])
        hd_obs = float(state[S.heading_dot])
        hd_kin = v * np.tan(steer) / L if L > 1e-6 else float("nan")

        # contact points of each wheel with the plane (0 -> airborne)
        fc = len(p.getContactPoints(bike, linkIndexA=_FRONT_WHEEL_JOINT,
                                    physicsClientId=cid))
        bc = len(p.getContactPoints(bike, linkIndexA=_BACK_WHEEL_JOINT,
                                    physicsClientId=cid))

        if step % 100 == 0:
            logger.info(
                "step %4d | v=%.2f steer=%+.3f(%+.0f°) | hd_obs=%+.3f hd_kin=%+.3f "
                "| ratio=%+.2f | contacts f/b=%d/%d lean=%+.3f",
                step, v, steer, np.degrees(steer), hd_obs, hd_kin,
                (hd_obs / hd_kin) if abs(hd_kin) > 1e-3 else float("nan"),
                fc, bc, float(state[S.lean]),
            )
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    run()
