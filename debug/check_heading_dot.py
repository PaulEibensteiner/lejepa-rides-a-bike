"""Check whether ang_vel[2] is a faithful heading-rate, or whether lean leaks in.

For each step we compare:
  - heading_dot_recorded = ang_vel[2]      (what _observe stores in S.heading_dot)
  - heading_dot_findiff  = wrap(heading_t - heading_{t-1}) / dt
and we also measure the z-component of the body forward axis (should be ~0 if the
forward axis stays horizontal, in which case ang_vel[2] == d(heading)/dt exactly)
and the world-z projection of the pure-lean rotation axis.
"""

from __future__ import annotations

import logging

import numpy as np
import pybullet as p

from bike.environment import BikeEnv
from bike.state import S
from train import ManualControllerPolicy

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _wrap(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def run(dt: float = 0.001, max_steps: int = 6000) -> None:
    env = BikeEnv(max_steps=max_steps, wind_std=0.0, dt=dt)
    policy = ManualControllerPolicy(desired_heading=0.0, torque_diversion_std=0.0)
    policy.reset_episode()
    state, _ = env.reset(seed=7)

    prev_heading = float(state[S.heading])
    max_abs_err = 0.0
    max_fwd_z = 0.0

    for step in range(max_steps):
        action = policy(state)
        state, _, terminated, truncated, _ = env.step(action)

        heading = float(state[S.heading])
        hd_recorded = float(state[S.heading_dot])
        hd_findiff = _wrap(heading - prev_heading) / dt
        prev_heading = heading

        # Forward (body-x) axis z-component: if ~0, ang_vel[2] == d(heading)/dt.
        _, orn = p.getBasePositionAndOrientation(
            env._dyn.bike_id, physicsClientId=env._dyn.client_id
        )
        rot = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float64).reshape(3, 3)
        fwd_z = float(rot[2, 0])

        err = abs(hd_recorded - hd_findiff)
        max_abs_err = max(max_abs_err, err)
        max_fwd_z = max(max_fwd_z, abs(fwd_z))

        if step % 500 == 0:
            logger.info(
                "step %4d | heading=%+.4f lean=%+.4f | ang_vel_z=%+.5f "
                "findiff=%+.5f err=%.2e | fwd_axis_z=%+.2e",
                step,
                heading,
                float(state[S.lean]),
                hd_recorded,
                hd_findiff,
                err,
                fwd_z,
            )

        if terminated or truncated:
            break

    logger.info(
        "MAX |ang_vel_z - d(heading)/dt| = %.3e rad/s | MAX |fwd_axis_z| = %.3e",
        max_abs_err,
        max_fwd_z,
    )
    env.close()


if __name__ == "__main__":
    run()
