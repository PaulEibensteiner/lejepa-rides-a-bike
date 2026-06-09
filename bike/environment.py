"""Gymnasium environment for bicycle riding (constant-speed model).

Observation (4-D): [x, y, lean, heading]
    — speed is fixed (BIKE_SPEED) and is **not** part of the observation.

Action (1-D): [steering_angle]  ∈ [-MAX_STEER, MAX_STEER]

Reward per step:
    + BIKE_SPEED * cos(heading)   # forward-progress bonus
    - 10 * lean²                  # lean penalty
    -  5 * heading²               # heading-deviation penalty
    - 100  (one-off on fall)

Episode ends when:
    • |lean| > π/4  (fallen)           → terminated = True
    • step count ≥ max_steps           → truncated  = True

Wind: additive Gaussian noise on lean and heading at each physics step,
analogous to the "wind" perturbations in the two-neuron-network paper.

Rendering: optional PyBullet GUI (human mode only, optional dependency).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from bike.dynamics import (
    step as dyn_step,
    is_fallen,
    MAX_STEER,
    FALL_THRESHOLD,
    BIKE_SPEED,
    LATENT_DIM,
)


class BikeEnv(gym.Env):
    """Simulated constant-speed bicycle environment."""

    metadata = {"render_modes": ["human"]}

    STATE_DIM = LATENT_DIM   # 4
    ACTION_DIM = 1

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 500,
        dt: float = 0.05,
        detect_falls: bool = True,
        wind_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.dt = dt
        self.detect_falls = detect_falls
        self.wind_std = wind_std

        self.action_space = spaces.Box(
            low=-MAX_STEER, high=MAX_STEER, shape=(1,), dtype=np.float32
        )
        # [x, y] unbounded; lean in (-π/4, π/4); heading in (-π, π)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -FALL_THRESHOLD, -np.pi], dtype=np.float32),
            high=np.array([np.inf, np.inf, FALL_THRESHOLD, np.pi], dtype=np.float32),
        )

        self._state: np.ndarray = np.zeros(self.STATE_DIM, dtype=np.float32)
        self._step_count: int = 0
        self._pb_client: int | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        lean = self.np_random.uniform(-0.05, 0.05)
        heading = self.np_random.uniform(-0.1, 0.1)
        self._state = np.array([0.0, 0.0, lean, heading], dtype=np.float32)
        self._step_count = 0
        return self._state.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        steering = float(np.asarray(action, dtype=np.float32).flatten()[0])

        self._state = dyn_step(self._state, steering, self.dt, self.wind_std)
        self._step_count += 1

        fallen = self.detect_falls and is_fallen(self._state)
        timeout = self._step_count >= self.max_steps

        terminated = fallen
        truncated = timeout and not fallen

        reward = self._compute_reward(fallen)

        if self.render_mode == "human":
            self.render()

        return self._state.copy(), reward, terminated, truncated, {}

    def render(self) -> None:
        try:
            import pybullet as p  # optional dependency

            if self._pb_client is None:
                self._pb_client = p.connect(p.GUI)

            x, y, lean, heading = self._state
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0,
                cameraYaw=float(np.degrees(heading)),
                cameraPitch=-20,
                cameraTargetPosition=[x, y, 0.5],
            )
        except Exception as exc:
            print(f"[BikeEnv] PyBullet render unavailable: {exc}")
            self.render_mode = None  # disable further attempts

    def close(self) -> None:
        if self._pb_client is not None:
            try:
                import pybullet as p

                p.disconnect(self._pb_client)
            except Exception:
                pass
            self._pb_client = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_reward(self, fallen: bool) -> float:
        x, y, lean, heading = self._state
        if fallen:
            return -100.0
        progress = BIKE_SPEED * np.cos(heading)
        lean_penalty = -10.0 * lean**2
        heading_penalty = -5.0 * heading**2
        return float(progress + lean_penalty + heading_penalty)
