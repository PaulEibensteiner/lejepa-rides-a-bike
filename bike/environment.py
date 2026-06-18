"""Gymnasium environment for bicycle riding.

Observation (10-D): [x, y, lean, heading, x_dot, y_dot, lean_dot, heading_dot,
                     steer_angle, steer_rate]

Action (1-D): [steering_torque]  ∈ [-MAX_STEER, MAX_STEER]

Reward per step:
    + BIKE_SPEED * cos(heading)   # forward-progress bonus
    - 10 * lean²                  # lean penalty
    -  5 * heading²               # heading-deviation penalty
    - 100  (one-off on fall)

Episode ends when:
    • |lean| > 5π/16 = 56.25°(fallen)      → terminated = True
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
    PyBulletBikeDynamics,
    MAX_STEER,
    FALL_THRESHOLD,
    BIKE_SPEED,
    LATENT_DIM,
)
from bike.state import S


def detailed_gymspace_validation(
    space: spaces.Space, instance, name: str = "", all_keys_mandatory: bool = False
):
    """
    Validates that the provided instance matches the expected action space.
    Raises ValueError if the instance does not match the action space.
    """
    if isinstance(space, spaces.Dict):
        if not isinstance(instance, dict):
            raise ValueError(
                f"Expected a dictionary for {name} space {space}, but got"
                f" {type(instance)}"
            )
        for key, subspace in space.spaces.items():
            if key in instance:
                detailed_gymspace_validation(
                    subspace, instance[key], f"{name}.{key}", all_keys_mandatory
                )
            elif all_keys_mandatory:
                raise ValueError(f"Key '{key}' is missing in the {name} instance.")
        for key in instance.keys():
            if key not in space.spaces:
                raise ValueError(
                    f"Unexpected entry '{key}: {instance[key]}' in the {name} instance."
                )
    elif isinstance(space, spaces.Box):
        if not isinstance(instance, np.ndarray):
            instance = np.asarray(instance)
        for i, val in enumerate(instance.flat):
            if val < space.low.flat[i]:
                raise ValueError(
                    f"{name} instance index {i} has value {val}, which is below the"
                    f" lower bound {space.low.flat[i]}"
                )
            if val > space.high.flat[i]:
                raise ValueError(
                    f"{name} instance index {i} has value {val}, which is above the"
                    f" upper bound {space.high.flat[i]}"
                )
    elif not space.contains(instance):
        raise ValueError(
            f"{name} instance {instance} does not match the expected {name} space"
            f" {space}."
        )


class BikeEnv(gym.Env):
    """Simulated velocity-aware bicycle environment."""

    metadata = {"render_modes": ["human"]}

    STATE_DIM = LATENT_DIM  # 10
    ACTION_DIM = 1

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 15000,
        dt: float = 0.001,
        detect_falls: bool = True,
        wind_std: float = 0.02,
        video_path: str | None = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.dt = dt
        self.detect_falls = detect_falls
        self.wind_std = wind_std
        self._dyn = PyBulletBikeDynamics(video_path)

        self.action_space = spaces.Box(
            low=-MAX_STEER, high=MAX_STEER, shape=(1,), dtype=np.float32
        )
        # Position is bounded by total travel distance; planar speed by BIKE_SPEED.
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -max_steps
                    * BIKE_SPEED
                    * self.dt,  # x, no more than what we can move in the given steps
                    -max_steps * BIKE_SPEED * self.dt,  # y
                    -FALL_THRESHOLD - np.pi / 8.0,  # lean
                    -np.pi,  # heading, no more than 180 deg
                    -BIKE_SPEED - 0.2,  # x_dot, no more than bike speed
                    -BIKE_SPEED - 0.2,  # y_dot, no more than bike speed
                    -np.pi * 2,  # lean_dot, no more than 180 deg / s
                    -np.pi / 2,  # heading_dot, no more than 90 deg / s
                    -np.pi / 2,  # steer_angle, no more than 90 deg
                    -10
                    * np.pi,  # steer_rate, no more than 180 deg in a tenth of a second
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    max_steps * BIKE_SPEED * self.dt,  # x
                    max_steps * BIKE_SPEED * self.dt,  # y
                    FALL_THRESHOLD + np.pi / 8.0,  # lean
                    np.pi,  # heading
                    BIKE_SPEED + 0.2,  # x_dot
                    BIKE_SPEED + 0.2,  # y_dot
                    np.pi * 2,  # lean_dot
                    np.pi / 2,  # heading_dot
                    np.pi / 2,  # steer_angle
                    10 * np.pi,  # steer_rate
                ],
                dtype=np.float32,
            ),
        )

        self._state: np.ndarray = np.zeros(self.STATE_DIM, dtype=np.float32)
        self._step_count: int = 0

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
        lean_dot = 0.0
        heading = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        heading_dot = 0.0
        x = 0.0
        y = 0.0
        x_dot = BIKE_SPEED * np.cos(heading)
        y_dot = BIKE_SPEED * np.sin(heading)
        self._state = np.array(
            [x, y, lean, heading, x_dot, y_dot, lean_dot, heading_dot, 0.0, 0.0],
            dtype=np.float32,
        )
        self._state = self._dyn.reset(self._state)
        self._step_count = 0
        detailed_gymspace_validation(self.observation_space, self._state)
        return self._state.copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        detailed_gymspace_validation(self.action_space, action)
        steering = float(np.asarray(action, dtype=np.float32).flatten()[0])

        detailed_gymspace_validation(self.observation_space, self._state)
        _old_state = self._state
        self._state = self._dyn.step(self._state, steering, self.dt, self.wind_std)

        self._step_count += 1

        fallen = self.detect_falls and self._dyn.is_fallen()
        try:
            detailed_gymspace_validation(self.observation_space, self._state)
        except ValueError as e:
            print("Invalid state after step:", e)
            print("Full state:", self._state)
            fallen = True  # we don't need degenerate episodes to continue

        timeout = self._step_count >= self.max_steps

        terminated = fallen
        truncated = timeout and not fallen

        reward = self._compute_reward(fallen)

        if self.render_mode == "human":
            self.render()

        return self._state.copy(), reward, terminated, truncated, {}

    def render(self) -> None:
        # Main simulation runs in DIRECT mode for reproducible training.
        # 3D videos are produced in a dedicated evaluation renderer.
        return None

    def close(self) -> None:
        self._dyn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_reward(self, fallen: bool) -> float:
        lean = self._state[S.lean]
        heading = self._state[S.heading]
        if fallen:
            return -100.0
        progress = BIKE_SPEED * np.cos(heading)
        lean_penalty = -10.0 * lean**2
        heading_penalty = -5.0 * heading**2
        return float(progress + lean_penalty + heading_penalty)
