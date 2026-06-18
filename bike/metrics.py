"""Observability and success-metrics tracking for bike RL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


def _angle_diff(angle: np.ndarray, target: float) -> np.ndarray:
    """Shortest signed angular difference angle-target in [-pi, pi)."""
    return (angle - target + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class EpisodeMetrics:
    """Metrics from a single episode."""

    distance: float = 0.0
    duration: int = 0
    mean_speed: float = 0.0
    fell: bool = False
    final_lean: float = 0.0
    rms_lean: float = 0.0
    rms_heading_error: float = 0.0
    path_length: float = 0.0
    survived: bool = False
    states: list[list[float]] = field(default_factory=list)
    actions: list[list[float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "distance": self.distance,
            "duration": self.duration,
            "mean_speed": self.mean_speed,
            "fell": self.fell,
            "final_lean": self.final_lean,
            "rms_lean": self.rms_lean,
            "rms_heading_error": self.rms_heading_error,
            "path_length": self.path_length,
            "survived": self.survived,
        }


@dataclass
class TrainingMetrics:
    """Accumulator for metrics across multiple episodes."""

    episodes: List[EpisodeMetrics] = field(default_factory=list)

    def add_episode(
        self,
        states: List[np.ndarray],
        fell: bool,
        actions: List[np.ndarray] | None = None,
        keep_trajectory: bool = False,
        target_heading: float = 0.0,
    ) -> None:
        """Record one episode.

        Parameters
        ----------
        states : list of state arrays  [x, y, lean, heading, x_dot, y_dot, lean_dot, heading_dot]
        fell   : whether the episode ended with a fall
        """
        if len(states) < 2:
            return

        states_arr = np.asarray(states, dtype=np.float32)
        positions = states_arr[:, :2]
        distance = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        lean = states_arr[:, 2]
        heading = states_arr[:, 3]

        episode_actions: list[list[float]]
        if actions is None:
            episode_actions = []
        else:
            episode_actions = np.asarray(actions, dtype=np.float32).tolist()

        episode_states = states_arr.tolist() if keep_trajectory else []

        self.episodes.append(
            EpisodeMetrics(
                distance=distance,
                duration=len(states_arr),
                mean_speed=0.0,
                fell=fell,
                final_lean=float(abs(states_arr[-1, 2])),
                rms_lean=float(np.sqrt(np.mean(np.square(lean)))),
                rms_heading_error=float(
                    np.sqrt(np.mean(np.square(_angle_diff(heading, target_heading))))
                ),
                path_length=distance,
                survived=not fell,
                states=episode_states,
                actions=episode_actions,
            )
        )

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if not e.fell) / len(self.episodes)

    @property
    def mean_distance(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.distance for e in self.episodes]))

    @property
    def mean_rms_lean(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.rms_lean for e in self.episodes]))

    @property
    def mean_rms_heading_error(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.rms_heading_error for e in self.episodes]))

    @property
    def mean_survival_steps(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.duration for e in self.episodes]))

    def to_summary_dict(self) -> dict:
        return {
            "num_episodes": len(self.episodes),
            "success_rate": self.success_rate,
            "mean_distance": self.mean_distance,
            "mean_rms_lean": self.mean_rms_lean,
            "mean_rms_heading_error": self.mean_rms_heading_error,
            "mean_survival_steps": self.mean_survival_steps,
        }

    def episodes_as_dicts(self) -> list[dict]:
        return [e.as_dict() for e in self.episodes]

    def summary(self) -> str:
        return (
            f"n={len(self.episodes)}  "
            f"success={self.success_rate:.0%}  "
            f"dist={self.mean_distance:.1f}m"
        )
