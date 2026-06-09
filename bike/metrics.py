"""Observability and success-metrics tracking for bike RL.

Success metrics per episode:
  • distance     – total path length (m)
  • duration     – number of environment steps
  • mean_speed   – always BIKE_SPEED (logged for consistency)
  • fell         – whether the bike fell over
  • final_lean   – |lean| at the last step (rad)

Aggregate metrics (TrainingMetrics):
  • success_rate   – fraction without a fall
  • mean_distance  – average path length (m)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics from a single episode."""

    distance: float = 0.0
    duration: int = 0
    mean_speed: float = 0.0
    fell: bool = False
    final_lean: float = 0.0


@dataclass
class TrainingMetrics:
    """Accumulator for metrics across multiple episodes."""

    episodes: List[EpisodeMetrics] = field(default_factory=list)

    def add_episode(self, states: List[np.ndarray], fell: bool) -> None:
        """Record one episode.

        Parameters
        ----------
        states : list of state arrays  [x, y, lean, heading]
        fell   : whether the episode ended with a fall
        """
        if len(states) < 2:
            return

        positions = np.array([[s[0], s[1]] for s in states])
        distance = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())

        self.episodes.append(
            EpisodeMetrics(
                distance=distance,
                duration=len(states),
                mean_speed=0.0,   # speed is fixed; left as 0 for clarity
                fell=fell,
                final_lean=float(abs(states[-1][2])),
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

    def summary(self) -> str:
        return (
            f"n={len(self.episodes)}  "
            f"success={self.success_rate:.0%}  "
            f"dist={self.mean_distance:.1f}m"
        )
