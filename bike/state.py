"""Named indices for the 10D latent bicycle state vector."""

from __future__ import annotations

from enum import IntEnum


class S(IntEnum):
    """Index names for [x, y, lean, heading, x_dot, y_dot, lean_dot, heading_dot, steer_angle, steer_rate]."""

    x = 0
    y = 1
    lean = 2
    heading = 3
    x_dot = 4
    y_dot = 5
    lean_dot = 6
    heading_dot = 7
    steer_angle = 8
    steer_rate = 9


S_RATE: tuple[int, ...] = (
    S.x_dot,
    S.y_dot,
    S.lean_dot,
    S.heading_dot,
    S.steer_rate,
)
