"""Linearised Whipple bicycle dynamics (constant-speed model).

State vector  : [x, y, lean, heading]   (LATENT_DIM = 4)
Action scalar : steering angle δ  (radians, clipped to ±MAX_STEER)

Speed is **fixed** (not a state variable or action) — the bike always rides
at BIKE_SPEED m s⁻¹.  This keeps the latent dimension small and matches the
experimental setup described in goal.md.

Lean acceleration (linearised capsize / weave):
    lean_acc = (g / h) * lean  −  (v² / (h · L)) · δ

Heading rate (kinematic constraint):
    ψ̇ = v · tan(δ) / L

Wind: optional additive Gaussian noise applied to [lean, heading] at each
step, as used in the "two-neuron" paper to stress-test controllers.
"""

import numpy as np

GRAVITY: float = 9.81   # m s⁻²
COM_HEIGHT: float = 1.0  # m
WHEELBASE: float = 1.0   # m
BIKE_SPEED: float = 3.0  # m s⁻¹  — fixed, never changes
MAX_STEER: float = 0.5   # radians (~28.6°)

# Lean magnitude at which we declare the bike "fallen"
FALL_THRESHOLD: float = np.pi / 4  # 45°

LATENT_DIM: int = 4  # len([x, y, lean, heading])


def step(
    state: np.ndarray,
    steering: float,
    dt: float = 0.05,
    wind_std: float = 0.0,
) -> np.ndarray:
    """Advance the bicycle one time-step using semi-implicit Euler.

    Parameters
    ----------
    state    : [x, y, lean, heading]
    steering : steering angle in radians (will be clipped)
    dt       : time-step (s)
    wind_std : standard deviation of Gaussian noise added to lean & heading

    Returns
    -------
    next_state : [x, y, lean, heading]  float32
    """
    x, y, lean, heading = state.astype(float)
    delta = float(np.clip(steering, -MAX_STEER, MAX_STEER))
    v = BIKE_SPEED

    # Lean acceleration (linearised capsize dynamics)
    lean_acc = (GRAVITY / COM_HEIGHT) * lean - (v**2 / (COM_HEIGHT * WHEELBASE)) * delta

    # Heading rate (kinematic constraint)
    heading_rate = v * np.tan(delta) / WHEELBASE

    # Semi-implicit Euler integration
    new_lean = lean + lean_acc * dt * dt
    new_heading = heading + heading_rate * dt
    new_x = x + v * np.cos(heading) * dt
    new_y = y + v * np.sin(heading) * dt

    # Wind: Gaussian perturbations on lean and heading
    if wind_std > 0.0:
        new_lean += np.random.normal(0.0, wind_std)
        new_heading += np.random.normal(0.0, wind_std * 0.5)

    return np.array([new_x, new_y, new_lean, new_heading], dtype=np.float32)


def state_delta(
    state: np.ndarray,
    steering: float,
    dt: float = 0.05,
    wind_std: float = 0.0,
) -> np.ndarray:
    """Return Δstate = next_state − state."""
    return step(state, steering, dt, wind_std) - state.astype(np.float32)


def is_fallen(state: np.ndarray) -> bool:
    """Return True when the lean angle exceeds the fall threshold."""
    return bool(abs(state[2]) > FALL_THRESHOLD)
