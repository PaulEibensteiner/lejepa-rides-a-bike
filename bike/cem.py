"""Cross-Entropy Method (CEM) planner for model-predictive control.

At each planning call the planner:
  1. Maintains a Gaussian distribution over H-step action sequences.
  2. Samples N sequences, rolls them out through the learned dynamics model.
  3. Applies a per-step cost at *every* step of the rollout:
         cost_t = lean² + w_heading * (heading − target_heading)²
                - w_progress * cos(heading)   (forward-progress bonus)
     This drives the bike toward an upright, straight ride in the
     target direction.
  4. Selects the top-K (elite) sequences by total rollout cost.
  5. Refits the Gaussian (mean, std) to the elite sequences.
  6. Repeats for ``n_iters`` refinement iterations.
  7. Returns the first action of the converged mean sequence.

The planner is model-agnostic: any model implementing
    predict_delta(state_tensor, action_tensor) -> differential_tensor
works.  Models that only predict [dlean/dt, dheading/dt] (Model 1) are
handled transparently — only their reduced lean/heading state is propagated.
"""

from __future__ import annotations

import numpy as np
import torch

from bike.dynamics import BIKE_SPEED, FALL_THRESHOLD, MAX_STEER
from bike.models import MODEL_DIFF_DT
from bike.state import S, S_RATE

_DT = MODEL_DIFF_DT  # planner rollout step in model time (0.02s)
_INIT_VAR = 25.0
_INIT_STD = float(np.sqrt(_INIT_VAR))


def _angle_diff(angle: np.ndarray, target: float) -> np.ndarray:
    """Shortest signed angular difference angle-target in [-pi, pi)."""
    return (angle - target + np.pi) % (2.0 * np.pi) - np.pi


def _clamp_states(states: np.ndarray) -> np.ndarray:
    """Clamp imagined states to numerically stable ranges."""
    states = states.copy()
    states[:, S.lean] = np.clip(states[:, S.lean], -FALL_THRESHOLD, FALL_THRESHOLD)
    states[:, S.heading] = (states[:, S.heading] + np.pi) % (2.0 * np.pi) - np.pi
    states[:, S.x_dot] = np.clip(states[:, S.x_dot], -10.0, 10.0)
    states[:, S.y_dot] = np.clip(states[:, S.y_dot], -10.0, 10.0)
    states[:, S.lean_dot] = np.clip(states[:, S.lean_dot], -20.0, 20.0)
    states[:, S.heading_dot] = np.clip(states[:, S.heading_dot], -20.0, 20.0)
    if states.shape[1] > S.steer_angle:
        states[:, S.steer_angle] = np.clip(states[:, S.steer_angle], -1.2, 1.2)
    if states.shape[1] > S.steer_rate:
        states[:, S.steer_rate] = np.clip(states[:, S.steer_rate], -30.0, 30.0)
    return states


class CEMPlanner:
    """Cross-Entropy Method planner.

    Parameters
    ----------
    model          : learned dynamics model with .predict_delta method
    action_dim     : dimensionality of the action vector (1)
    horizon        : planning horizon H (number of look-ahead steps)
    n_samples      : candidate action-sequence population size N
    n_elite        : elite samples kept per iteration K
    n_iters        : CEM refinement iterations per plan() call
    action_bounds  : (low, high) clip bounds for steering
    dt             : simulation time-step (must match BikeEnv.dt)
    target_heading : desired heading (rad); None = ignore heading and use a
                     purely lean-based cost
    w_lean         : cost weight for lean deviation
    w_heading      : cost weight for heading deviation
    w_progress     : reward weight for forward progress
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        action_dim: int = 1,
        horizon: int = 200,
        n_samples: int = 200,
        n_elite: int = 20,
        n_iters: int = 5,
        action_bounds: tuple[float, float] = (-MAX_STEER, MAX_STEER),
        dt: float = _DT,
        min_std: float = 10.0,
        loss_mode: str = "direction_lean",
        target_heading: float | None = None,
        w_lean: float = 10.0,
        w_heading: float = 5.0,
        w_progress: float = 1.0,
    ) -> None:
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iters = n_iters
        self.action_lb = action_bounds[0]
        self.action_ub = action_bounds[1]
        self.dt = dt
        self.min_std = float(min_std)
        if loss_mode not in {"direction_lean", "lean_only"}:
            raise ValueError("loss_mode must be one of {'direction_lean', 'lean_only'}")
        self.loss_mode = loss_mode
        self.target_heading = target_heading
        self.w_lean = w_lean
        self.w_heading = w_heading
        self.w_progress = w_progress

        self._seq_dim = horizon * action_dim
        self._mean = np.zeros(self._seq_dim, dtype=np.float32)
        self._std = np.full(
            self._seq_dim, max(_INIT_STD, self.min_std), dtype=np.float32
        )
        self._model_to_planner_step = self.dt / MODEL_DIFF_DT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the warm-start distribution (call at each episode start)."""
        self._mean[:] = 0.0
        self._std[:] = max(_INIT_STD, self.min_std)

    def plan(self, current_state: np.ndarray) -> np.ndarray:
        """Return the optimal first action for *current_state*.

        Parameters
        ----------
        current_state : (LATENT_DIM,) = [x, y, lean, heading, x_dot, y_dot, lean_dot, heading_dot]

        Returns
        -------
        action : (action_dim,)
        """
        mean = self._mean.copy()
        std = self._std.copy()

        for _ in range(self.n_iters):
            noise = np.random.randn(self.n_samples, self._seq_dim).astype(np.float32)
            sequences = np.clip(mean + std * noise, self.action_lb, self.action_ub)

            costs = self._rollout_costs(current_state, sequences)

            elite_idx = np.argsort(costs)[: self.n_elite]
            elites = sequences[elite_idx]

            mean = elites.mean(axis=0)
            std = elites.std(axis=0)

        # Warm-start: shift sequence left by one step
        self._mean[: -self.action_dim] = mean[self.action_dim :]
        self._mean[-self.action_dim :] = 0.0
        std = np.maximum(std, self.min_std)
        self._std[:] = std

        return mean[: self.action_dim]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _rollout_costs(
        self, initial_state: np.ndarray, sequences: np.ndarray
    ) -> np.ndarray:
        """Cumulative cost for each action sequence.

        Cost is accumulated at *every* step of the rollout.

        Parameters
        ----------
        initial_state : (LATENT_DIM,)
        sequences     : (N, horizon * action_dim)

        Returns
        -------
        costs : (N,)
        """
        device = self._model_device()
        n = sequences.shape[0]
        states = np.tile(initial_state.astype(np.float32), (n, 1))
        total_costs = np.zeros(n, dtype=np.float32)

        state_t = torch.from_numpy(states).to(device)
        predict_fn = getattr(self.model, "predict_delta")

        for t in range(self.horizon):
            a_np = sequences[:, t * self.action_dim : (t + 1) * self.action_dim]
            a_t = torch.from_numpy(a_np).to(device)

            delta_t = predict_fn(state_t, a_t)
            # Model outputs are increments over MODEL_DIFF_DT; scale to planner dt.
            delta_np = delta_t.cpu().numpy() * self._model_to_planner_step

            if delta_np.shape[1] == states.shape[1] and states.shape[1] >= 10:
                # Full-state model: update rates then integrate positions/angles analytically.
                states[:, S_RATE] += delta_np[:, S_RATE]
                states[:, S.x] += states[:, S.x_dot] * self.dt
                states[:, S.y] += states[:, S.y_dot] * self.dt
                states[:, S.lean] += states[:, S.lean_dot] * self.dt
                states[:, S.heading] += states[:, S.heading_dot] * self.dt
                states[:, S.steer_angle] += states[:, S.steer_rate] * self.dt
            elif delta_np.shape[1] == states.shape[1]:
                # Backward-compatible path for older 8D models.
                states[:, 4:8] += delta_np[:, 4:8]
                states[:, 0:4] += states[:, 4:8] * self.dt
            elif delta_np.shape[1] == 4:
                # 4D output model: apply accelerations, analytically integrate positions
                states[:, S.lean_dot] += delta_np[:, 2]
                states[:, S.heading_dot] += delta_np[:, 3]
                states[:, S.lean] += states[:, S.lean_dot] * self.dt
                states[:, S.heading] += states[:, S.heading_dot] * self.dt
            else:
                # Lean/heading-only model (Model 1):
                # integrate only the reduced state this model actually represents.
                states[:, S.lean] += delta_np[:, 0]
                states[:, S.heading] += delta_np[:, 1]
                # Keep imagined rate channels in per-second units for consistency.
                states[:, S.lean_dot] = delta_np[:, 0] / self.dt
                states[:, S.heading_dot] = delta_np[:, 1] / self.dt

            states = _clamp_states(states)
            state_t = torch.from_numpy(states.astype(np.float32)).to(device)
            total_costs += self._step_cost(states)

        return total_costs

    def _step_cost(self, states: np.ndarray) -> np.ndarray:
        """Per-step cost for a batch of states (lower = better ride).

        A "perfect upright ride in a specified direction" has:
            lean = 0  and  heading = target_heading

        When ``target_heading`` is None the cost is purely lean-based.
        """
        lean = states[:, S.lean]
        lean_cost = self.w_lean * lean**2
        if self.target_heading is None or self.loss_mode == "lean_only":
            return lean_cost

        heading = states[:, S.heading]
        heading_err = _angle_diff(heading, self.target_heading)
        heading_cost = self.w_heading * heading_err**2
        progress_bonus = -self.w_progress * BIKE_SPEED * np.cos(heading)

        return lean_cost + heading_cost + progress_bonus

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
