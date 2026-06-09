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
    predict_delta(state_tensor, action_tensor) -> delta_tensor
works.  Models that only predict [Δlean, Δheading] (Model 1) are
handled transparently — x and y are propagated analytically.
"""

from __future__ import annotations

import numpy as np
import torch

from bike.dynamics import BIKE_SPEED, WHEELBASE

_DT = 0.05  # must match BikeEnv.dt


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
    target_heading : desired heading (rad); 0.0 = ride straight ahead
    w_lean         : cost weight for lean deviation
    w_heading      : cost weight for heading deviation
    w_progress     : reward weight for forward progress
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        action_dim: int = 1,
        horizon: int = 20,
        n_samples: int = 200,
        n_elite: int = 20,
        n_iters: int = 5,
        action_bounds: tuple[float, float] = (-0.5, 0.5),
        dt: float = _DT,
        target_heading: float = 0.0,
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
        self.target_heading = target_heading
        self.w_lean = w_lean
        self.w_heading = w_heading
        self.w_progress = w_progress

        self._seq_dim = horizon * action_dim
        self._mean = np.zeros(self._seq_dim, dtype=np.float32)
        self._std = np.full(self._seq_dim, 0.3, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the warm-start distribution (call at each episode start)."""
        self._mean[:] = 0.0
        self._std[:] = 0.3

    def plan(self, current_state: np.ndarray) -> np.ndarray:
        """Return the optimal first action for *current_state*.

        Parameters
        ----------
        current_state : (LATENT_DIM,) = [x, y, lean, heading]

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
            std = elites.std(axis=0) + 1e-5

        # Warm-start: shift sequence left by one step
        self._mean[: -self.action_dim] = mean[self.action_dim :]
        self._mean[-self.action_dim :] = 0.0
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

        for t in range(self.horizon):
            a_np = sequences[:, t * self.action_dim : (t + 1) * self.action_dim]
            a_t = torch.from_numpy(a_np).to(device)

            delta_t = self.model.predict_delta(state_t, a_t)
            delta_np = delta_t.cpu().numpy()

            if delta_np.shape[1] == states.shape[1]:
                # Full-state model: add delta directly
                states = states + delta_np
            else:
                # Lean/heading-only model (Model 1):
                # update lean and heading from the model; propagate x, y analytically
                states[:, 2] += delta_np[:, 0]   # Δlean
                states[:, 3] += delta_np[:, 1]   # Δheading
                states[:, 0] += BIKE_SPEED * np.cos(states[:, 3]) * self.dt
                states[:, 1] += BIKE_SPEED * np.sin(states[:, 3]) * self.dt

            state_t = torch.from_numpy(states.astype(np.float32)).to(device)
            total_costs += self._step_cost(states)

        return total_costs

    def _step_cost(self, states: np.ndarray) -> np.ndarray:
        """Per-step cost for a batch of states (lower = better ride).

        A "perfect upright ride in a specified direction" has:
            lean = 0  and  heading = target_heading
        """
        lean = states[:, 2]
        heading = states[:, 3]
        heading_err = heading - self.target_heading

        lean_cost = self.w_lean * lean**2
        heading_cost = self.w_heading * heading_err**2
        progress_bonus = -self.w_progress * BIKE_SPEED * np.cos(heading)

        return lean_cost + heading_cost + progress_bonus

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
