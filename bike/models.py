"""Three bicycle dynamics models with progressively richer architectures.

All models predict the **state delta** Δs given (state s, action a):
    predicted_Δs  ≈  s_{t+1} - s_t

Latent state  s : [x, y, lean, heading, x_dot, y_dot, lean_dot, heading_dot,
                   steer_angle, steer_rate]
                              (LATENT_DIM = 10)
Action       a : [steering_angle]                         (ACTION_DIM = 1)


Model 1 — SimpleLeanHeadingModel
    Predicts lean/heading and their rates (no position channels).
    Input  : [steering_torque, lean, heading, lean_dot, heading_dot,
              steer_angle, steer_rate]  →  7-D
    Output : [Δlean, Δheading, Δlean_dot, Δheading_dot]  →  4-D
    Architecture: Linear(7 → 16) → ReLU → Linear(16 → 4)

Model 2 — FullStateModel
    Predicts the full state delta.
    Input  : [steering, state]                →  11-D
    Output : [Δstate]                         →  10-D
    Architecture: Linear(11 → 32) → ReLU → Linear(32 → 32) → ReLU → Linear(32 → 10)

Model 3 — SigRegFullStateModel
    Same I/O as Model 2.  All hidden layers have dimension LATENT_DIM = 10
    ("latent variable dimensions").  SIGReg is applied to the output of
    the encoder (h2) during training to encourage those intermediate
    activations to follow an isotropic Gaussian distribution.

    The architecture is split into an encoder and a predictor.  The encoder
    is the part before SIGReg and only consumes state.  The predictor is the
    part after SIGReg and consumes [encoded_state, action].

    Architecture (5 linear layers):
        Encoder:   Linear(10 → 10) → ReLU → Linear(10 → 10) → ReLU → h2
        Predictor: Linear(11 → 10) → ReLU → Linear(10 → 10) → ReLU → Linear(10 → 10)

    SIGReg (Sketched Isotropic Gaussian Regularizer) — exact implementation
    from the LeWorldModel codebase (lucas-maes/le-wm).  It projects the
    batch of activations onto M random unit-norm directions, computes the
    Epps-Pulley characteristic-function test statistic along each projection
    (comparing the empirical CF against N(0,1)), and returns the average.

"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Protocol, Tuple, cast

from scipy.odr import Output
import torch
import torch.nn as nn

from bike.state import S, S_RATE

logger = logging.getLogger(__name__)

# ── constants dimension ─────────────────────────────────────────────
LATENT_DIM = 10  # [x, y, lean, heading, x_dot, y_dot, lean_dot, heading_dot, steer_angle, steer_rate]
ACTION_DIM = 1  # [steering_angle]
MODEL_DIFF_DT = 1.0 / 50.0  # Model-facing differential basis (0.02s)
DIFF_SCALE_TO_MODEL = MODEL_DIFF_DT  # per-second rate -> per-0.02s increment
ACTION_SCALE = 100.0  # normalize torque-like steering action to O(1)


def _wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + torch.pi, 2.0 * torch.pi) - torch.pi


def prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE with wrapped heading-delta error to avoid angle discontinuities."""
    err = pred - target
    heading_idx = 1 if pred.shape[-1] in (2, 4) else 3
    err = err.clone()
    err[:, heading_idx] = _wrap_to_pi_torch(err[:, heading_idx])

    sq = err.square()
    if pred.shape[-1] >= 8:
        weights = pred.new_ones(pred.shape[-1])
        sq = sq * weights

    return torch.mean(sq)


def _augment_state(state: torch.Tensor) -> torch.Tensor:
    """Lift 4D states to the 10D latent representation used by the models."""
    if state.shape[-1] == LATENT_DIM:
        return state
    if state.shape[-1] != 4:
        raise ValueError(
            f"Expected state dimension 4 or {LATENT_DIM}, got {state.shape[-1]}"
        )

    x_dot = 3.0 * torch.cos(state[..., S.heading : S.heading + 1])
    y_dot = 3.0 * torch.sin(state[..., S.heading : S.heading + 1])
    lean_dot = torch.zeros_like(state[..., S.lean : S.lean + 1])
    heading_dot = torch.zeros_like(state[..., S.heading : S.heading + 1])
    steer = torch.zeros_like(state[..., S.lean : S.lean + 1])
    steer_dot = torch.zeros_like(state[..., S.lean : S.lean + 1])
    return torch.cat(
        [state, x_dot, y_dot, lean_dot, heading_dot, steer, steer_dot], dim=-1
    )


def _scale_state_differentials_for_model(state: torch.Tensor) -> torch.Tensor:
    """Scale velocity/rate state channels to the model differential basis."""
    scaled = state.clone()
    scaled[..., S_RATE] = scaled[..., S_RATE] * DIFF_SCALE_TO_MODEL
    return scaled


def _scale_action_for_model(action: torch.Tensor) -> torch.Tensor:
    """Normalize steering action magnitude for stable optimization."""
    return action / ACTION_SCALE


class SubsetModel(Protocol):
    """Protocol for trainable bike models with subset-index metadata.

    Combines the model-specific prediction API with the nn.Module methods used
    by training, evaluation, and checkpointing code.
    """

    INPUT_INDICES: list[int]
    OUTPUT_INDICES: list[int]

    def to(self, *args: Any, **kwargs: Any) -> nn.Module: ...

    def train(self, mode: bool = True) -> nn.Module: ...

    def eval(self) -> nn.Module: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the state delta for the model's subset of input/output channels."""
        ...

    def loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the loss for the model's subset of output channels."""
        ...


# ═══════════════════════════════════════
# SIGReg — Sketched Isotropic Gaussian Regularizer
# Source: https://github.com/lucas-maes/le-wm/blob/main/module.py
# ══════════════════════════════════════════════════════════════════════════


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer.

    Encourages a batch of embeddings to follow an isotropic N(0,I)
    distribution by minimising the Epps-Pulley characteristic-function
    distance between the empirical distribution of one-dimensional random
    projections and the standard normal CF.

    Parameters
    ----------
    knots    : number of integration knots for the trapezoidal rule
    num_proj : number of random unit-norm projection directions M
    """

    def __init__(self, knots: int = 17, num_proj: int = 256) -> None:
        super().__init__()
        self.num_proj = num_proj

        #  [0, 3]  with trapezoidal weights × Gaussian windowt
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)  # N(0,1) characteristic function

        self.register_buffer("t", t)
        self.register_buffer("phi", window)  # target CF  exp(-t²/2)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute the SIGReg loss for a batch of embeddings.

        Parameters
        ----------
        proj : (T, B, D)   T = time-steps (use T=1 for a single minibatch),
                           B = batch size, D = embedding dimension.

        Returns
        -------
        Scalar loss: average Epps-Pulley statistic over projections and time.
        """
        # Sample M random unit-norm projection directions  (D, M)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        t = cast(torch.Tensor, self.t)
        phi = cast(torch.Tensor, self.phi)
        weights = cast(torch.Tensor, self.weights)

        # Projected coordinates  (T, B, M, knots)
        x_t = (proj @ A).unsqueeze(-1) * t

        # Epps-Pulley statistic: squared distance between empirical CF and N(0,1) CF
        # mean over B (batch) → (T, M, knots)
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()

        # Integrate over t with trapezoidal rule × Gaussian window; scale by B
        statistic = (err @ weights) * proj.size(-2)

        return statistic.mean()  # average over T and M


# ══════════════════════════════════════════
# Model 1 — SimpleLeanHeadingModel
# ══════════════════════════════════════════════════════════


class SimpleLeanHeadingModel(nn.Module):
    """Model 1: 3-layer network predicting lean and heading changes only.

    Useful for understanding whether position information is necessary.
    """

    INPUT_INDICES = [
        S.heading,
        S.lean,
        S.lean_dot,
        S.steer_angle,
        S.steer_rate,
    ]
    OUTPUT_INDICES = INPUT_INDICES
    L1_WEIGHT = 1e-1

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(self.INPUT_INDICES) + ACTION_DIM, len(self.INPUT_INDICES))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def pretty_print(self, zero_tol: float = 1e-3) -> str:
        """Format learned linear factors for each output channel.

        Parameters
        ----------
        zero_tol : absolute value threshold below which factors are omitted.

        Returns
        -------
        A human-readable multi-line string. The same text is also logged.
        """
        linear = cast(nn.Linear, self.net[0])
        weight = linear.weight.detach().cpu()
        bias = linear.bias.detach().cpu()

        input_names = ["action"] + [S(int(idx)).name for idx in self.INPUT_INDICES]
        output_names = [S(int(idx)).name for idx in self.OUTPUT_INDICES]

        lines = [f"SimpleLeanHeadingModel factors (|w| > {zero_tol:g}):"]
        for out_i, out_name in enumerate(output_names):
            terms: list[str] = []
            for in_i, in_name in enumerate(input_names):
                coeff = float(weight[out_i, in_i].item())
                if abs(coeff) > zero_tol:
                    terms.append(f"{coeff:+.6f}*{in_name}")

            bias_value = float(bias[out_i].item())
            if abs(bias_value) > zero_tol:
                terms.append(f"{bias_value:+.6f}")

            rhs = " ".join(terms) if terms else "0"
            lines.append(f"  {out_name} = {rhs}")

        message = "\n".join(lines)
        logger.info(message)
        return message

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict [Δlean, Δheading] from full state + action.

        Parameters
        ----------
        state  : (..., LATENT_DIM)  [x, y, lean, heading, x_dot, y_dot, lean_dot, heading_dot, steer_angle, steer_rate]
        action : (..., ACTION_DIM)  [steering]
        """
        state = _scale_state_differentials_for_model(_augment_state(state))
        lean_heading_dots = state[
            ...,
            self.INPUT_INDICES,
        ]
        action = _scale_action_for_model(action)
        return self(torch.cat([action, lean_heading_dots], dim=-1))

    def loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the loss for the model's subset of output channels."""
        pred_loss = prediction_loss(pred, target[..., self.OUTPUT_INDICES])
        l1_loss = torch.zeros_like(pred_loss)
        for param in self.parameters():
            l1_loss = l1_loss + param.abs().sum()
        l1_loss = l1_loss * self.L1_WEIGHT
        return pred_loss, l1_loss


# ══════════════════════════════════════════════════════════════════════
# Model 2 — FullStateModel
# ══════════════════════════════════════════════════════════════════════════


class FullStateModel(nn.Module):
    """Model 2: 3-layer network predicting the full state delta."""

    INPUT_DIM = ACTION_DIM + LATENT_DIM  # 9
    OUTPUT_DIM = LATENT_DIM  # 8

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = _scale_state_differentials_for_model(_augment_state(state))
        action = _scale_action_for_model(action)
        return self(torch.cat([action, state], dim=-1))


# ═══
# Model 3 — SigRegFullStateModel
# ═════════════════════


class SigRegFullStateModel(nn.Module):
    """Model 3: encoder/predictor split with SIGReg on encoder output.

    All hidden dimensions equal LATENT_DIM (8), i.e. "latent variable
    dimensions", as specified in goal.md.

    Architecture (5 linear layers):
        Encoder Layer 1: Linear(8 → 8) → ReLU  → h1
        Encoder Layer 2: Linear(8 → 8) → ReLU  → h2   ← SIGReg applied here
        Predictor Layer 1: Linear(9 → 8) → ReLU  → h3   [input: h2 + action]
        Predictor Layer 2: Linear(8 → 8) → ReLU  → h4
        Predictor Layer 3: Linear(8 → 8)         → Δstate
    """

    INPUT_DIM = ACTION_DIM + LATENT_DIM  # 9 (predictor input)
    OUTPUT_DIM = LATENT_DIM  # 8

    def __init__(
        self,
        sigreg_lambda: float = 0.1,
        sigreg_num_proj: int = 256,
        sigreg_knots: int = 17,
    ) -> None:
        super().__init__()
        self.sigreg_lambda = sigreg_lambda
        D = LATENT_DIM

        # Encoder (before SIGReg): state-only.
        self.encoder = nn.Sequential(
            nn.Linear(LATENT_DIM, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU(),
        )

        # Predictor (after SIGReg): action-conditioned.
        self.predictor = nn.Sequential(
            nn.Linear(D + ACTION_DIM, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, self.OUTPUT_DIM),
        )

        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state into latent h2 representation (pre-SIGReg)."""
        state = _scale_state_differentials_for_model(_augment_state(state))
        return self.encoder(state)

    def _predict_from_encoded(
        self, encoded_state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        action = _scale_action_for_model(action)
        x_in = torch.cat([encoded_state, action], dim=-1)
        return self.predictor(x_in)

    # ------------------------------------------------------------------
    # Core forward (used by CEM — no SIGReg overhead)
    # ------------------------------------------------------------------

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict Δstate from state + action (CEM-compatible, no SIGReg)."""
        h2 = self.encode_state(state)
        return self._predict_from_encoded(h2, action)

    # ------------------------------------------------------------------
    # Training forward — returns (Δstate, regularization_loss)
    # ------------------------------------------------------------------

    def forward_with_sigreg(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute delta and the SIGReg loss on encoder activations.

        Parameters
        ----------
        state  : (B, LATENT_DIM)
        action : (B, ACTION_DIM)

        Returns
        -------
        delta       : (B, OUTPUT_DIM)
        regularization_loss : scalar tensor
        """
        h2 = self.encode_state(state)  # (B, D)
        delta = self._predict_from_encoded(h2, action)

        # SIGReg expects (T, B, D); use T=1 for a single minibatch step
        regularization_loss = self.sigreg(h2.unsqueeze(0)) * self.sigreg_lambda

        return delta, regularization_loss
