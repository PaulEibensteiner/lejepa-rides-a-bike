"""Three bicycle dynamics models with progressively richer architectures.

All models predict the **state delta** Δs given (state s, action a):
    predicted_Δs  ≈  s_{t+1} − s_t

Fixed-speed latent state  s : [x, y, lean, heading]     (LATENT_DIM = 4)
Action                    a : [steering_angle]           (ACTION_DIM = 1)


Model 1 — SimpleLeanHeadingModel
    Predicts only lean and heading changes (no position).
    Input  : [steering, lean, heading]   →  3-D
    Output : [Δlean, Δheading]           →  2-D
    Architecture: Linear(3 → 16) → ReLU → Linear(16 → 2)

Model 2 — FullStateModel
    Predicts the full state delta.
    Input  : [steering, x, y, lean, heading]  →  5-D
    Output : [Δx, Δy, Δlean, Δheading]        →  4-D
    Architecture: Linear(5 → 32) → ReLU → Linear(32 → 4)

Model 3 — SigRegFullStateModel
    Same I/O as Model 2.  All hidden layers have dimension LATENT_DIM = 4
    ("latent variable dimensions").  SIGReg is applied to the output of
    the second layer (h2) during training to encourage those intermediate
    activations to follow an isotropic Gaussian distribution.

    Architecture (5 linear layers):
        Linear(5 → 4) → ReLU → h1
        Linear(4 → 4) → ReLU → h2   ← SIGReg applied here
        Linear(4 → 4) → ReLU → h3
        Linear(4 → 4) → ReLU → h4
        Linear(4 → 4) → output (Δstate)

    SIGReg (Sketched Isotropic Gaussian Regularizer) — exact implementation
    from the LeWorldModel codebase (lucas-maes/le-wm).  It projects the
    batch of activations onto M random unit-norm directions, computes the
    Epps-Pulley characteristic-function test statistic along each projection
    (comparing the empirical CF against N(0,1)), and returns the average.

"""

from __future__ import annotations

import torch
import torch.nn as nn

# ── constants dimension ─────────────────────────────────────────────
LATENT_DIM = 4   # [x, y, lean, heading]
ACTION_DIM = 1   # [steering_angle]


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
        window = torch.exp(-t.square() / 2.0)   # N(0,1) characteristic function

        self.register_buffer("t", t)
        self.register_buffer("phi", window)           # target CF  exp(-t²/2)
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

        # Projected coordinates  (T, B, M, knots)
        x_t = (proj @ A).unsqueeze(-1) * self.t

        # Epps-Pulley statistic: squared distance between empirical CF and N(0,1) CF
        # mean over B (batch) → (T, M, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()

        # Integrate over t with trapezoidal rule × Gaussian window; scale by B
        statistic = (err @ self.weights) * proj.size(-2)

        return statistic.mean()   # average over T and M


# ══════════════════════════════════════════
# Model 1 — SimpleLeanHeadingModel
# ══════════════════════════════════════════════════════════

class SimpleLeanHeadingModel(nn.Module):
    """Model 1: 2-layer network predicting lean and heading changes only.

    Useful for understanding whether position information is necessary.
    """

    INPUT_DIM = ACTION_DIM + 2   # [steering, lean, heading]
    OUTPUT_DIM = 2               # [Δlean, Δheading]

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, self.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_delta(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict [Δlean, Δheading] from full state + action.

        Parameters
        ----------
        state  : (..., LATENT_DIM)  [x, y, lean, heading]
        action : (..., ACTION_DIM)  [steering]
        """
        lean = state[..., 2:3]
        heading = state[..., 3:4]
        return self(torch.cat([action, lean, heading], dim=-1))


# ══════════════════════════════════════════════════════════════════════
# Model 2 — FullStateModel
# ══════════════════════════════════════════════════════════════════════════

class FullStateModel(nn.Module):
    """Model 2: 2-layer network predicting the full state delta."""

    INPUT_DIM = ACTION_DIM + LATENT_DIM   # 5
    OUTPUT_DIM = LATENT_DIM               # 4

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, self.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_delta(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self(torch.cat([action, state], dim=-1))


# ═══
# Model 3 — SigRegFullStateModel
# ═════════════════════

class SigRegFullStateModel(nn.Module):
    """Model 3: 4-hidden-layer network with SIGReg on layer-2 activations.

    All hidden dimensions equal LATENT_DIM (4), i.e. "latent variable
    dimensions", as specified in goal.md.

    Architecture (5 linear layers):
        Layer 1: Linear(5 → 4) → ReLU  → h1
        Layer 2: Linear(4 → 4) → ReLU  → h2   ← SIGReg applied here
        Layer 3: Linear(4 → 4) → ReLU  → h3
        Layer 4: Linear(4 → 4) → ReLU  → h4
        Layer 5: Linear(4 → 4)         → Δstate
    """

    INPUT_DIM = ACTION_DIM + LATENT_DIM   # 5
    OUTPUT_DIM = LATENT_DIM               # 4

    def __init__(
        self,
        sigreg_lambda: float = 0.1,
        sigreg_num_proj: int = 256,
        sigreg_knots: int = 17,
    ) -> None:
        super().__init__()
        self.sigreg_lambda = sigreg_lambda
        D = LATENT_DIM

        # Layers 1–2 (SIGReg is applied after layer 2)
        self.layers_12 = nn.Sequential(
            nn.Linear(self.INPUT_DIM, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU(),
        )

        # Layers 3–5
        self.layers_345 = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, self.OUTPUT_DIM),
        )

        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    # ------------------------------------------------------------------
    # Core forward (used by CEM — no SIGReg overhead)
    # ------------------------------------------------------------------

    def predict_delta(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict Δstate from state + action (CEM-compatible, no SIGReg)."""
        x_in = torch.cat([action, state], dim=-1)
        h2 = self.layers_12(x_in)
        return self.layers_345(h2)

    # ------------------------------------------------------------------
    # Training forward — returns (Δstate, sigreg_loss)
    # ------------------------------------------------------------------

    def forward_with_sigreg(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute delta and the SIGReg loss on layer-2 activations.

        Parameters
        ----------
        state  : (B, LATENT_DIM)
        action : (B, ACTION_DIM)

        Returns
        -------
        delta       : (B, OUTPUT_DIM)
        sigreg_loss : scalar tensor
        """
        x_in = torch.cat([action, state], dim=-1)
        h2 = self.layers_12(x_in)           # (B, D)
        delta = self.layers_345(h2)

        # SIGReg expects (T, B, D); use T=1 for a single minibatch step
        sigreg_loss = self.sigreg(h2.unsqueeze(0)) * self.sigreg_lambda

        return delta, sigreg_loss
