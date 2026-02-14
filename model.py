"""
model.py — Fluid-Geometric Network.

Architecture
============
1. GRU Encoder   → compresses a rolling window into a hidden state h.
2. Neural ODE    → integrates h forward in "time" via a learned ODE dh/dt = f(h)
                   using a hand-written RK4 solver (no torchdiffeq dependency).
3. Action Head   → maps evolved state to a position ∈ [-1, 1].
"""

import torch
import torch.nn as nn

from config import INPUT_DIM, HIDDEN_DIM, GRU_LAYERS, ODE_STEPS


# ──────────────────────── ODE Dynamics ────────────────────────
class ODEFunc(nn.Module):
    """
    Learnable vector field  dh/dt = f(h).

    A lightweight two-layer MLP with Tanh activation models the
    continuous-time dynamics of the hidden state.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ──────────────────────── RK4 Solver ─────────────────────────
class NeuralODE(nn.Module):
    """
    Custom 4th-order Runge-Kutta integrator.

    Given an initial state h₀ the solver computes:
        k₁ = f(hₙ)
        k₂ = f(hₙ + dt/2 · k₁)
        k₃ = f(hₙ + dt/2 · k₂)
        k₄ = f(hₙ + dt   · k₃)
        h_{n+1} = hₙ + (dt / 6)(k₁ + 2k₂ + 2k₃ + k₄)

    The integration is performed over a unit interval [0, 1]
    split into `n_steps` sub-steps.
    """

    def __init__(self, func: ODEFunc, n_steps: int = ODE_STEPS):
        super().__init__()
        self.func = func
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        for _ in range(self.n_steps):
            k1 = self.func(h)
            k2 = self.func(h + 0.5 * dt * k1)
            k3 = self.func(h + 0.5 * dt * k2)
            k4 = self.func(h + dt * k3)
            h = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return h


# ──────────────────────── Full Model ─────────────────────────
class FluidGeometricNet(nn.Module):
    """
    End-to-end model:
      Input  (B, T, 2) → GRU → LayerNorm → NeuralODE → Linear → Tanh → (B, 1)
    """

    def __init__(self):
        super().__init__()
        # --- Encoder: State-Space Model proxy ---
        self.gru = nn.GRU(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=GRU_LAYERS,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(HIDDEN_DIM)

        # --- Core dynamics ---
        self.ode = NeuralODE(ODEFunc(HIDDEN_DIM))

        # --- Action head ---
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 1),
            nn.Tanh(),  # clamps to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, LOOKBACK, INPUT_DIM)

        Returns
        -------
        positions : (B, 1)  ∈ [-1, 1]
        """
        # Encode the window into a hidden state
        _, h_n = self.gru(x)  # h_n: (num_layers, B, HIDDEN_DIM)
        h = h_n[-1]  # take last layer: (B, HIDDEN_DIM)
        h = self.layer_norm(h)

        # Integrate hidden state through learned ODE dynamics
        h = self.ode(h)

        # Map to position
        return self.head(h)
