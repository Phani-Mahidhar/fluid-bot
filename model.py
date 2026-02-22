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
from torchdiffeq import odeint

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

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # torchdiffeq requires the signature (t, y)
        return self.net(h)


# ──────────────────────── Full Model ─────────────────────────
class FluidGeometricNet(nn.Module):
    """
    End-to-end model:
      Input  (B, T, 2) → GRU → LayerNorm → ODEInt → Linear → Tanh → (B, 1)
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
        self.ode_func = ODEFunc(HIDDEN_DIM)
        # Integration time interval [0, 1]
        # Registered as a buffer so it moves to the correct device automatically
        self.register_buffer("integration_time", torch.tensor([0.0, 1.0]))

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

        # Integrate hidden state through learned ODE dynamics using dopri5
        # odeint returns outputs at all requested t, so shape is (len(t), B, HIDDEN_DIM)
        out = odeint(self.ode_func, h, self.integration_time, method="dopri5")
        
        # We only care about the final state at t=1
        h_final = out[-1]

        # Map to position
        return self.head(h_final)
