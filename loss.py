"""
loss.py — Differentiable Sharpe Ratio Loss with Volatility-Anchored Regularization.

Loss = -annualised_sharpe + dynamic_reg * mean(positions²)
where dynamic_reg = base_reg_factor * batch_volatility.
"""

import math
import torch
import torch.nn as nn

from config import BASE_REG_FACTOR


class DifferentiableSharpeRatioLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-8,
        annualise: bool = True,
        base_reg_factor: float = BASE_REG_FACTOR,
    ):
        super().__init__()
        self.eps = eps
        self.scale = math.sqrt(252) if annualise else 1.0
        self.base_reg_factor = base_reg_factor

    def forward(self, positions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        strategy_returns = positions * returns
        mean_r = strategy_returns.mean()
        std_r = strategy_returns.std() + self.eps
        sharpe = (mean_r / std_r) * self.scale

        # Volatility-anchored regularization
        batch_vol = torch.std(returns).detach()
        dynamic_reg = self.base_reg_factor * batch_vol
        position_penalty = dynamic_reg * torch.mean(positions**2)

        return -sharpe + position_penalty
