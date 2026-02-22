"""
loss.py — Differentiable Sortino Ratio Loss with Volatility-Anchored Regularization.

Loss = -annualised_sortino + dynamic_reg * mean(positions²)
where dynamic_reg = base_reg_factor * batch_volatility.
"""

import math
import torch
import torch.nn as nn

from config import BASE_REG_FACTOR


class DifferentiableSortinoRatioLoss(nn.Module):
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
        
        # Sortino isolates downside deviation
        # Mask out positive returns, replacing them with 0
        downside_returns = torch.where(
            strategy_returns < 0, strategy_returns, torch.tensor(0.0, device=strategy_returns.device)
        )
        downside_std = downside_returns.std() + self.eps
        sortino = (mean_r / downside_std) * self.scale

        # Volatility-anchored regularization
        batch_vol = torch.std(returns).detach()
        dynamic_reg = self.base_reg_factor * batch_vol
        position_penalty = dynamic_reg * torch.mean(positions**2)

        return -sortino + position_penalty
