"""
train.py — Training loop.

Trains FluidGeometricNet by minimising the DifferentiableSharpeRatioLoss.
"""

import torch
from torch.utils.data import DataLoader

from model import FluidGeometricNet
from loss import DifferentiableSharpeRatioLoss


def train(
    model: FluidGeometricNet,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    quiet: bool = False,
    base_reg_factor: float | None = None,
) -> list[float]:
    """
    Train the model and return per-epoch losses.

    Parameters
    ----------
    quiet           : suppress per-epoch prints
    base_reg_factor : override base regularization

    Returns
    -------
    losses : list[float]
    """
    kwargs = {} if base_reg_factor is None else {"base_reg_factor": base_reg_factor}
    criterion = DifferentiableSharpeRatioLoss(**kwargs)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        epoch_positions = []
        epoch_returns = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            positions = model(X_batch)
            epoch_positions.append(positions)
            epoch_returns.append(y_batch)

        all_positions = torch.cat(epoch_positions, dim=0)
        all_returns = torch.cat(epoch_returns, dim=0)

        loss = criterion(all_positions, all_returns)

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        losses.append(loss.item())
        if not quiet:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():+.4f}")

    return losses
