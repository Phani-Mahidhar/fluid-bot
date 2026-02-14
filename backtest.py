"""
backtest.py — Vectorized backtest and performance reporting.

Generates positions on the test set, shifts by +1 day to prevent look-ahead
bias, and computes strategy equity curve + key risk metrics.

Supports both single-asset and multi-asset stress-test plotting.
"""

import math
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from model import FluidGeometricNet


# ──────────────────────── Backtest Engine ─────────────────────
@torch.no_grad()
def run_backtest(
    model: FluidGeometricNet,
    test_loader: DataLoader,
    test_returns: np.ndarray,
    device: torch.device,
) -> dict:
    """
    Run the trained model on the test set and compute backtest metrics.

    Returns
    -------
    dict with keys: strategy_equity, benchmark_equity,
                    cumulative_return, sharpe, benchmark_sharpe, max_drawdown
    """
    model.eval()

    # --- Generate raw position signals ---
    signals = []
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        positions = model(X_batch)  # (B, 1)
        signals.append(positions.cpu().numpy().flatten())

    raw_signals = np.concatenate(signals)

    # --- Shift signals by +1 day (avoid look-ahead bias) ---
    # Position decided at close of day t is applied to return of day t+1
    shifted_signals = np.zeros_like(test_returns)
    shifted_signals[1:] = raw_signals[:-1]  # first day: flat (0)

    # --- Strategy returns ---
    strategy_returns = shifted_signals * test_returns

    # --- Cumulative equity curves ---
    strategy_equity = np.cumprod(1.0 + strategy_returns)
    benchmark_equity = np.cumprod(1.0 + test_returns)

    # --- Metrics ---
    cum_ret = strategy_equity[-1] - 1.0
    sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-8)) * math.sqrt(
        252
    )
    benchmark_sharpe = (test_returns.mean() / (test_returns.std() + 1e-8)) * math.sqrt(
        252
    )
    running_max = np.maximum.accumulate(strategy_equity)
    drawdowns = (strategy_equity - running_max) / running_max
    max_dd = drawdowns.min()

    return {
        "strategy_equity": strategy_equity,
        "benchmark_equity": benchmark_equity,
        "cumulative_return": cum_ret,
        "sharpe": sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "max_drawdown": max_dd,
    }


# ─────────────────── Multi-Asset Stress Plot ──────────────────
def plot_stress_test(
    results: dict[str, dict],
    save_path: str = "stress_test.png",
) -> None:
    """
    Generate a 5×1 subplot figure comparing strategy vs buy-and-hold
    for each asset in the stress test.

    Parameters
    ----------
    results : dict mapping asset name → backtest metrics dict
    """
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)

    for i, (asset, metrics) in enumerate(results.items()):
        ax = axes[i, 0]
        ax.plot(
            metrics["strategy_equity"],
            label="Fluid-Geometric Strategy",
            color="#2ecc71",
            linewidth=1.5,
        )
        ax.plot(
            metrics["benchmark_equity"],
            label="Buy & Hold",
            color="#3498db",
            linewidth=1.5,
            linestyle="--",
        )
        ax.set_title(
            f"{asset}  |  Strategy Sharpe: {metrics['sharpe']:.2f}  "
            f"vs  Benchmark Sharpe: {metrics['benchmark_sharpe']:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Cumulative Value ($1)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊  Stress test plot saved → {save_path}")
