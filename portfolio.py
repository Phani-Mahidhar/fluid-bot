"""
portfolio.py — Production portfolio-level backtest ("The Sniper").

Long-only strategy: hold stocks only when model confidence > threshold.
Aggregates per-stock signals into a High Conviction Portfolio and
compares against the Nifty benchmark.
"""

import math
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    EPOCHS,
    LR,
    PERIOD,
    LOOKBACK,
    BASE_REG_FACTOR,
    CONFIDENCE_THRESHOLD,
    TRANSACTION_COST,
    STOP_LOSS_PCT,
)
from data import fetch_data, get_dataloaders, MarketDataset
from model import FluidGeometricNet
from train import train


# ──────────────────────── Per-Stock Pipeline ──────────────────
def train_and_signal(ticker: str) -> dict | None:
    """
    Train a fresh model on one stock.  Returns:
      - test_positions : np.ndarray   raw model output on test set
      - test_returns   : np.ndarray   actual log returns on test set
      - latest_action  : float        signal on the very last data point
    """
    try:
        train_loader, test_loader, test_returns = get_dataloaders(ticker, PERIOD)
    except Exception as e:
        print(f"    ⚠  {ticker}: {e}")
        return None

    if len(test_returns) < 10:
        print(f"    ⚠  {ticker}: insufficient data, skipping.")
        return None

    model = FluidGeometricNet().to(DEVICE)
    train(
        model,
        train_loader,
        EPOCHS,
        LR,
        DEVICE,
        quiet=True,
        base_reg_factor=BASE_REG_FACTOR,
    )

    # ── Test-set positions ────────────────────────────────────
    model.eval()
    positions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            pos = model(X_batch.to(DEVICE))
            positions.append(pos.cpu().numpy().flatten())
    test_positions = np.concatenate(positions)

    # ── Latest signal (today) ─────────────────────────────────
    try:
        df = fetch_data(ticker, PERIOD)
        latest = df.values[-LOOKBACK:].astype(np.float32)
        x = torch.tensor(latest).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            latest_action = model(x).cpu().item()
    except Exception:
        latest_action = 0.0

    return {
        "test_positions": test_positions,
        "test_returns": test_returns,
        "latest_action": latest_action,
    }


# ──────────────────────── Portfolio Backtest ──────────────────
def backtest_portfolio(stock_results: dict[str, dict]) -> dict:
    """
    Build a High Conviction Portfolio:
    - On each test day, go Long only stocks with confidence > threshold
    - Equal-weight across qualifying stocks
    - Compare against benchmark

    Returns aggregated metrics dict.
    """
    threshold = CONFIDENCE_THRESHOLD / 10.0  # convert 0-10 scale to 0-1

    # Find the shortest test set length (align all stocks)
    min_len = min(len(r["test_returns"]) for r in stock_results.values())
    n_days = min_len

    # ── Vectorized Backtest ──────────────────────────────────
    n_stocks = len(stock_results)
    pos_matrix = np.zeros((n_days, n_stocks))
    ret_matrix = np.zeros((n_days, n_stocks))
    
    for i, (ticker, res) in enumerate(stock_results.items()):
        p = res["test_positions"][:n_days]
        r = res["test_returns"][:n_days]
        # Pad if slightly short
        if len(p) < n_days:
            p = np.pad(p, (0, n_days - len(p)), 'constant')
        if len(r) < n_days:
            r = np.pad(r, (0, n_days - len(r)), 'constant')
            
        pos_matrix[:, i] = p
        ret_matrix[:, i] = r

    # Shift positions by 1 day (trade at yesterday's close/today's open)
    pos_matrix = np.roll(pos_matrix, 1, axis=0)
    pos_matrix[0, :] = 0.0

    # 1. Action filtering
    # Go long only if confident
    trade_mask = (pos_matrix > threshold)
    
    # 2. Continuous Weighting
    # Allocate proportionally to confidence score across qualifying assets on each day
    raw_weights = np.where(trade_mask, pos_matrix, 0.0)
    row_sums = raw_weights.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.where(row_sums > 0, raw_weights / row_sums, 0.0)

    # 3. Stop-Loss Trigger
    # Cap daily return at STOP_LOSS_PCT
    capped_returns = np.where(ret_matrix < STOP_LOSS_PCT, STOP_LOSS_PCT, ret_matrix)

    # 4. Gross Portfolio Daily Returns
    gross_daily_returns = (weights * capped_returns).sum(axis=1)

    # 5. Transaction Costs
    # Change in absolute weight * cost
    turnover = np.abs(np.diff(weights, axis=0, prepend=0)).sum(axis=1)
    daily_returns = gross_daily_returns - (turnover * TRANSACTION_COST)

    equity_curve = np.cumprod(1.0 + daily_returns)
    daily_n_held = trade_mask.sum(axis=1)
    
    # Trading stats for reporting
    active_mask = (weights > 0).flatten()
    sniper_trades_total = np.sum(active_mask)
    if sniper_trades_total > 0:
        sniper_trades_profit = np.sum((weights * capped_returns).flatten()[active_mask] > 0)
    else:
        sniper_trades_profit = 0

    # ── Metrics ───────────────────────────────────────────────
    total_return = equity_curve[-1] - 1.0
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-8)) * math.sqrt(252)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = drawdowns.min()

    sniper_accuracy = (
        (sniper_trades_profit / sniper_trades_total * 100)
        if sniper_trades_total > 0
        else 0.0
    )

    return {
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "n_days": n_days,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "sniper_accuracy": sniper_accuracy,
        "sniper_trades_total": sniper_trades_total,
        "avg_stocks_held": np.mean(daily_n_held),
    }


# ──────────────────────── Benchmark Returns ──────────────────
def get_benchmark_equity(n_days: int) -> np.ndarray:
    """Fetch benchmark equity curve aligned to test period length."""
    from config import BENCHMARK, TRAIN_RATIO

    df = fetch_data(BENCHMARK, PERIOD)
    values = df.values
    split = int(len(values) * TRAIN_RATIO)
    bench_returns = values[split:, 0]  # log returns

    # Align length
    bench_returns = bench_returns[:n_days]
    return np.cumprod(1.0 + bench_returns)


# ──────────────────────── Plotting ────────────────────────────
def plot_portfolio(
    strategy_equity: np.ndarray,
    benchmark_equity: np.ndarray,
    save_path: str = "production_backtest.png",
) -> None:
    """Plot high conviction portfolio vs Nifty benchmark."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        strategy_equity, label="High Conviction Portfolio", color="#2ecc71", linewidth=2
    )
    ax.plot(
        benchmark_equity,
        label="NIFTY 50 Benchmark",
        color="#3498db",
        linewidth=2,
        linestyle="--",
    )

    ax.set_title(
        "Production Backtest: Sniper Strategy vs NIFTY 50",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Trading Day (Test Period)")
    ax.set_ylabel("Cumulative Value ($1 start)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊  Plot saved → {save_path}")
