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

    daily_returns = []
    daily_n_held = []
    sniper_trades_profit = 0
    sniper_trades_total = 0

    for day in range(n_days):
        contributing_returns = []

        for ticker, res in stock_results.items():
            pos = res["test_positions"]
            rets = res["test_returns"]

            if day >= len(pos) or day >= len(rets):
                continue

            # Shift by +1: today's signal applies to tomorrow's return
            if day == 0:
                continue  # no signal for first day

            signal = pos[day - 1]  # yesterday's model output
            confidence = abs(signal)

            if confidence > threshold and signal > 0:
                # Long-only: only enter when action > threshold (BUY)
                contributing_returns.append(rets[day])
                sniper_trades_total += 1
                if rets[day] > 0:
                    sniper_trades_profit += 1

        if contributing_returns:
            # Equal-weight portfolio return
            port_ret = np.mean(contributing_returns)
        else:
            port_ret = 0.0  # flat / cash

        daily_returns.append(port_ret)
        daily_n_held.append(len(contributing_returns))

    daily_returns = np.array(daily_returns)
    equity_curve = np.cumprod(1.0 + daily_returns)

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
