"""
daily_inference.py — Daily scan entry point with BUY + EXIT signals.

1. Gap-fills data via SQLite-backed MarketDataManager
2. Trains fresh models per stock
3. Generates tiered BUY signals AND explicit EXIT signals
4. Tracks holdings in a JSON file between runs
5. Runs full backtest with simulated portfolio results
"""

import json
import math
import os
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DEVICE,
    EPOCHS,
    LR,
    PERIOD,
    LOOKBACK,
    BATCH_SIZE,
    TRAIN_RATIO,
    TOP_N,
    BENCHMARK,
    BASE_REG_FACTOR,
    CONFIDENCE_THRESHOLD,
    SNIPER_THRESHOLD,
    VOL_WINDOW,
)
from data import get_nifty500_universe
from db import MarketDataManager
from model import FluidGeometricNet
from train import train
from alerts import send_email_alert

from torch.utils.data import Dataset, DataLoader


HOLDINGS_FILE = os.path.join(os.path.dirname(__file__), "holdings.json")


# ──────────────────────── Holdings Tracker ────────────────────
def load_holdings() -> dict:
    """Load current holdings from JSON file."""
    if os.path.exists(HOLDINGS_FILE):
        with open(HOLDINGS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_holdings(holdings: dict) -> None:
    """Persist holdings to JSON file."""
    with open(HOLDINGS_FILE, "w") as f:
        json.dump(holdings, f, indent=2)


# ──────────────────────── Dataset from DB features ────────────
class _WindowDataset(Dataset):
    def __init__(self, data: np.ndarray, lookback: int = LOOKBACK):
        self.data = data.astype(np.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        w = self.data[idx : idx + self.lookback]
        t = self.data[idx + self.lookback, 0]
        return torch.tensor(w), torch.tensor([t])


# ──────────────────────── Signal Labeling ─────────────────────
def label_signal(action: float, confidence: float) -> str:
    """Tiered signal label based on confidence (long-only)."""
    if action <= 0:
        return "—"
    if confidence > SNIPER_THRESHOLD:
        return "SNIPER BUY"
    if confidence > CONFIDENCE_THRESHOLD:
        return "STRONG BUY"
    return "—"


# ──────────────────────── Per-Stock Pipeline ──────────────────
def process_stock(ticker: str, mgr: MarketDataManager) -> dict | None:
    """Gap-fill data → train → generate test positions + latest signal."""
    # 1. Gap-fill
    try:
        new_rows = mgr.update_data(ticker)
    except Exception as e:
        print(f"    ⚠  {ticker}: data error — {e}")
        return None

    # 2. Load features from DB
    try:
        df = mgr.get_features(ticker)
    except Exception as e:
        print(f"    ⚠  {ticker}: feature error — {e}")
        return None

    values = df.values
    if len(values) < LOOKBACK + 20:
        print(f"    ⚠  {ticker}: insufficient data ({len(values)} rows)")
        return None

    # 3. Split
    split = int(len(values) * TRAIN_RATIO)
    train_data = values[:split]
    test_data = values[split - LOOKBACK :]

    train_ds = _WindowDataset(train_data)
    test_ds = _WindowDataset(test_data)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_returns = test_data[LOOKBACK:, 0]

    # 4. Train
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

    # 5. Test-set positions
    model.eval()
    positions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            pos = model(X_batch.to(DEVICE))
            positions.append(pos.cpu().numpy().flatten())
    test_positions = np.concatenate(positions)

    # 6. Latest signal
    latest = values[-LOOKBACK:].astype(np.float32)
    x = torch.tensor(latest).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        latest_action = model(x).cpu().item()

    return {
        "test_positions": test_positions,
        "test_returns": test_returns,
        "latest_action": latest_action,
        "db_rows": mgr.ticker_count(ticker),
        "new_rows": new_rows,
    }


# ──────────────────────── Portfolio Backtest ──────────────────
def backtest_portfolio(stock_results: dict[str, dict]) -> dict:
    """
    Long-only portfolio: hold when action > threshold, exit when it drops.
    Equal-weight across qualifying stocks.
    """
    threshold = CONFIDENCE_THRESHOLD / 10.0
    min_len = min(len(r["test_returns"]) for r in stock_results.values())

    daily_returns = []
    daily_n_held = []
    trades_profit = 0
    trades_total = 0
    entries = 0
    exits = 0
    prev_holdings = set()

    for day in range(1, min_len):
        contributing = []
        current_holdings = set()

        for ticker, res in stock_results.items():
            if day >= len(res["test_positions"]) or day >= len(res["test_returns"]):
                continue
            signal = res["test_positions"][day - 1]
            if signal > threshold:
                contributing.append(res["test_returns"][day])
                current_holdings.add(ticker)
                trades_total += 1
                if res["test_returns"][day] > 0:
                    trades_profit += 1

        # Count entries and exits
        entries += len(current_holdings - prev_holdings)
        exits += len(prev_holdings - current_holdings)
        prev_holdings = current_holdings

        port_ret = np.mean(contributing) if contributing else 0.0
        daily_returns.append(port_ret)
        daily_n_held.append(len(contributing))

    daily_returns = np.array(daily_returns)
    equity = np.cumprod(1.0 + daily_returns)

    total_ret = equity[-1] - 1.0 if len(equity) > 0 else 0.0
    sharpe = (
        ((daily_returns.mean() / (daily_returns.std() + 1e-8)) * math.sqrt(252))
        if len(daily_returns) > 0
        else 0.0
    )
    running_max = np.maximum.accumulate(equity)
    dd = ((equity - running_max) / running_max).min() if len(equity) > 0 else 0.0
    accuracy = (trades_profit / trades_total * 100) if trades_total > 0 else 0.0

    return {
        "equity": equity,
        "total_return": total_ret,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "accuracy": accuracy,
        "trades": trades_total,
        "entries": entries,
        "exits": exits,
        "avg_held": np.mean(daily_n_held) if daily_n_held else 0.0,
    }


# ──────────────────────── Benchmark ───────────────────────────
def get_benchmark_equity(mgr: MarketDataManager, n_days: int) -> np.ndarray:
    mgr.update_data(BENCHMARK)
    df = mgr.get_features(BENCHMARK)
    values = df.values
    split = int(len(values) * TRAIN_RATIO)
    bench_ret = values[split:, 0][:n_days]
    return np.cumprod(1.0 + bench_ret)


# ──────────────────────── Plotting ────────────────────────────
def plot_results(strategy, benchmark, path="production_backtest.png"):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(strategy, label="High Conviction Portfolio", color="#2ecc71", linewidth=2)
    ax.plot(
        benchmark,
        label="NIFTY 50 Benchmark",
        color="#3498db",
        linewidth=2,
        linestyle="--",
    )
    ax.set_title(
        f"Sniper Strategy vs NIFTY 50  (threshold > {CONFIDENCE_THRESHOLD}/10)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Trading Day (Test Period)")
    ax.set_ylabel("Cumulative Value (₹1 start)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n📊  Plot saved → {path}")


# ──────────────────────── Main Entry Point ────────────────────
def main() -> None:
    print("=" * 65)
    print("  Fluid-Geometric Sniper — Daily Inference + Backtest")
    print(f"  Device: {DEVICE}  |  Period: {PERIOD}  |  Epochs: {EPOCHS}")
    print(f"  Universe: Nifty 500 (Top {TOP_N})")
    print(
        f"  Thresholds: STRONG BUY > {CONFIDENCE_THRESHOLD}  |  "
        f"SNIPER BUY > {SNIPER_THRESHOLD}"
    )
    print("=" * 65)

    mgr = MarketDataManager()
    prev_holdings = load_holdings()

    # ── 1. Universe ───────────────────────────────────────────
    print("\n[1/5] Building stock universe …")
    universe = get_nifty500_universe()
    print(f"       {len(universe)} stocks selected.\n")

    # ── 2. Process each stock ─────────────────────────────────
    print("[2/5] Gap-filling data & training models …\n")
    results: dict[str, dict] = {}
    buy_signals: list[dict] = []
    exit_signals: list[dict] = []

    for i, ticker in enumerate(universe, 1):
        print(f"  [{i:2d}/{len(universe)}] {ticker}", end=" … ")
        res = process_stock(ticker, mgr)
        if res is None:
            print("skipped")
            # If we were holding this stock and it can't be processed, flag it
            if ticker in prev_holdings:
                exit_signals.append(
                    {
                        "ticker": ticker,
                        "action": 0.0,
                        "confidence": 0.0,
                        "direction": "EXIT ⚠️",
                        "reason": "Data unavailable",
                    }
                )
            continue

        results[ticker] = res
        action = res["latest_action"]
        conf = min(round(abs(action) * 10, 1), 10.0)
        label = label_signal(action, conf)
        suffix = f"  ← 🎯 {label}" if label != "—" else ""
        print(f"action={action:+.3f}  conf={conf}  db={res['db_rows']}{suffix}")

        if label != "—":
            buy_signals.append(
                {
                    "ticker": ticker,
                    "action": action,
                    "confidence": conf,
                    "direction": label,
                }
            )
        elif ticker in prev_holdings:
            # Was held, now below threshold → EXIT
            exit_signals.append(
                {
                    "ticker": ticker,
                    "action": action,
                    "confidence": conf,
                    "direction": "EXIT",
                    "reason": f"Confidence dropped to {conf}",
                }
            )

    if not results:
        print("\n⚠  No stocks processed. Exiting.")
        return

    # ── 3. Portfolio Backtest ─────────────────────────────────
    print(f"\n[3/5] Backtesting portfolio (threshold > {CONFIDENCE_THRESHOLD}/10) …")
    metrics = backtest_portfolio(results)
    bench = get_benchmark_equity(mgr, len(metrics["equity"]))

    # ── 4. Report ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PRODUCTION BACKTEST REPORT")
    print("=" * 65)
    print(f"  Total Return         : {metrics['total_return']:+.2%}")
    print(f"  Annualised Sharpe    : {metrics['sharpe']:.2f}")
    print(f"  Max Drawdown         : {metrics['max_drawdown']:.2%}")
    print(
        f"  Sniper Accuracy      : {metrics['accuracy']:.1f}%"
        f"  ({metrics['trades']} trades)"
    )
    print(f"  Total Entries        : {metrics['entries']}")
    print(f"  Total Exits          : {metrics['exits']}")
    print(f"  Avg Stocks Held/Day  : {metrics['avg_held']:.1f}")
    print("=" * 65)

    plot_results(metrics["equity"], bench)

    # ── 5. Daily Scan: BUY + EXIT Signals ─────────────────────
    print("\n" + "=" * 65)
    print("  DAILY SCAN — Actionable Signals for Tomorrow")
    print("=" * 65)

    # BUY signals
    if buy_signals:
        sorted_buys = sorted(buy_signals, key=lambda x: x["confidence"], reverse=True)
        print(f"\n  🟢 BUY / HOLD  ({len(buy_signals)} stocks)")
        print(f"  {'Ticker':<15} {'Action':>8} {'Conf':>6} {'Signal':<12}")
        print("  " + "─" * 45)
        for s in sorted_buys:
            marker = "★" if s["ticker"] not in prev_holdings else "↔"
            print(
                f"  {s['ticker']:<15} {s['action']:>+7.4f}"
                f" {s['confidence']:>5}/10"
                f" {s['direction']:<12} {marker}"
            )
        print(f"\n  ★ = New entry   ↔ = Continue holding")
    else:
        print("\n  🟢 No BUY signals today.")

    # EXIT signals
    if exit_signals:
        print(f"\n  🔴 EXIT  ({len(exit_signals)} stocks)")
        print(f"  {'Ticker':<15} {'Action':>8} {'Conf':>6} {'Reason':<30}")
        print("  " + "─" * 60)
        for s in exit_signals:
            print(
                f"  {s['ticker']:<15} {s['action']:>+7.4f}"
                f" {s['confidence']:>5}/10"
                f" {s['reason']:<30}"
            )
    elif prev_holdings:
        print(f"\n  🔴 No EXIT signals — all {len(prev_holdings)} holdings maintained.")
    else:
        print(f"\n  🔴 No previous holdings to exit.")

    print("\n" + "=" * 65)

    # ── Update holdings file ──────────────────────────────────
    new_holdings = {
        s["ticker"]: {
            "action": s["action"],
            "confidence": s["confidence"],
            "signal": s["direction"],
        }
        for s in buy_signals
    }
    save_holdings(new_holdings)

    n_new = len(set(new_holdings) - set(prev_holdings))
    n_exited = len(set(prev_holdings) - set(new_holdings))
    n_held = len(set(prev_holdings) & set(new_holdings))
    print(
        f"  Holdings updated: {len(new_holdings)} total"
        f" ({n_new} new, {n_held} held, {n_exited} exited)"
    )
    print(f"  Saved to: {HOLDINGS_FILE}")
    print("=" * 65)

    # ── Email alert (BUY + EXIT combined) ─────────────────────
    all_signals = buy_signals + exit_signals
    if all_signals:
        send_email_alert(all_signals)


if __name__ == "__main__":
    main()
