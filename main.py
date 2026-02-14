"""
main.py — Production Entry Point: Nifty 500 Sniper Backtest.

1. Fetch top-N most liquid NSE stocks from Nifty 500
2. Train a NeuralODE model per stock (1.5yr train)
3. Run sniper backtest on test period (0.5yr)
4. Aggregate into High Conviction Portfolio
5. Print daily scan picks + console report
"""

from config import DEVICE, EPOCHS, PERIOD, TOP_N, CONFIDENCE_THRESHOLD
from data import get_nifty500_universe
from portfolio import (
    train_and_signal,
    backtest_portfolio,
    get_benchmark_equity,
    plot_portfolio,
)
from alerts import send_email_alert


def main() -> None:
    print("=" * 65)
    print("  Fluid-Geometric Sniper — Production Backtest")
    print(f"  Device: {DEVICE}  |  Period: {PERIOD}  |  Epochs: {EPOCHS}")
    print(f"  Universe: Nifty 500 (Top {TOP_N} by volume)")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}/10")
    print("=" * 65)

    # ── 1. Universe Selection ─────────────────────────────────
    print("\n[1/4] Building stock universe …")
    universe = get_nifty500_universe()
    print(f"       Final universe: {len(universe)} stocks\n")

    # ── 2. Train & Signal per stock ───────────────────────────
    print("[2/4] Training models …\n")
    stock_results: dict[str, dict] = {}
    today_signals: list[dict] = []

    for i, ticker in enumerate(universe, 1):
        print(f"  [{i:2d}/{len(universe)}] {ticker}", end=" … ")
        result = train_and_signal(ticker)
        if result is None:
            print("skipped")
            continue

        stock_results[ticker] = result
        action = result["latest_action"]
        confidence = min(round(abs(action) * 10, 1), 10.0)

        # Long-only: BUY only when action > threshold
        if action > CONFIDENCE_THRESHOLD / 10.0:
            direction = "BUY"
        else:
            direction = "—"

        status = f"action={action:+.3f}  conf={confidence}"
        if direction == "BUY":
            status += "  ← 🎯 BUY"
        print(status)

        # Collect today's high-conviction picks
        if confidence > CONFIDENCE_THRESHOLD:
            today_signals.append(
                {
                    "ticker": ticker,
                    "action": action,
                    "confidence": confidence,
                    "direction": direction,
                }
            )

    if not stock_results:
        print("\n⚠  No stocks processed successfully. Exiting.")
        return

    # ── 3. Portfolio Backtest ─────────────────────────────────
    print(
        f"\n[3/4] Building High Conviction Portfolio "
        f"(threshold > {CONFIDENCE_THRESHOLD}/10) …"
    )
    metrics = backtest_portfolio(stock_results)
    benchmark_equity = get_benchmark_equity(len(metrics["equity_curve"]))

    # ── 4. Report ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PRODUCTION BACKTEST REPORT")
    print("=" * 65)
    print(f"  Total Return         : {metrics['total_return']:+.2%}")
    print(f"  Annualised Sharpe    : {metrics['sharpe']:.2f}")
    print(f"  Max Drawdown         : {metrics['max_drawdown']:.2%}")
    print(
        f"  Sniper Accuracy      : {metrics['sniper_accuracy']:.1f}%"
        f"  ({metrics['sniper_trades_total']} trades)"
    )
    print(f"  Avg Stocks Held/Day  : {metrics['avg_stocks_held']:.1f}")
    print("=" * 65)

    # ── Plot ──────────────────────────────────────────────────
    plot_portfolio(metrics["equity_curve"], benchmark_equity)

    # ── Daily Scan Output: Tomorrow's Picks ───────────────────
    print("\n" + "=" * 65)
    print(f"  DAILY SCAN — Picks for Tomorrow (Confidence > {CONFIDENCE_THRESHOLD})")
    print("=" * 65)
    if today_signals:
        print(f"  {'Ticker':<15} {'Action':>8} {'Confidence':>11} {'Direction':>10}")
        print("  " + "─" * 48)
        for s in sorted(today_signals, key=lambda x: x["confidence"], reverse=True):
            print(
                f"  {s['ticker']:<15} {s['action']:>+7.4f}"
                f" {s['confidence']:>10}/10"
                f" {s['direction']:>10}"
            )
        send_email_alert(today_signals)
    else:
        print("  No stocks meet the confidence threshold today.")
    print("=" * 65)


if __name__ == "__main__":
    main()
