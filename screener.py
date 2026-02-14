"""
screener.py — Fluid-Geometric Generalist Screener.

Scans Indian Equities (NSE) and Crypto assets, trains a per-asset
NeuralODE model with asset-type-aware regularization, and produces
a Buy/Sell/Hold signal with 0-10 confidence rating.
"""

import torch
import numpy as np
import pandas as pd

from config import (
    DEVICE,
    EPOCHS,
    LR,
    PERIOD,
    CRYPTO_ASSETS,
    EQUITY_ASSETS,
    REG_CRYPTO,
    REG_EQUITY,
    LOOKBACK,
)
from data import get_dataloaders, fetch_data, MarketDataset
from model import FluidGeometricNet
from train import train


# ──────────────────────── Asset Type Detection ────────────────
def detect_type(ticker: str) -> str:
    """Auto-detect 'crypto' or 'equity' from ticker suffix."""
    if ticker.upper().endswith("-USD"):
        return "crypto"
    return "equity"


# ──────────────────────── Signal Generation ───────────────────
@torch.no_grad()
def generate_signal(
    model: FluidGeometricNet, ticker: str, device: torch.device
) -> float:
    """
    Run the model on the latest LOOKBACK-day window to produce
    a position signal ∈ [-1, 1].
    """
    model.eval()
    df = fetch_data(ticker, PERIOD)
    latest_window = df.values[-LOOKBACK:].astype(np.float32)  # (LOOKBACK, 2)
    x = torch.tensor(latest_window).unsqueeze(0).to(device)  # (1, LOOKBACK, 2)
    action = model(x).cpu().item()
    return action


# ──────────────────────── Screener Loop ───────────────────────
def run_screener() -> pd.DataFrame:
    """Train per-asset model, generate signal, return results DataFrame."""
    all_tickers = CRYPTO_ASSETS + EQUITY_ASSETS
    rows: list[dict] = []

    print("=" * 65)
    print("  Fluid-Geometric Generalist Screener")
    print(f"  Device: {DEVICE}  |  Period: {PERIOD}  |  Epochs: {EPOCHS}")
    print(
        f"  Assets: {len(all_tickers)}  ({len(CRYPTO_ASSETS)} crypto, "
        f"{len(EQUITY_ASSETS)} equity)"
    )
    print("=" * 65)

    for i, ticker in enumerate(all_tickers, 1):
        asset_type = detect_type(ticker)
        base_reg = REG_CRYPTO if asset_type == "crypto" else REG_EQUITY

        print(
            f"\n[{i}/{len(all_tickers)}] {ticker} ({asset_type}, "
            f"base_reg={base_reg})"
        )

        # ── Fetch & train ─────────────────────────────────────
        try:
            train_loader, _, _ = get_dataloaders(ticker, PERIOD)
        except Exception as e:
            print(f"  ⚠  Skipped — {e}")
            continue

        model = FluidGeometricNet().to(DEVICE)
        train(
            model,
            train_loader,
            EPOCHS,
            LR,
            DEVICE,
            quiet=True,
            base_reg_factor=base_reg,
        )

        # ── Signal from latest data ──────────────────────────
        try:
            action = generate_signal(model, ticker, DEVICE)
        except Exception as e:
            print(f"  ⚠  Signal error — {e}")
            continue

        confidence = round(abs(action) * 10, 1)
        confidence = min(confidence, 10.0)

        if action > 0.02:
            direction = "BUY"
        elif action < -0.02:
            direction = "SELL"
        else:
            direction = "HOLD"

        print(
            f"  ✓ Action: {action:+.4f}  |  Confidence: {confidence}/10  "
            f"|  {direction}"
        )

        rows.append(
            {
                "Asset": ticker,
                "Type": asset_type.upper(),
                "Action Signal": round(action, 4),
                "Confidence (0-10)": confidence,
                "Direction": direction,
            }
        )

    # ── Build & sort results ──────────────────────────────────
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Confidence (0-10)", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────── Entry Point ─────────────────────────
def main() -> None:
    results = run_screener()

    print("\n" + "=" * 65)
    print("  SCREENER REPORT — Sorted by Confidence (High → Low)")
    print("=" * 65)
    if results.empty:
        print("  No results generated.")
    else:
        print(results.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
