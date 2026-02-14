"""
data.py — Data fetching, geometric feature engineering, and dataset creation.

Features are a geometric representation of the market:
  - Log Returns:  ln(P_t / P_{t-1})
  - Volatility:   rolling std of log returns (VOL_WINDOW days)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import Dataset, DataLoader
from niftystocks import ns

from config import LOOKBACK, VOL_WINDOW, TRAIN_RATIO, BATCH_SIZE, TOP_N


# ──────────────────────── Universe Selection ──────────────────
def get_nifty500_universe(top_n: int | None = TOP_N) -> list[str]:
    """
    Fetch Nifty 500 tickers and optionally filter to the top N
    by average daily volume (most liquid).

    Set top_n = None to return the full Nifty 500 universe.
    """
    tickers = ns.get_nifty500_with_ns()

    if top_n is None:
        return tickers

    # Fetch summary volume for ranking
    print(f"  Ranking {len(tickers)} Nifty 500 stocks by volume …")
    vol_data = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            df = yf.download(
                batch,
                period="1mo",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if "Volume" in df.columns:
                vol = df["Volume"]
                if isinstance(vol, pd.DataFrame):
                    for t in vol.columns:
                        col = vol[t] if isinstance(t, str) else vol[t]
                        avg = col.mean()
                        if not np.isnan(avg):
                            ticker_str = (
                                t
                                if isinstance(t, str)
                                else t[-1] if isinstance(t, tuple) else str(t)
                            )
                            vol_data[ticker_str] = avg
                elif isinstance(vol, pd.Series):
                    avg = vol.mean()
                    if not np.isnan(avg):
                        vol_data[batch[0]] = avg
        except Exception:
            continue

    # Sort by volume descending, take top N
    sorted_tickers = sorted(vol_data, key=vol_data.get, reverse=True)
    selected = sorted_tickers[:top_n]
    print(f"  Selected top {len(selected)} by volume.")
    return selected


# ──────────────────────── Fetch & Engineer ────────────────────
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    """Download OHLCV, compute log returns + rolling volatility, drop bad rows."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    close = df["Close"].squeeze()
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(VOL_WINDOW).std()

    features = pd.DataFrame({"log_ret": log_ret, "vol": vol}, index=df.index)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)
    return features


# ──────────────────────── Torch Dataset ───────────────────────
class MarketDataset(Dataset):
    """
    Sliding-window dataset.
    X : (LOOKBACK, 2)   — window of [log_ret, vol]
    y : (1,)            — next-day log return
    """

    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data) - LOOKBACK

    def __getitem__(self, idx: int):
        window = self.data[idx : idx + LOOKBACK]
        target = self.data[idx + LOOKBACK, 0]
        return torch.tensor(window), torch.tensor([target])


# ──────────────────────── DataLoaders ─────────────────────────
def get_dataloaders(ticker: str, period: str):
    """
    Returns
    -------
    train_loader, test_loader : DataLoader
    test_returns : np.ndarray
    """
    df = fetch_data(ticker, period)
    values = df.values

    split = int(len(values) * TRAIN_RATIO)
    train_data, test_data = values[:split], values[split - LOOKBACK :]

    train_ds = MarketDataset(train_data)
    test_ds = MarketDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    test_returns = test_data[LOOKBACK:, 0]

    return train_loader, test_loader, test_returns
