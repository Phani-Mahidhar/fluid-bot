"""
db.py — SQLite-backed MarketDataManager with gap-filling.

Maintains a local cache of OHLCV data. On each call to update_data():
  1. Checks the latest date in the DB for the given ticker
  2. If empty → full 2yr fetch
  3. If stale → fetches only the missing candles since last date + 1 day
  4. Appends new rows to the DB
"""

import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config import DB_PATH, PERIOD, VOL_WINDOW


class MarketDataManager:
    """SQLite-backed data cache with smart gap-filling."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ticker TEXT NOT NULL,
                    date   TEXT NOT NULL,
                    open   REAL,
                    high   REAL,
                    low    REAL,
                    close  REAL,
                    volume REAL,
                    PRIMARY KEY (ticker, date)
                )
            """
            )
            conn.commit()

    # ──────────────────── Core: Gap-Filling Update ────────────
    def update_data(self, ticker: str) -> int:
        """
        Fetch missing data for `ticker` and append to DB.

        Returns the number of new rows inserted.
        """
        latest = self._get_latest_date(ticker)

        if latest is None:
            # DB is empty for this ticker — full historical fetch
            df = yf.download(
                ticker,
                period=PERIOD,
                auto_adjust=True,
                progress=False,
            )
        else:
            start_date = latest + timedelta(days=1)
            today = datetime.today().date()

            if start_date > today:
                return 0  # data is already up to date

            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )

        if df.empty:
            return 0

        return self._store(ticker, df)
        
    def update_data_batch(self, tickers: list[str], max_workers: int = 20) -> int:
        """
        Concurrent fetching for multiple tickers.
        Returns total new rows inserted.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        total_rows = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.update_data, t): t for t in tickers}
            for future in as_completed(futures):
                try:
                    rows = future.result()
                    total_rows += rows
                except Exception as e:
                    t = futures[future]
                    print(f"    ⚠ Error fetching '{t}': {e}")
        return total_rows

    # ──────────────────── Read Features from DB ───────────────
    def get_features(self, ticker: str) -> pd.DataFrame:
        """
        Read OHLCV from DB, compute log returns + rolling volatility.
        Returns DataFrame with columns [log_ret, vol].
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                "SELECT date, close FROM ohlcv WHERE ticker = ? ORDER BY date",
                conn,
                params=(ticker,),
                parse_dates=["date"],
                index_col="date",
            )

        if df.empty:
            raise ValueError(f"No data in DB for {ticker}")

        close = df["close"].squeeze()
        log_ret = np.log(close / close.shift(1))
        vol = log_ret.rolling(VOL_WINDOW).std()

        features = pd.DataFrame({"log_ret": log_ret, "vol": vol}, index=df.index)
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.dropna(inplace=True)
        return features

    # ──────────────────── Internals ───────────────────────────
    def _get_latest_date(self, ticker: str):
        """Return the latest date in DB for ticker, or None if empty."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM ohlcv WHERE ticker = ?",
                (ticker,),
            ).fetchone()

        if row and row[0]:
            return datetime.strptime(row[0], "%Y-%m-%d").date()
        return None

    def _store(self, ticker: str, df: pd.DataFrame) -> int:
        """Insert OHLCV rows into the DB. Returns count of new rows."""
        records = []
        for date, row in df.iterrows():
            dt = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)

            # Handle MultiIndex columns from yfinance
            open_val = (
                float(row["Open"].iloc[0])
                if hasattr(row["Open"], "iloc")
                else float(row["Open"])
            )
            high_val = (
                float(row["High"].iloc[0])
                if hasattr(row["High"], "iloc")
                else float(row["High"])
            )
            low_val = (
                float(row["Low"].iloc[0])
                if hasattr(row["Low"], "iloc")
                else float(row["Low"])
            )
            close_val = (
                float(row["Close"].iloc[0])
                if hasattr(row["Close"], "iloc")
                else float(row["Close"])
            )
            vol_val = (
                float(row["Volume"].iloc[0])
                if hasattr(row["Volume"], "iloc")
                else float(row["Volume"])
            )

            records.append(
                (ticker, dt, open_val, high_val, low_val, close_val, vol_val)
            )

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)",
                records,
            )
            conn.commit()

        return len(records)

    def ticker_count(self, ticker: str) -> int:
        """Return number of rows for a ticker."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        return row[0] if row else 0
