# Fluid-Geometric Trading System

> A Neural ODE–based algorithmic trading system for Indian NSE equities.  
> Trains per-stock models on geometric market features, generates tiered BUY signals, and runs production-grade backtests.

---

## Quick Start

```bash
# 1. Clone & setup
cd fluid-trader
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Launch the web dashboard
python app.py
# → Open http://localhost:5000

# 3. Or run the daily inference via CLI
python daily_inference.py
```

---

## Project Structure

```
fluid-trader/
│
├── config.py             # All hyperparameters + device detection
├── db.py                 # SQLite data cache with gap-filling
├── data.py               # Nifty 500 universe + feature engineering
├── model.py              # GRU → Neural ODE (RK4) → Tanh action head
├── loss.py               # Differentiable Sharpe Ratio + vol-anchored reg
├── train.py              # Training loop with gradient clipping
│
├── daily_inference.py    # ★ Production entry — daily scan + backtest
├── main.py               # Stress-test backtest mode
├── screener.py           # Quick signal screener
│
├── portfolio.py          # Portfolio-level backtest aggregation
├── alerts.py             # Email alert placeholder (HTML formatter)
├── backtest.py           # Per-stock backtest + plotting utilities
│
├── app.py                # ★ Flask web dashboard server
├── templates/index.html  # Dashboard UI
├── static/style.css      # Dark glassmorphism theme
├── static/app.js         # Frontend logic + live polling
│
├── requirements.txt
├── holdings.json         # Auto-managed current positions
└── market_data.db        # Auto-created SQLite cache
```

---

## Web Dashboard

```bash
python app.py
# → http://localhost:5000
```

The dashboard provides a visual interface for the entire system:

| Panel | Purpose |
|-------|--------|
| **⚡ Run Inference** | Trigger daily or monthly scans with live progress bar |
| **📊 My Positions** | View, add, or remove holdings. Manual entries take ticker + direction only |
| **🎯 Bot Suggestions** | Signal cards sorted by confidence — SNIPER BUY / STRONG BUY / EXIT |
| **🔥 Trending** | All scanned stocks ranked by confidence with visual bars |

**Running inference from the UI:**
- Click **Daily Scan** → trains on 2yr data, scans top 150 stocks
- Click **Monthly Scan** → trains on 5yr data for longer-term signals
- Progress updates live in the progress bar and log box
- Results auto-populate the Suggestions and Trending panels

**Managing positions:**
- Click **+ Add Position** → enter ticker (e.g. `RELIANCE.NS`) and direction (LONG/SHORT)
- Confidence is computed by the model, not manually entered
- Click **✕** to remove a position
- Bot-managed and manual positions are tracked separately

---

## CLI Entry Points

### 1. `daily_inference.py` — Production Daily Scan

The primary entry point. Does everything end-to-end:

```bash
python daily_inference.py
```

**What it does:**
1. Fetches top 150 most liquid Nifty 500 stocks (by volume)
2. Gap-fills missing OHLCV data into SQLite (smart — only fetches new candles)
3. Trains a fresh NeuralODE model per stock (50 epochs)
4. Runs portfolio backtest on the 6-month test period
5. Prints tomorrow's BUY picks with tiered confidence labels
6. Triggers email alert placeholder

**Output includes:**
- Portfolio return, Sharpe, max drawdown, sniper accuracy
- Equity curve plot (`production_backtest.png`)
- **🟢 BUY signals** with tiered labels (SNIPER BUY / STRONG BUY)
- **🔴 EXIT signals** when holdings drop below threshold
- Holdings tracked in `holdings.json` between runs

---

## When to Buy & When to Sell

### Buy Rules (Long-Only)

| Model Action | Confidence | Signal |
|-------------|-----------|--------|
| `> 0.6` | **> 6/10** | 🎯 **SNIPER BUY** — highest conviction |
| `> 0.3` | **> 3/10** | **STRONG BUY** — enter long |
| `≤ 0.3` | ≤ 3/10 | No trade — stay in cash |

### Sell Rules (Exit)

You **exit** a position when:

1. **Confidence drops below threshold** — The model's conviction in the stock has weakened. If you bought at confidence 7 and it drops to 2, exit at next open.
2. **Action flips negative** — The model flips bearish. Exit immediately.
3. **Stock becomes unavailable** — Data error or delisting. Exit flagged with ⚠️.

The system tracks this automatically via `holdings.json`:

```
Day 1:  INFY.NS  conf=9.8  →  🟢 SNIPER BUY  (★ New entry)
Day 2:  INFY.NS  conf=7.2  →  🟢 SNIPER BUY  (↔ Continue holding)
Day 3:  INFY.NS  conf=2.1  →  🔴 EXIT         (Confidence dropped to 2.1)
```

### Holdings File

`holdings.json` is auto-managed — it persists your current positions between runs:

```json
{
  "INFY.NS": {"action": 0.998, "confidence": 10.0, "signal": "SNIPER BUY"},
  "IDBI.NS": {"action": 0.995, "confidence": 10.0, "signal": "SNIPER BUY"}
}
```

Each run compares today's signals against this file to generate EXIT alerts.

---

### 2. `main.py` — Stress-Test Backtest

Runs per-stock backtests with individual equity curves:

```bash
python main.py
```

---

### 3. `screener.py` — Quick Signal Screener

Lightweight scan — trains and prints a signal table without portfolio aggregation:

```bash
python screener.py
```

---

## Configuration Reference

All tunable parameters live in **`config.py`**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PERIOD` | `"2y"` | Historical data window |
| `LOOKBACK` | `30` | Rolling window (trading days) |
| `VOL_WINDOW` | `20` | Volatility calculation window |
| `TRAIN_RATIO` | `0.75` | Train/test split (75% train) |
| `TOP_N` | `150` | Top N liquid stocks from Nifty 500. Set `None` for full 500 |
| `BENCHMARK` | `"^NSEI"` | Benchmark index for comparison |
| `EPOCHS` | `50` | Training epochs per stock |
| `LR` | `1e-3` | Adam learning rate |
| `BATCH_SIZE` | `128` | DataLoader batch size |
| `BASE_REG_FACTOR` | `0.01` | Regularization strength (lower = bolder signals) |
| `ODE_STEPS` | `10` | RK4 integration sub-steps |
| **`CONFIDENCE_THRESHOLD`** | **`3`** | **Min confidence to enter a trade (0-10)** |
| **`SNIPER_THRESHOLD`** | **`6`** | **Min confidence for "SNIPER BUY" label** |

### Tuning the Thresholds

```
Confidence > SNIPER_THRESHOLD  →  🎯 SNIPER BUY  (highest conviction)
Confidence > CONFIDENCE_THRESHOLD  →  STRONG BUY
Confidence ≤ CONFIDENCE_THRESHOLD  →  No trade (cash)
```

| Preset | `CONFIDENCE_THRESHOLD` | `SNIPER_THRESHOLD` | Behaviour |
|--------|----------------------|-------------------|-----------|
| Aggressive | 1 | 4 | Many trades, lower accuracy |
| **Balanced** | **3** | **6** | **~4 stocks/day, ~48% accuracy** |
| Conservative | 5 | 8 | Few trades, high conviction |
| Ultra-sniper | 8 | 9 | Very rare, extreme conviction |

### Tuning Regularization

`BASE_REG_FACTOR` controls how bold the model's positions are:

| Value | Effect |
|-------|--------|
| `0.001` | Very aggressive — signals saturate near ±1.0 |
| **`0.01`** | **Balanced — signals reach ±0.5 to ±1.0** |
| `0.05` | Conservative — signals rarely exceed ±0.4 |
| `0.5` | Ultra-conservative (for crypto noise filtering) |

---

## Model Architecture

```
Input (B, 30, 2)                    ← [log_return, volatility] × 30 days
    │
    ▼
GRU Encoder (2 layers, 64 hidden)  ← compresses time series → hidden state
    │
    ▼
LayerNorm                           ← stabilizes activations
    │
    ▼
Neural ODE (RK4, 10 steps)         ← integrates h through learned dynamics dh/dt = f(h)
    │
    ▼
Linear → Tanh                      ← position ∈ [-1, 1]
```

**Loss:** `−Sharpe + dynamic_reg × mean(positions²)`  
where `dynamic_reg = BASE_REG_FACTOR × batch_volatility`

---

## Gap-Filling Data Manager

The `MarketDataManager` in `db.py` maintains a SQLite cache:

```python
from db import MarketDataManager

mgr = MarketDataManager()           # uses market_data.db
mgr.update_data("RELIANCE.NS")      # fetches only missing candles
df = mgr.get_features("RELIANCE.NS") # returns [log_ret, vol] DataFrame
```

**How it works:**
- First run → downloads full 2yr history
- Subsequent runs → queries `MAX(date)`, fetches only new candles
- Handles weekends/holidays/script outages automatically

---

## Email Alerts

`alerts.py` formats picks into HTML. To activate delivery:

1. Open `alerts.py`
2. Uncomment the SMTP block at the bottom
3. Fill in your credentials:

```python
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "your_email@gmail.com"
SMTP_PASS = "your_app_password"
TO_EMAIL   = "recipient@example.com"
```

---

## Full Nifty 500 Scan

To scan all 500 stocks instead of top 50:

```python
# In config.py, change:
TOP_N = None    # disables the volume filter
```

> ⚠️ Full scan takes ~2-3 hours depending on hardware.

---

## Dependencies

```
torch          # PyTorch (MPS/CUDA/CPU auto-detected)
yfinance       # Market data from Yahoo Finance
niftystocks    # Nifty 500 ticker lists
flask          # Web dashboard
numpy
pandas
matplotlib
```

---

## Disclaimer

This system is for **educational and research purposes only**. It is not financial advice. Always do your own due diligence before trading.
