"""
config.py — Centralized hyperparameters and device detection.

Production configuration for Indian NSE equity market.
"""

import torch
import os

# ──────────────────────────── Data ────────────────────────────
PERIOD = "2y"
LOOKBACK = 30  # rolling window length (trading days)
VOL_WINDOW = 20  # rolling volatility window
TRAIN_RATIO = 0.75  # 1.5yr train / 0.5yr test
DB_PATH = os.path.join(os.path.dirname(__file__), "market_data.db")

# ──────────────────────────── Universe ────────────────────────
TOP_N = 150  # top N most liquid stocks from Nifty 500
# Set TOP_N = None to scan the full Nifty 500
BENCHMARK = "^NSEI"  # Nifty 50 index for comparison

# ──────────────────────────── Model ───────────────────────────
INPUT_DIM = 2  # [log_return, volatility]
HIDDEN_DIM = 64
GRU_LAYERS = 2
ODE_STEPS = 10  # RK4 integration sub-steps

# ──────────────────────────── Training ────────────────────────
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 128
BASE_REG_FACTOR = 0.01  # light reg — allows bold positions

# ──────────────────────────── Signal Tiers ────────────────────
CONFIDENCE_THRESHOLD = 3  # min confidence to trade (0-10)
SNIPER_THRESHOLD = 6  # confidence for "SNIPER BUY" label


# ──────────────────────────── Device ──────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
