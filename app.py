"""
app.py — Flask API server for the Fluid-Geometric Trading Dashboard.

Endpoints:
  GET    /                        → Dashboard UI
  GET    /api/holdings            → Current holdings
  POST   /api/holdings            → Add/update a position
  DELETE /api/holdings/<ticker>   → Remove a position
  GET    /api/signals             → Latest bot signals (from last run)
  POST   /api/run/inference       → Trigger inference (daily or monthly)
  GET    /api/run/status          → Check inference status
"""

import json
import os
import threading
import time
from datetime import datetime

from flask import Flask, jsonify, request, render_template

from config import CONFIDENCE_THRESHOLD, SNIPER_THRESHOLD, TOP_N

app = Flask(__name__)

HOLDINGS_FILE = os.path.join(os.path.dirname(__file__), "holdings.json")
SIGNALS_FILE = os.path.join(os.path.dirname(__file__), "signals.json")

# Inference state
_inference_lock = threading.Lock()
_inference_state = {
    "running": False,
    "progress": "",
    "started_at": None,
    "completed_at": None,
    "mode": None,
    "logs": [],
}


# ──────────────────────── Holdings CRUD ───────────────────────
def _load_holdings() -> dict:
    if os.path.exists(HOLDINGS_FILE):
        with open(HOLDINGS_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_holdings(data: dict):
    with open(HOLDINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _load_signals() -> list:
    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, "r") as f:
            return json.load(f)
    return []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/holdings", methods=["GET"])
def get_holdings():
    return jsonify(_load_holdings())


@app.route("/api/holdings", methods=["POST"])
def upsert_holding():
    data = request.json
    ticker = data.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    holdings = _load_holdings()
    holdings[ticker] = {
        "action": data.get("action", 0.0),
        "confidence": data.get("confidence", 0.0),
        "signal": data.get("signal", "MANUAL"),
        "direction": data.get("direction", "LONG"),
        "added_at": datetime.now().isoformat(),
        "manual": True,
    }
    _save_holdings(holdings)
    return jsonify({"ok": True, "holdings": holdings})


@app.route("/api/holdings/<ticker>", methods=["DELETE"])
def delete_holding(ticker):
    holdings = _load_holdings()
    ticker = ticker.upper()
    if ticker in holdings:
        del holdings[ticker]
        _save_holdings(holdings)
    return jsonify({"ok": True, "holdings": holdings})


@app.route("/api/signals", methods=["GET"])
def get_signals():
    return jsonify(_load_signals())


# ──────────────────────── Inference Runner ────────────────────
def _run_inference_thread(mode: str):
    """Background thread that runs inference and saves signals."""
    import config

    # Adjust config for mode
    if mode == "monthly":
        config.PERIOD = "5y"
        config.TRAIN_RATIO = 0.80
    else:
        config.PERIOD = "2y"
        config.TRAIN_RATIO = 0.75

    from db import MarketDataManager
    from data import get_nifty500_universe
    from model import FluidGeometricNet
    from train import train
    from daily_inference import process_stock, label_signal

    mgr = MarketDataManager()
    signals = []

    try:
        with _inference_lock:
            _inference_state["logs"].append("Building stock universe …")
            _inference_state["progress"] = "Building universe"

        universe = get_nifty500_universe()

        with _inference_lock:
            _inference_state["logs"].append(f"Universe: {len(universe)} stocks")
            _inference_state["progress"] = f"Training 0/{len(universe)}"

        prev_holdings = _load_holdings()

        for i, ticker in enumerate(universe, 1):
            with _inference_lock:
                _inference_state["progress"] = f"Training {i}/{len(universe)}: {ticker}"

            res = process_stock(ticker, mgr)
            if res is None:
                continue

            action = res["latest_action"]
            conf = min(round(abs(action) * 10, 1), 10.0)
            label = label_signal(action, conf)

            entry = {
                "ticker": ticker,
                "action": round(action, 4),
                "confidence": conf,
                "signal": label,
                "is_buy": label != "—",
                "is_exit": (ticker in prev_holdings and label == "—"),
                "is_new": (ticker not in prev_holdings and label != "—"),
                "is_held": (ticker in prev_holdings and label != "—"),
            }
            signals.append(entry)

        # Sort by confidence descending
        signals.sort(key=lambda x: x["confidence"], reverse=True)

        # Save signals
        with open(SIGNALS_FILE, "w") as f:
            json.dump(signals, f, indent=2)

        # Update holdings with buy signals
        new_holdings = {}
        for s in signals:
            if s["is_buy"]:
                new_holdings[s["ticker"]] = {
                    "action": s["action"],
                    "confidence": s["confidence"],
                    "signal": s["signal"],
                    "direction": "LONG",
                    "added_at": datetime.now().isoformat(),
                    "manual": False,
                }
        # Preserve manual holdings
        old = _load_holdings()
        for t, h in old.items():
            if h.get("manual") and t not in new_holdings:
                new_holdings[t] = h
        _save_holdings(new_holdings)

        with _inference_lock:
            _inference_state["logs"].append(
                f"Done: {len(signals)} signals, "
                f"{sum(1 for s in signals if s['is_buy'])} buys"
            )

    except Exception as e:
        with _inference_lock:
            _inference_state["logs"].append(f"Error: {e}")
    finally:
        with _inference_lock:
            _inference_state["running"] = False
            _inference_state["completed_at"] = datetime.now().isoformat()
            _inference_state["progress"] = "Complete"


@app.route("/api/run/inference", methods=["POST"])
def run_inference():
    data = request.json or {}
    mode = data.get("mode", "daily")

    with _inference_lock:
        if _inference_state["running"]:
            return jsonify({"error": "Inference already running"}), 409

        _inference_state["running"] = True
        _inference_state["started_at"] = datetime.now().isoformat()
        _inference_state["completed_at"] = None
        _inference_state["mode"] = mode
        _inference_state["progress"] = "Starting …"
        _inference_state["logs"] = []

    t = threading.Thread(target=_run_inference_thread, args=(mode,), daemon=True)
    t.start()

    return jsonify({"ok": True, "mode": mode})


@app.route("/api/run/status", methods=["GET"])
def run_status():
    with _inference_lock:
        return jsonify(dict(_inference_state))


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(
        {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "sniper_threshold": SNIPER_THRESHOLD,
            "top_n": TOP_N,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
