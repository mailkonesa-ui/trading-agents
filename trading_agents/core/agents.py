"""
Five strategy agents. Each agent is parameterised so the user can tune
them from the UI (sliders in the sidebar).

Each agent returns a tuple:
    (action: "BUY" | "SELL" | "HOLD", confidence: float in [0, 1], reason: str)

Confidence = how strongly the strategy's signal fires (0 = no edge, 1 = max).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .indicators import bollinger, ema, macd, momentum, rsi, sma


@dataclass
class AgentConfig:
    name: str
    params: dict
    weight: float = 1.0   # updated by online learning
    wins: int = 0
    losses: int = 0
    history: list[dict] = field(default_factory=list)


def _sigmoid(x: float) -> float:
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _agent_sma(df: pd.DataFrame, params: dict) -> tuple[str, float, str]:
    short = int(params.get("short", 10))
    long = int(params.get("long", 30))
    if long <= short:
        return "HOLD", 0.0, "long<=short"
    close = df["Close"]
    s = sma(close, short)
    l = sma(close, long)
    if pd.isna(s.iloc[-1]) or pd.isna(l.iloc[-1]):
        return "HOLD", 0.0, "not enough data"
    s_now, l_now = float(s.iloc[-1]), float(l.iloc[-1])
    s_prev, l_prev = float(s.iloc[-2]), float(l.iloc[-2])
    cross_up = s_prev <= l_prev and s_now > l_now
    cross_dn = s_prev >= l_prev and s_now < l_now
    gap = (s_now - l_now) / l_now
    conf = _sigmoid(gap * 200)
    if cross_up:
        return "BUY", conf, f"SMA{short} crossed above SMA{long} (gap {gap*100:.2f}%)"
    if cross_dn:
        return "SELL", conf, f"SMA{short} crossed below SMA{long} (gap {gap*100:.2f}%)"
    return "HOLD", 0.0, "no crossover"


def _agent_rsi(df: pd.DataFrame, params: dict) -> tuple[str, float, str]:
    period = int(params.get("period", 14))
    low = float(params.get("low", 30))
    high = float(params.get("high", 70))
    r = rsi(df["Close"], period)
    if pd.isna(r.iloc[-1]):
        return "HOLD", 0.0, "not enough data"
    val = float(r.iloc[-1])
    if val < low:
        conf = _sigmoid((low - val) / 5)
        return "BUY", conf, f"RSI={val:.1f} < {low} (oversold)"
    if val > high:
        conf = _sigmoid((val - high) / 5)
        return "SELL", conf, f"RSI={val:.1f} > {high} (overbought)"
    return "HOLD", 0.0, f"RSI={val:.1f} in neutral zone"


def _agent_bollinger(df: pd.DataFrame, params: dict) -> tuple[str, float, str]:
    period = int(params.get("period", 20))
    std = float(params.get("std", 2.0))
    upper, mid, lower = bollinger(df["Close"], period, std)
    if pd.isna(upper.iloc[-1]):
        return "HOLD", 0.0, "not enough data"
    price = float(df["Close"].iloc[-1])
    u, m, lo = float(upper.iloc[-1]), float(mid.iloc[-1]), float(lower.iloc[-1])
    width = (u - lo) / m if m else 0
    if price < lo:
        conf = _sigmoid((lo - price) / m * 200)
        return "BUY", conf, f"Price below lower band (BB width {width*100:.2f}%)"
    if price > u:
        conf = _sigmoid((price - u) / m * 200)
        return "SELL", conf, f"Price above upper band (BB width {width*100:.2f}%)"
    return "HOLD", 0.0, "price inside bands"


def _agent_macd(df: pd.DataFrame, params: dict) -> tuple[str, float, str]:
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 26))
    sig = int(params.get("signal", 9))
    line, signal_line, hist = macd(df["Close"], fast, slow, sig)
    if pd.isna(line.iloc[-1]) or pd.isna(signal_line.iloc[-1]):
        return "HOLD", 0.0, "not enough data"
    h_now = float(hist.iloc[-1])
    h_prev = float(hist.iloc[-2])
    cross_up = h_prev <= 0 < h_now
    cross_dn = h_prev >= 0 > h_now
    conf = _sigmoid(h_now * 50)
    if cross_up:
        return "BUY", conf, f"MACD histogram crossed above 0 ({h_now:.4f})"
    if cross_dn:
        return "SELL", conf, f"MACD histogram crossed below 0 ({h_now:.4f})"
    return "HOLD", 0.0, f"hist={h_now:.4f} (no cross)"


def _agent_momentum(df: pd.DataFrame, params: dict) -> tuple[str, float, str]:
    period = int(params.get("period", 20))
    threshold = float(params.get("threshold", 0.05))
    m = momentum(df["Close"], period)
    if pd.isna(m.iloc[-1]):
        return "HOLD", 0.0, "not enough data"
    val = float(m.iloc[-1])
    if val > threshold:
        conf = _sigmoid((val - threshold) * 20)
        return "BUY", conf, f"Momentum({period})={val*100:.2f}% > +{threshold*100:.1f}%"
    if val < -threshold:
        conf = _sigmoid((val + threshold) * 20)
        return "SELL", conf, f"Momentum({period})={val*100:.2f}% < -{threshold*100:.1f}%"
    return "HOLD", 0.0, f"momentum={val*100:.2f}%"


AGENTS: dict[str, Callable] = {
    "SMA Crossover": _agent_sma,
    "RSI Mean Reversion": _agent_rsi,
    "Bollinger Breakout": _agent_bollinger,
    "MACD Crossover": _agent_macd,
    "Momentum": _agent_momentum,
}


# Default parameter sets (also used to seed UI sliders)
DEFAULT_PARAMS: dict[str, dict] = {
    "SMA Crossover":        {"short": 10, "long": 30},
    "RSI Mean Reversion":   {"period": 14, "low": 30, "high": 70},
    "Bollinger Breakout":   {"period": 20, "std": 2.0},
    "MACD Crossover":       {"fast": 12, "slow": 26, "signal": 9},
    "Momentum":             {"period": 20, "threshold": 0.05},
}


# Bounds the user can tune in the UI
PARAM_BOUNDS: dict[str, dict] = {
    "SMA Crossover":        {"short": (5, 50), "long": (20, 200)},
    "RSI Mean Reversion":   {"period": (5, 30), "low": (10, 40), "high": (60, 90)},
    "Bollinger Breakout":   {"period": (10, 50), "std": (1.0, 3.0)},
    "MACD Crossover":       {"fast": (5, 20), "slow": (15, 50), "signal": (5, 20)},
    "Momentum":             {"period": (5, 60), "threshold": (0.01, 0.20)},
}


def run_agent(name: str, df: pd.DataFrame, params: dict) -> tuple[str, float, str]:
    fn = AGENTS.get(name)
    if fn is None:
        return "HOLD", 0.0, "unknown agent"
    try:
        return fn(df, params)
    except Exception as e:
        return "HOLD", 0.0, f"error: {e}"
