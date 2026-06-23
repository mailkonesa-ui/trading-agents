"""
Data layer: fetch 5y history + latest 2 weeks of online data via Yahoo Finance.
Caches history to disk so we don't hammer the API.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "history"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "_").replace(".", "_").replace("-", "_")
    return CACHE_DIR / f"{safe}.parquet"


def fetch_history(ticker: str, years: int = 5) -> pd.DataFrame:
    """
    Fetch up to `years` years of daily OHLCV for `ticker`.
    Cached on disk; refreshes only files older than 24h.
    """
    path = _cache_path(ticker)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < 24 * 3600:
            return pd.read_parquet(path)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(years * 365.25))
    df = yf.download(
        ticker,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_parquet(path)
    return df


def fetch_recent(ticker: str, days: int = 14) -> pd.DataFrame:
    """Fetch only the last `days` of data (used for live 'online' updates)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = yf.download(
        ticker,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def latest_price(ticker: str) -> float | None:
    """Best-effort latest price: cache the last close, fall back to network."""
    try:
        df = fetch_history(ticker, years=5)
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def combined_series(ticker: str) -> pd.DataFrame:
    """Return history with at least the most recent 2 weeks of online data merged in."""
    hist = fetch_history(ticker, years=5)
    recent = fetch_recent(ticker, days=14)
    if recent.empty:
        return hist
    combined = pd.concat([hist, recent])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined
