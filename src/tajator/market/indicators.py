"""Indicator math on 1-minute bars. All timestamps are US/Eastern tz-aware."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from ..models import Bar, Snapshot

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
PREMARKET_OPEN = time(4, 0)


def bars_to_df(bars: list[Bar]) -> pd.DataFrame:
    df = pd.DataFrame([b.model_dump() for b in bars])
    if df.empty:
        return df
    df = df.set_index("ts").sort_index()
    return df


def session_df(df: pd.DataFrame, ts: datetime) -> pd.DataFrame:
    """Regular-session (>= 09:30 ET) bars of the day containing `ts`."""
    day = ts.astimezone(ET).date()
    idx = df.index.tz_convert(ET)
    mask = (idx.date == day) & (idx.time >= RTH_OPEN)
    return df[mask]


def premarket_df(df: pd.DataFrame, ts: datetime) -> pd.DataFrame:
    """Premarket (04:00–09:30 ET) bars of the day containing `ts`."""
    day = ts.astimezone(ET).date()
    idx = df.index.tz_convert(ET)
    mask = (idx.date == day) & (idx.time >= PREMARKET_OPEN) & (idx.time < RTH_OPEN)
    return df[mask]


def ema(closes: pd.Series, span: int) -> float | None:
    if len(closes) < span:
        return None
    return float(closes.ewm(span=span, adjust=False).mean().iloc[-1])


def session_vwap(session: pd.DataFrame) -> float | None:
    """VWAP anchored at the 09:30 ET regular-session open."""
    if session.empty or session["volume"].sum() <= 0:
        return None
    typical = (session["high"] + session["low"] + session["close"]) / 3.0
    return float((typical * session["volume"]).sum() / session["volume"].sum())


def build_snapshot(symbol: str, bars: list[Bar]) -> Snapshot:
    df = bars_to_df(bars)
    ts = bars[-1].ts
    price = bars[-1].close
    session = session_df(df, ts)
    return Snapshot(
        symbol=symbol,
        ts=ts,
        price=price,
        ema9=ema(session["close"], 9) if not session.empty else None,
        ema50=ema(session["close"], 50) if not session.empty else None,
        vwap=session_vwap(session),
        hod=float(session["high"].max()) if not session.empty else None,
        lod=float(session["low"].min()) if not session.empty else None,
    )
