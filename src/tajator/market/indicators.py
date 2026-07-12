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


def atr(session: pd.DataFrame, window: int = 14) -> float | None:
    if len(session) < window + 1:
        return None
    prev_close = session["close"].shift(1)
    tr = pd.concat([
        session["high"] - session["low"],
        (session["high"] - prev_close).abs(),
        (session["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.tail(window).mean())


def _vwap_series(session: pd.DataFrame) -> pd.Series:
    typical = (session["high"] + session["low"] + session["close"]) / 3.0
    volume = session["volume"].cumsum()
    return (typical * session["volume"]).cumsum() / volume.where(volume > 0)


def _regime(session: pd.DataFrame, current_vwap: float | None, current_atr: float | None) -> tuple:
    if current_vwap is None or len(session) < 15:
        return None, None, None, "unknown"
    vwaps = _vwap_series(session)
    slope5 = float(vwaps.iloc[-1] - vwaps.iloc[-6]) if len(vwaps) >= 6 else None
    slope15 = float(vwaps.iloc[-1] - vwaps.iloc[-16]) if len(vwaps) >= 16 else None
    frac = float((session["close"].tail(15) > vwaps.tail(15)).mean())
    threshold = (current_atr or 0.0) * 0.35
    if slope15 is not None and slope15 > threshold and frac >= 0.8:
        regime = "trend_up"
    elif slope15 is not None and slope15 < -threshold and frac <= 0.2:
        regime = "trend_down"
    elif current_atr and (session["high"].tail(15).max() - session["low"].tail(15).min()) > 3 * current_atr:
        regime = "high_volatility"
    else:
        regime = "range"
    return slope5, slope15, frac, regime


def build_snapshot(symbol: str, bars: list[Bar], *, atr_window: int = 14) -> Snapshot:
    df = bars_to_df(bars)
    ts = bars[-1].ts
    price = bars[-1].close
    session = session_df(df, ts)
    current_vwap = session_vwap(session)
    current_atr = atr(session, atr_window)
    slope5, slope15, frac, regime = _regime(session, current_vwap, current_atr)
    return Snapshot(
        symbol=symbol,
        ts=ts,
        price=price,
        ema9=ema(session["close"], 9) if not session.empty else None,
        ema50=ema(session["close"], 50) if not session.empty else None,
        vwap=current_vwap,
        hod=float(session["high"].max()) if not session.empty else None,
        lod=float(session["low"].min()) if not session.empty else None,
        atr=current_atr,
        vwap_slope_5=slope5,
        vwap_slope_15=slope15,
        closes_above_vwap_frac=frac,
        regime=regime,
    )
