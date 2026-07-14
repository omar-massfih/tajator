"""Historical data fetch + on-disk cache for backtesting.

Underlying 1-min bars and option 1-min bars are cached as CSVs in the same
`ts,open,high,low,close,volume` shape `StubBroker.from_csv` already parses,
so a populated cache can be inspected/reused with the existing tooling.
"""

from __future__ import annotations

import csv
import logging
import time as time_mod
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ..models import Bar, SelectedContract

ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)

# small pause between individual IB historical-data requests to stay clear of pacing limits
IB_REQUEST_PAUSE_S = 1.0


def trading_days(start: date, end: date) -> list[date]:
    """Weekdays only — no holiday calendar (same limitation as v1's live loop)."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _read_csv(path: Path) -> list[Bar]:
    bars = []
    with path.open() as f:
        for row in csv.DictReader(f):
            bars.append(
                Bar(
                    ts=datetime.fromisoformat(row["ts"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return bars


def _write_csv(path: Path, bars: list[Bar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "open", "high", "low", "close", "volume"])
        for b in bars:
            writer.writerow([b.ts.isoformat(), b.open, b.high, b.low, b.close, b.volume])


def _underlying_cache_path(cache_dir: Path, symbol: str, day: date) -> Path:
    return cache_dir / symbol / f"{day.isoformat()}.csv"


def ensure_underlying_bars(
    ib, symbol: str, day: date, cache_dir: Path, *, refresh: bool = False
) -> list[Bar]:
    """1-min underlying bars for one session, cached to disk after the first fetch."""
    path = _underlying_cache_path(cache_dir, symbol, day)
    if path.exists() and not refresh:
        # IB may answer a holiday request with the preceding session. Never
        # replay those bars under the requested date.
        return [b for b in _read_csv(path) if b.ts.astimezone(ET).date() == day]
    if ib is None:
        return [b for b in _read_csv(path) if b.ts.astimezone(ET).date() == day] if path.exists() else []
    end = datetime.combine(day, datetime.min.time(), tzinfo=ET).replace(hour=20)
    raw = ib.ib.reqHistoricalData(
        ib._underlying(symbol),
        endDateTime=end,
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
    )
    time_mod.sleep(IB_REQUEST_PAUSE_S)
    bars = [
        Bar(
            ts=b.date.astimezone(ET), open=b.open, high=b.high, low=b.low, close=b.close,
            volume=float(b.volume) if b.volume else 0.0,
        )
        for b in raw
        if b.date.astimezone(ET).date() == day
    ]
    if bars:
        _write_csv(path, bars)
    return bars


def fetch_daily_series(ib, symbol: str, start: date, end: date) -> list[Bar]:
    """One historical-data call for daily OHLC covering the whole window (plus a lookback pad),
    used to derive each day's *previous* session high/low without re-fetching per day."""
    pad_days = (end - start).days + 140
    stop = datetime.combine(end, datetime.min.time(), tzinfo=ET).replace(hour=20)
    raw = ib.ib.reqHistoricalData(
        ib._underlying(symbol),
        endDateTime=stop,
        durationStr=f"{pad_days} D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=2,
    )
    time_mod.sleep(IB_REQUEST_PAUSE_S)
    return [
        Bar(
            ts=b.date if isinstance(b.date, datetime) else datetime.combine(b.date, datetime.min.time()),
            open=b.open, high=b.high, low=b.low, close=b.close,
            volume=float(b.volume) if b.volume else 0.0,
        )
        for b in raw
    ]


def daily_series_from_underlying_cache(cache_dir: Path, symbol: str) -> list[Bar]:
    """Aggregate cached minute sessions into daily bars for causal cached research."""
    daily: list[Bar] = []
    symbol_dir = cache_dir / symbol
    if not symbol_dir.exists():
        return daily
    for path in sorted(symbol_dir.glob("????-??-??.csv")):
        bars = _read_csv(path)
        session = [
            bar for bar in bars
            if (bar.ts.astimezone(ET).hour, bar.ts.astimezone(ET).minute) >= (9, 30)
            and (bar.ts.astimezone(ET).hour, bar.ts.astimezone(ET).minute) <= (16, 0)
        ]
        if not session:
            continue
        day = session[0].ts.astimezone(ET).date()
        daily.append(
            Bar(
                ts=datetime.combine(day, datetime.min.time(), tzinfo=ET),
                open=session[0].open,
                high=max(bar.high for bar in session),
                low=min(bar.low for bar in session),
                close=session[-1].close,
                volume=sum(bar.volume for bar in session),
            )
        )
    return daily


def prev_day_range_for(daily_series: list[Bar], day: date) -> tuple[float | None, float | None]:
    """(high, low) of the most recent session strictly before `day`."""
    prior = [b for b in daily_series if b.ts.date() < day]
    if not prior:
        return None, None
    prev = prior[-1]
    return prev.high, prev.low


def _option_cache_path(cache_dir: Path, contract: SelectedContract, day: date) -> Path:
    name = f"{contract.expiry}_{contract.strike:g}{contract.right}_{day.isoformat()}.csv"
    return cache_dir / contract.symbol / "options" / name


def ensure_option_bars(ib, contract: SelectedContract, day: date, cache_dir: Path) -> list[Bar] | None:
    """1-min option bars for one session; None means no real data was available."""
    path = _option_cache_path(cache_dir, contract, day)
    if path.exists():
        bars = _read_csv(path)
        return bars or None
    if ib is None:
        return None
    end = datetime.combine(day, datetime.min.time(), tzinfo=ET).replace(hour=20)
    try:
        opt = ib._option(contract, include_expired=True)
    except Exception as exc:  # noqa: BLE001 — expired contracts may fail to qualify
        log.warning("could not qualify %s for %s: %s — no real price available, "
                    "the backtest will abort if this contract must be priced",
                    contract.local_name, day, exc)
        return None
    bars: list[Bar] = []
    for what in ("TRADES", "MIDPOINT"):
        raw = ib.ib.reqHistoricalData(
            opt, endDateTime=end, durationStr="1 D", barSizeSetting="1 min",
            whatToShow=what, useRTH=False, formatDate=2,
        )
        time_mod.sleep(IB_REQUEST_PAUSE_S)
        if raw:
            bars = [
                Bar(
                    ts=b.date.astimezone(ET), open=b.open, high=b.high, low=b.low, close=b.close,
                    volume=float(b.volume) if b.volume else 0.0,
                )
                for b in raw
            ]
            break
    if not bars:
        log.warning("no historical option data for %s on %s — no real price available, "
                     "the backtest will abort if this contract must be priced",
                     contract.local_name, day)
        # cache the miss too (empty file) so a rerun doesn't re-hit IB for the same gap
        _write_csv(path, [])
        return None
    _write_csv(path, bars)
    return bars
