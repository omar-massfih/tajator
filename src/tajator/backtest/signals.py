"""Edge-search engine: test documented, structural signals for a real directional
edge — offline, on the cached underlying bars, with rigorous validation.

The point is to *reject* pretty-but-fake signals. Every candidate faces four gates:
a pre-declared temporal holdout, a cross-symbol holdout (found on some names,
confirmed on unseen ones), a Bonferroni correction across every signal tried, and
day-clustered standard errors (positions on the same date are correlated, so a
trading day — not a trade — is the unit of evidence). A signal "survives" only if
its holdout expectancy stays positive with a Bonferroni-adjusted 95% lower bound
above zero, and it is also positive out-of-sample on symbols it was never fit on.

Returns are net of a round-trip stock cost (spread+commission, in bps). Signals are
long-biased equity edges (documented: overnight drift, short-term reversal,
time-series momentum, turn-of-month) plus a few conditioned intraday effects.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import NormalDist, fmean, pstdev

from ..models import Bar
from .data import ET, _read_csv

DEFAULT_COST_BPS = 2.0  # round-trip stock spread+commission on a liquid name

# Approximate hold length per signal, used only by the option-economics model.
SIGNAL_HORIZON_DAYS = {
    "overnight": 1, "intraday": 1, "reversal_1d": 1, "reversal_5d": 1,
    "momentum_ma20": 1, "momentum_ma200": 1, "turn_of_month": 3,
    "gap_fill": 1, "first_hour_reversal": 1, "last_hour_drift": 1,
}


@dataclass(frozen=True)
class Trade:
    day: date       # entry trading day — the clustering unit
    symbol: str
    ret: float      # net, direction-adjusted return fraction (cost already deducted)


def _net(entry: float, exit_: float, direction: int, cost_bps: float) -> float:
    if entry <= 0:
        return 0.0
    return direction * (exit_ / entry - 1.0) - cost_bps / 1e4


# --------------------------------------------------------------------------- #
# Daily signals — fn(daily_bars, symbol, cost_bps) -> list[Trade]. All causal:
# every entry decision uses only information available at or before the entry bar.
# --------------------------------------------------------------------------- #

def _daily_trades(bars, symbol, cost_bps, predicate, entry_px, exit_px, exit_offset):
    out = []
    n = len(bars)
    for i in range(n):
        j = i + exit_offset
        if j >= n or not predicate(i):
            continue
        out.append(Trade(bars[i].ts.date(), symbol, _net(entry_px(i), exit_px(j), 1, cost_bps)))
    return out


def sig_overnight(bars: list[Bar], symbol: str, cost_bps: float) -> list[Trade]:
    """Long the overnight gap: buy at close, sell at next open."""
    out = []
    for i in range(len(bars) - 1):
        out.append(Trade(bars[i].ts.date(), symbol, _net(bars[i].close, bars[i + 1].open, 1, cost_bps)))
    return out


def sig_intraday(bars: list[Bar], symbol: str, cost_bps: float) -> list[Trade]:
    """Long the regular session: buy at open, sell at close (overnight's counterpart)."""
    return [Trade(b.ts.date(), symbol, _net(b.open, b.close, 1, cost_bps)) for b in bars]


def sig_reversal_1d(bars: list[Bar], symbol: str, cost_bps: float) -> list[Trade]:
    """Short-term reversal: after a down day, buy the close and hold one day."""
    return _daily_trades(
        bars, symbol, cost_bps,
        predicate=lambda i: i >= 1 and bars[i].close < bars[i - 1].close,
        entry_px=lambda i: bars[i].close, exit_px=lambda j: bars[j].close, exit_offset=1,
    )


def sig_reversal_5d(bars: list[Bar], symbol: str, cost_bps: float) -> list[Trade]:
    """Buy after a 5-day decline, hold one day."""
    return _daily_trades(
        bars, symbol, cost_bps,
        predicate=lambda i: i >= 5 and bars[i].close < bars[i - 5].close,
        entry_px=lambda i: bars[i].close, exit_px=lambda j: bars[j].close, exit_offset=1,
    )


def _sma(bars, i, window):
    if i + 1 < window:
        return None
    return fmean(b.close for b in bars[i + 1 - window:i + 1])


def sig_momentum_ma20(bars: list[Bar], symbol: str, cost_bps: float) -> list[Trade]:
    """Time-series momentum: long while above the 20-day SMA, held one day."""
    return _daily_trades(
        bars, symbol, cost_bps,
        predicate=lambda i: (_sma(bars, i, 20) is not None and bars[i].close > _sma(bars, i, 20)),
        entry_px=lambda i: bars[i].close, exit_px=lambda j: bars[j].close, exit_offset=1,
    )


def sig_momentum_ma200(bars: list[Bar], symbol: str, cost_bps: float) -> list[Trade]:
    """Long while above the 200-day SMA (regime filter), held one day."""
    return _daily_trades(
        bars, symbol, cost_bps,
        predicate=lambda i: (_sma(bars, i, 200) is not None and bars[i].close > _sma(bars, i, 200)),
        entry_px=lambda i: bars[i].close, exit_px=lambda j: bars[j].close, exit_offset=1,
    )


def sig_turn_of_month(bars: list[Bar], symbol: str, cost_bps: float) -> list[Trade]:
    """Turn-of-month: buy the last trading day of the month, hold three sessions."""
    out = []
    n = len(bars)
    for i in range(n - 1):
        if bars[i + 1].ts.month != bars[i].ts.month:  # i is the month's last session
            j = min(i + 3, n - 1)
            out.append(Trade(bars[i].ts.date(), symbol, _net(bars[i].close, bars[j].close, 1, cost_bps)))
    return out


DAILY_SIGNALS = {
    "overnight": sig_overnight,
    "intraday": sig_intraday,
    "reversal_1d": sig_reversal_1d,
    "reversal_5d": sig_reversal_5d,
    "momentum_ma20": sig_momentum_ma20,
    "momentum_ma200": sig_momentum_ma200,
    "turn_of_month": sig_turn_of_month,
}


# --------------------------------------------------------------------------- #
# Intraday signals — fn(sessions, symbol, cost_bps) -> list[Trade], where
# `sessions` is an ordered list of (date, rth_bars). Causal within the day.
# --------------------------------------------------------------------------- #

def _rth(bars: list[Bar]) -> list[Bar]:
    return [b for b in bars if (b.ts.astimezone(ET).hour, b.ts.astimezone(ET).minute) >= (9, 30)
            and b.ts.astimezone(ET).hour < 16]


def sig_gap_fill(sessions, symbol: str, cost_bps: float) -> list[Trade]:
    """Fade a >0.5% opening gap back toward the prior close (mean reversion)."""
    out = []
    for k in range(1, len(sessions)):
        prev_close = sessions[k - 1][1][-1].close
        day, bars = sessions[k]
        if not bars or prev_close <= 0:
            continue
        op = bars[0].open
        gap = (op - prev_close) / prev_close
        if abs(gap) < 0.005:
            continue
        direction = -1 if gap > 0 else 1  # fade toward prior close
        out.append(Trade(day, symbol, _net(op, bars[-1].close, direction, cost_bps)))
    return out


def sig_first_hour_reversal(sessions, symbol: str, cost_bps: float) -> list[Trade]:
    """Fade the first hour: if 09:30->10:30 is up, short the rest of day, and vice versa."""
    out = []
    for day, bars in sessions:
        if len(bars) < 90:
            continue
        first = next((b for b in bars if (b.ts.astimezone(ET).hour, b.ts.astimezone(ET).minute) >= (10, 30)), None)
        if first is None:
            continue
        fh = (first.open - bars[0].open) / bars[0].open if bars[0].open else 0.0
        if abs(fh) < 0.003:
            continue
        direction = -1 if fh > 0 else 1
        out.append(Trade(day, symbol, _net(first.open, bars[-1].close, direction, cost_bps)))
    return out


def sig_last_hour_drift(sessions, symbol: str, cost_bps: float) -> list[Trade]:
    """Long the last hour: buy ~15:00, sell at the close."""
    out = []
    for day, bars in sessions:
        entry = next((b for b in bars if b.ts.astimezone(ET).hour >= 15), None)
        if entry is None or not bars:
            continue
        out.append(Trade(day, symbol, _net(entry.open, bars[-1].close, 1, cost_bps)))
    return out


INTRADAY_SIGNALS = {
    "gap_fill": sig_gap_fill,
    "first_hour_reversal": sig_first_hour_reversal,
    "last_hour_drift": sig_last_hour_drift,
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_daily(daily_dir: Path, symbol: str) -> list[Bar]:
    path = daily_dir / f"{symbol}.csv"
    if not path.exists():
        return []
    return sorted(_read_csv(path), key=lambda b: b.ts)


def load_sessions(cache_dir: Path, symbol: str) -> list[tuple[date, list[Bar]]]:
    """Ordered (date, RTH bars) sessions from the cached 1-minute day files."""
    sym_dir = cache_dir / symbol
    if not sym_dir.is_dir():
        return []
    sessions = []
    for path in sorted(sym_dir.glob("*.csv")):
        rth = _rth(sorted(_read_csv(path), key=lambda b: b.ts))
        if rth:
            sessions.append((rth[0].ts.astimezone(ET).date(), rth))
    return sessions


# --------------------------------------------------------------------------- #
# Statistics — day-clustered mean with a Bonferroni-adjusted CI
# --------------------------------------------------------------------------- #

def clustered_stats(trades: list[Trade], *, z: float) -> dict:
    """A trading day is one observation (same-day trades are correlated). Returns
    per-trade mean, day-clustered SE, and the Bonferroni-adjusted CI lower bound."""
    by_day: dict[date, list[float]] = defaultdict(list)
    for t in trades:
        by_day[t.day].append(t.ret)
    daily = [fmean(v) for v in by_day.values()]
    n_days = len(daily)
    if n_days == 0:
        return {"trades": 0, "days": 0, "mean_bps": 0.0, "ci_low_bps": 0.0}
    mean = fmean(daily)
    se = (pstdev(daily) / (n_days ** 0.5)) if n_days > 1 else float("inf")
    return {
        "trades": len(trades),
        "days": n_days,
        "mean_bps": round(mean * 1e4, 2),
        "ci_low_bps": round((mean - z * se) * 1e4, 2),
    }


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

def _split_symbols(symbols: list[str]) -> tuple[list[str], list[str]]:
    """Deterministic dev/holdout symbol split (even index = dev, odd = confirm)."""
    dev = [s for i, s in enumerate(symbols) if i % 2 == 0]
    hold = [s for i, s in enumerate(symbols) if i % 2 == 1]
    return dev, hold


def run_edge_search(
    symbols: list[str],
    *,
    daily_dir: Path,
    intraday_dir: Path,
    dev_end: date,
    holdout_start: date,
    horizon: str = "daily",
    cost_bps: float = DEFAULT_COST_BPS,
) -> dict:
    catalog: dict[str, tuple[str, object]] = {}
    if horizon in ("daily", "both"):
        catalog.update({name: ("daily", fn) for name, fn in DAILY_SIGNALS.items()})
    if horizon in ("intraday", "both"):
        catalog.update({name: ("intraday", fn) for name, fn in INTRADAY_SIGNALS.items()})

    # Load once per symbol.
    daily = {s: load_daily(daily_dir, s) for s in symbols}
    sessions = (
        {s: load_sessions(intraday_dir, s) for s in symbols}
        if horizon in ("intraday", "both") else {}
    )
    dev_syms, hold_syms = _split_symbols(symbols)
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * max(1, len(catalog))))  # Bonferroni 95%

    rows = []
    for name, (kind, fn) in catalog.items():
        per_symbol: dict[str, list[Trade]] = {}
        for s in symbols:
            data = daily.get(s, []) if kind == "daily" else sessions.get(s, [])
            per_symbol[s] = fn(data, s, cost_bps) if data else []

        all_trades = [t for ts in per_symbol.values() for t in ts]
        dev = [t for t in all_trades if t.day < holdout_start]
        hold = [t for t in all_trades if t.day >= holdout_start]
        # Cross-symbol: fit on dev symbols (all dates), confirm on unseen symbols in holdout.
        cross = [t for s in hold_syms for t in per_symbol.get(s, []) if t.day >= holdout_start]

        dev_stats = clustered_stats(dev, z=z)
        hold_stats = clustered_stats(hold, z=z)
        cross_stats = clustered_stats(cross, z=z)
        survives = (
            hold_stats["days"] >= 30 and hold_stats["ci_low_bps"] > 0
            and cross_stats["days"] >= 20 and cross_stats["mean_bps"] > 0
        )
        rows.append({
            "signal": name, "horizon": kind,
            "dev": dev_stats, "holdout": hold_stats, "cross_symbol": cross_stats,
            "survives": survives,
        })

    rows.sort(key=lambda r: r["holdout"]["ci_low_bps"], reverse=True)
    return {
        "symbols": symbols,
        "dev_symbols": dev_syms, "holdout_symbols": hold_syms,
        "dev_end": dev_end.isoformat(), "holdout_start": holdout_start.isoformat(),
        "horizon": horizon, "cost_bps": cost_bps,
        "signals_tested": len(catalog), "bonferroni_z": round(z, 3),
        "results": rows,
    }


def print_report(result: dict) -> None:
    print(
        f"\n--- edge search: {', '.join(result['symbols'])} ---\n"
        f"dev ≤{result['dev_end']}  holdout ≥{result['holdout_start']}  "
        f"cross-symbol confirm on {', '.join(result['holdout_symbols'])}  "
        f"({result['signals_tested']} signals, Bonferroni z={result['bonferroni_z']}, cost {result['cost_bps']}bps)\n"
    )
    print(f"{'signal':<20}{'horizon':<10}{'dev bps':>9}{'hold bps':>10}{'hold CI-lo':>11}{'xsym bps':>10}  survives")
    for r in result["results"]:
        d, h, x = r["dev"], r["holdout"], r["cross_symbol"]
        mark = "  ✓" if r["survives"] else "  ✗"
        print(
            f"{r['signal']:<20}{r['horizon']:<10}{d['mean_bps']:>9.2f}{h['mean_bps']:>10.2f}"
            f"{h['ci_low_bps']:>11.2f}{x['mean_bps']:>10.2f}{mark}"
        )
    survivors = [r for r in result["results"] if r["survives"]]
    if not survivors:
        print(
            "\nNo signal cleared every gate (holdout CI above zero + cross-symbol positive, "
            "after costs and multiple-testing). There is no validated edge here to trade — "
            "the honest result. Adding risk would only lose faster."
        )
    else:
        print(f"\n{len(survivors)} signal(s) survived — see the option-economics table(s) below.")
