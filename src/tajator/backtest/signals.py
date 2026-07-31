"""Edge-search engine: test documented, structural signals for a real directional
edge — offline, on the cached underlying bars, with rigorous validation and
optional stop-loss / profit-protection.

Every candidate faces four gates: a pre-declared temporal holdout, a cross-symbol
holdout (found on some names, confirmed on unseen ones), a Bonferroni correction
across every signal tried, and day-clustered standard errors (a trading day, not a
trade, is the unit of evidence). A signal "survives" only if its holdout expectancy
stays positive with a Bonferroni-adjusted 95% lower bound above zero, and it is also
positive out-of-sample on symbols it was never fit on.

Signals are long-biased equity edges (overnight drift, short-term reversal,
time-series momentum, turn-of-month), their high-power **cross-sectional** forms
(rank the universe, long the losers / winners as a basket), and a few conditioned
intraday effects. Each is evaluated under three exit rules — no stop, a fixed
stop-loss, and a trailing profit-lock — using bar highs/lows to detect breaches, so
we learn which stop actually helps rather than assuming.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from math import log
from pathlib import Path
from statistics import NormalDist, fmean, pstdev

from ..models import Bar
from .data import ET, _read_csv

DEFAULT_COST_BPS = 2.0    # round-trip stock spread+commission on a liquid name
STOP_MODES = ("none", "fixed", "trailing")
DEFAULT_STOP_PCT = 0.03   # fixed stop distance from entry
DEFAULT_TRAIL_PCT = 0.03  # trailing profit-lock give-back
XS_DECILE = 0.1           # cross-sectional basket = top/bottom 10%

SIGNAL_HORIZON_DAYS = {
    "overnight": 1, "intraday": 1, "reversal_1d": 1, "reversal_5d": 1,
    "momentum_ma20": 1, "momentum_ma200": 1, "turn_of_month": 3,
    "xs_reversal_1d": 1, "xs_reversal_5d": 1, "xs_momentum": 21,
    "xs_lowvol": 21, "xs_highvol": 21, "xs_52w_high": 21,
    "oversold": 3, "sell_in_may": 21,
    "gap_fill": 1, "first_hour_reversal": 1, "last_hour_drift": 1,
}


@dataclass(frozen=True)
class Position:
    """One entered trade, held over `hold` bars (may be empty for an overnight gap).
    `horizon_exit` is the fallback exit price if no stop is hit."""
    entry_day: date
    symbol: str
    direction: int            # +1 long, -1 short
    entry: float
    horizon_exit: float
    hold: tuple = ()          # bars whose high/low can trigger a stop


@dataclass(frozen=True)
class Trade:
    day: date
    symbol: str
    ret: float                # net, direction-adjusted return fraction


def realize(pos: Position, mode: str, stop_pct: float, trail_pct: float, cost_bps: float) -> Trade:
    """Return the trade's net return under an exit rule, walking hold bars for stops."""
    exit_px = pos.horizon_exit
    long = pos.direction > 0
    if pos.hold and mode != "none":
        if mode == "fixed":
            stop = pos.entry * (1 - stop_pct) if long else pos.entry * (1 + stop_pct)
            for b in pos.hold:
                if (b.low <= stop) if long else (b.high >= stop):
                    exit_px = stop
                    break
        elif mode == "trailing":
            extreme = pos.entry
            for b in pos.hold:
                stop = extreme * (1 - trail_pct) if long else extreme * (1 + trail_pct)
                if (b.low <= stop) if long else (b.high >= stop):   # check prior stop first (conservative)
                    exit_px = stop
                    break
                extreme = max(extreme, b.high) if long else min(extreme, b.low)
    ret = pos.direction * (exit_px / pos.entry - 1.0) - cost_bps / 1e4 if pos.entry > 0 else 0.0
    return Trade(pos.entry_day, pos.symbol, ret)


# --------------------------------------------------------------------------- #
# Per-symbol daily signals — fn(daily_bars, symbol) -> list[Position]. Causal:
# every entry uses only information available at or before the entry bar.
# --------------------------------------------------------------------------- #

def _long(bars, i, symbol, exit_i, direction=1):
    exit_i = min(exit_i, len(bars) - 1)
    return Position(bars[i].ts.date(), symbol, direction, bars[i].close,
                    bars[exit_i].close, tuple(bars[i + 1:exit_i + 1]))


def sig_overnight(bars, symbol):
    """Long the overnight gap: buy at close, sell at next open (no intra-hold bars)."""
    return [Position(bars[i].ts.date(), symbol, 1, bars[i].close, bars[i + 1].open, ())
            for i in range(len(bars) - 1)]


def sig_intraday(bars, symbol):
    """Long the session: buy at open, sell at close (same-day low can stop)."""
    return [Position(b.ts.date(), symbol, 1, b.open, b.close, (b,)) for b in bars]


def sig_reversal_1d(bars, symbol):
    return [_long(bars, i, symbol, i + 1) for i in range(1, len(bars) - 1)
            if bars[i].close < bars[i - 1].close]


def sig_reversal_5d(bars, symbol):
    return [_long(bars, i, symbol, i + 1) for i in range(5, len(bars) - 1)
            if bars[i].close < bars[i - 5].close]


def _sma(bars, i, window):
    return fmean(b.close for b in bars[i + 1 - window:i + 1]) if i + 1 >= window else None


def sig_momentum_ma20(bars, symbol):
    return [_long(bars, i, symbol, i + 1) for i in range(len(bars) - 1)
            if (_sma(bars, i, 20) or 1e18) < bars[i].close]


def sig_momentum_ma200(bars, symbol):
    return [_long(bars, i, symbol, i + 1) for i in range(len(bars) - 1)
            if (_sma(bars, i, 200) or 1e18) < bars[i].close]


def sig_turn_of_month(bars, symbol):
    return [_long(bars, i, symbol, i + 3) for i in range(len(bars) - 1)
            if bars[i + 1].ts.month != bars[i].ts.month]


DAILY_SIGNALS = {
    "overnight": sig_overnight, "intraday": sig_intraday,
    "reversal_1d": sig_reversal_1d, "reversal_5d": sig_reversal_5d,
    "momentum_ma20": sig_momentum_ma20, "momentum_ma200": sig_momentum_ma200,
    "turn_of_month": sig_turn_of_month,
}


# --------------------------------------------------------------------------- #
# Cross-sectional signals — fn(series) -> list[Position], where series maps
# symbol -> its sorted daily bars. Each day, rank the universe and trade a basket.
# --------------------------------------------------------------------------- #

def _xs_rank(series, score_fn, min_i, horizon, top, decile=XS_DECILE):
    """Rank the universe each day by `score_fn(bars, i)` and long the top (or bottom)
    `decile` as an equal-weight basket held `horizon` sessions. Causal: `score_fn`
    may only look at bars up to index i."""
    idx = {s: {b.ts.date(): i for i, b in enumerate(bars)} for s, bars in series.items()}
    all_days = sorted({d for m in idx.values() for d in m})
    positions = []
    for d in all_days:
        scored = []
        for s, bars in series.items():
            i = idx[s].get(d)
            if i is None or i < min_i or i >= len(bars) - 1:
                continue
            sc = score_fn(bars, i)
            if sc is not None:
                scored.append((sc, s, i))
        if len(scored) < 3:
            continue
        scored.sort(reverse=top)  # top=True -> highest score first
        k = max(1, int(len(scored) * decile))
        for _, s, i in scored[:k]:
            positions.append(_long(series[s], i, s, i + horizon))
    return positions


def _ret(lb):
    return lambda b, i: b[i].close / b[i - lb].close - 1.0


def _vol(w=60):
    def f(b, i):
        r = [log(b[j].close / b[j - 1].close) for j in range(i - w + 1, i + 1) if b[j - 1].close > 0]
        return pstdev(r) if len(r) > 1 else None
    return f


def _near_52w_high(b, i):
    hi = max(x.close for x in b[i - 252:i + 1])
    return b[i].close / hi if hi > 0 else None


def xs_reversal_1d(series):
    return _xs_rank(series, _ret(1), 1, 1, top=False)


def xs_reversal_5d(series):
    return _xs_rank(series, _ret(5), 5, 1, top=False)


def xs_momentum(series):
    return _xs_rank(series, _ret(252), 252, 21, top=True)


def xs_lowvol(series):          # low-volatility anomaly: long the calmest names
    return _xs_rank(series, _vol(60), 60, 21, top=False)


def xs_highvol(series):         # its opposite, to check the sign
    return _xs_rank(series, _vol(60), 60, 21, top=True)


def xs_52w_high(series):        # 52-week-high effect (George & Hwang)
    return _xs_rank(series, _near_52w_high, 252, 21, top=True)


XS_SIGNALS = {
    "xs_reversal_1d": xs_reversal_1d, "xs_reversal_5d": xs_reversal_5d, "xs_momentum": xs_momentum,
    "xs_lowvol": xs_lowvol, "xs_highvol": xs_highvol, "xs_52w_high": xs_52w_high,
}


# Two more per-symbol families: oversold mean-reversion and calendar seasonality.

def sig_oversold(bars, symbol):
    """Buy a >=5% 3-day drop, hold 3 sessions (oversold bounce)."""
    return [_long(bars, i, symbol, i + 3) for i in range(3, len(bars) - 1)
            if bars[i].close / bars[i - 3].close - 1 <= -0.05]


def sig_sell_in_may(bars, symbol):
    """Seasonality: long only the Nov-Apr 'good' months, ~21-day holds."""
    return [_long(bars, i, symbol, i + 21) for i in range(1, len(bars) - 1)
            if bars[i].ts.month in (11, 12, 1, 2, 3, 4) and bars[i - 1].ts.month != bars[i].ts.month]


DAILY_SIGNALS.update({"oversold": sig_oversold, "sell_in_may": sig_sell_in_may})


# --------------------------------------------------------------------------- #
# Intraday session signals — fn(sessions, symbol) -> list[Position].
# --------------------------------------------------------------------------- #

def _rth(bars):
    return [b for b in bars if (b.ts.astimezone(ET).hour, b.ts.astimezone(ET).minute) >= (9, 30)
            and b.ts.astimezone(ET).hour < 16]


def sig_gap_fill(sessions, symbol):
    out = []
    for k in range(1, len(sessions)):
        prev_close = sessions[k - 1][1][-1].close
        day, bars = sessions[k]
        if not bars or prev_close <= 0:
            continue
        gap = (bars[0].open - prev_close) / prev_close
        if abs(gap) < 0.005:
            continue
        direction = -1 if gap > 0 else 1
        out.append(Position(day, symbol, direction, bars[0].open, bars[-1].close, tuple(bars)))
    return out


def sig_first_hour_reversal(sessions, symbol):
    out = []
    for day, bars in sessions:
        first = next((b for b in bars if (b.ts.astimezone(ET).hour, b.ts.astimezone(ET).minute) >= (10, 30)), None)
        if first is None or len(bars) < 90:
            continue
        fh = (first.open - bars[0].open) / bars[0].open if bars[0].open else 0.0
        if abs(fh) < 0.003:
            continue
        rest = [b for b in bars if b.ts >= first.ts]
        out.append(Position(day, symbol, -1 if fh > 0 else 1, first.open, bars[-1].close, tuple(rest)))
    return out


def sig_last_hour_drift(sessions, symbol):
    out = []
    for day, bars in sessions:
        rest = [b for b in bars if b.ts.astimezone(ET).hour >= 15]
        if not rest:
            continue
        out.append(Position(day, symbol, 1, rest[0].open, bars[-1].close, tuple(rest)))
    return out


INTRADAY_SIGNALS = {
    "gap_fill": sig_gap_fill, "first_hour_reversal": sig_first_hour_reversal,
    "last_hour_drift": sig_last_hour_drift,
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_daily(daily_dir: Path, symbol: str) -> list[Bar]:
    path = daily_dir / f"{symbol}.csv"
    return sorted(_read_csv(path), key=lambda b: b.ts) if path.exists() else []


def load_sessions(cache_dir: Path, symbol: str):
    sym_dir = cache_dir / symbol
    if not sym_dir.is_dir():
        return []
    sessions = []
    for path in sorted(sym_dir.glob("????-??-??.csv")):
        rth = _rth(sorted(_read_csv(path), key=lambda b: b.ts))
        if rth:
            sessions.append((rth[0].ts.astimezone(ET).date(), rth))
    return sessions


# --------------------------------------------------------------------------- #
# Statistics — day-clustered mean with a Bonferroni-adjusted CI
# --------------------------------------------------------------------------- #

def clustered_stats(trades, *, z: float) -> dict:
    by_day = defaultdict(list)
    for t in trades:
        by_day[t.day].append(t.ret)
    daily = [fmean(v) for v in by_day.values()]
    n = len(daily)
    if n == 0:
        return {"trades": 0, "days": 0, "mean_bps": 0.0, "ci_low_bps": 0.0, "t": 0.0}
    mean = fmean(daily)
    se = (pstdev(daily) / (n ** 0.5)) if n > 1 else float("inf")
    return {
        "trades": len(trades), "days": n,
        "mean_bps": round(mean * 1e4, 2),
        "ci_low_bps": round((mean - z * se) * 1e4, 2),
        "t": round(mean / se, 2) if se not in (0.0, float("inf")) else 0.0,
    }


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

def _split_symbols(symbols):
    return ([s for i, s in enumerate(symbols) if i % 2 == 0],
            [s for i, s in enumerate(symbols) if i % 2 == 1])


def _non_overlapping(positions):
    """Keep only non-overlapping positions per symbol. Multi-day holds otherwise
    autocorrelate (a 21-day momentum position overlaps its 20 neighbours), which
    day-clustering does not absorb and which inflates the t-stat."""
    by_sym = defaultdict(list)
    for p in positions:
        by_sym[p.symbol].append(p)
    kept = []
    for ps in by_sym.values():
        ps.sort(key=lambda p: p.entry_day)
        last_exit = None
        for p in ps:
            if last_exit is not None and p.entry_day < last_exit:
                continue
            kept.append(p)
            last_exit = p.hold[-1].ts.date() if p.hold else p.entry_day
    return kept


def _evaluate(positions, hold_syms, holdout_start, z, stop_pct, trail_pct, cost_bps):
    """Best-of stop-mode evaluation with dev/holdout/cross-symbol clustered stats."""
    best = None
    per_mode = {}
    for mode in STOP_MODES:
        trades = [realize(p, mode, stop_pct, trail_pct, cost_bps) for p in positions]
        dev = [t for t in trades if t.day < holdout_start]
        hold = [t for t in trades if t.day >= holdout_start]
        cross = [t for t in hold if t.symbol in hold_syms]
        row = {
            "mode": mode,
            "dev": clustered_stats(dev, z=z),
            "holdout": clustered_stats(hold, z=z),
            "cross_symbol": clustered_stats(cross, z=z),
        }
        per_mode[mode] = row
        # pick the mode with the strongest holdout t-stat
        if best is None or row["holdout"]["t"] > best["holdout"]["t"]:
            best = row
    survives = (
        best["holdout"]["days"] >= 30 and best["holdout"]["ci_low_bps"] > 0
        and best["cross_symbol"]["days"] >= 20 and best["cross_symbol"]["mean_bps"] > 0
    )
    return {"best_mode": best["mode"], **best, "per_mode": per_mode, "survives": survives}


def run_edge_search(
    symbols, *, daily_dir: Path, intraday_dir: Path, dev_end: date, holdout_start: date,
    horizon: str = "daily", cost_bps: float = DEFAULT_COST_BPS,
    stop_pct: float = DEFAULT_STOP_PCT, trail_pct: float = DEFAULT_TRAIL_PCT,
) -> dict:
    daily = {s: load_daily(daily_dir, s) for s in symbols}
    daily = {s: b for s, b in daily.items() if b}
    sessions = ({s: load_sessions(intraday_dir, s) for s in symbols}
                if horizon in ("intraday", "both") else {})
    dev_syms, hold_syms = _split_symbols(symbols)

    catalog = {}
    if horizon in ("daily", "both"):
        catalog.update({n: ("daily", f) for n, f in DAILY_SIGNALS.items()})
        catalog.update({n: ("xs", f) for n, f in XS_SIGNALS.items()})
    if horizon in ("intraday", "both"):
        catalog.update({n: ("intraday", f) for n, f in INTRADAY_SIGNALS.items()})
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * max(1, len(catalog))))

    rows = []
    for name, (kind, fn) in catalog.items():
        if kind == "daily":
            positions = [p for s, b in daily.items() for p in fn(b, s)]
        elif kind == "xs":
            positions = fn(daily)
        else:
            positions = [p for s in symbols for p in fn(sessions.get(s, []), s)]
        positions = _non_overlapping(positions)
        row = _evaluate(positions, set(hold_syms), holdout_start, z, stop_pct, trail_pct, cost_bps)
        rows.append({"signal": name, "horizon": kind, **row})

    rows.sort(key=lambda r: r["holdout"]["ci_low_bps"], reverse=True)
    return {
        "symbols": symbols, "dev_symbols": dev_syms, "holdout_symbols": hold_syms,
        "dev_end": dev_end.isoformat(), "holdout_start": holdout_start.isoformat(),
        "horizon": horizon, "cost_bps": cost_bps, "stop_pct": stop_pct, "trail_pct": trail_pct,
        "signals_tested": len(catalog), "bonferroni_z": round(z, 3), "results": rows,
    }


def print_report(result: dict) -> None:
    print(
        f"\n--- edge search: {len(result['symbols'])} symbols ---\n"
        f"dev ≤{result['dev_end']}  holdout ≥{result['holdout_start']}  "
        f"cross-symbol confirm on {len(result['holdout_symbols'])} unseen names  "
        f"({result['signals_tested']} signals, Bonferroni z={result['bonferroni_z']}, "
        f"cost {result['cost_bps']}bps, stops fixed {result['stop_pct']:.0%}/trail {result['trail_pct']:.0%})\n"
    )
    print(f"{'signal':<18}{'kind':<9}{'stop':<9}{'hold bps':>9}{'hold t':>8}{'hold CI-lo':>11}{'xsym bps':>10}  ok")
    for r in result["results"]:
        h, x = r["holdout"], r["cross_symbol"]
        mark = "  ✓" if r["survives"] else "  ✗"
        print(
            f"{r['signal']:<18}{r['horizon']:<9}{r['best_mode']:<9}{h['mean_bps']:>9.2f}"
            f"{h['t']:>8.2f}{h['ci_low_bps']:>11.2f}{x['mean_bps']:>10.2f}{mark}"
        )
    survivors = [r for r in result["results"] if r["survives"]]
    if not survivors:
        print(
            "\nNo signal cleared every gate (holdout CI above zero + cross-symbol positive, "
            "after costs, stops, and multiple-testing). At this universe size that is a "
            "high-power, honest 'no confirmed edge' — do not trade it."
        )
    else:
        print(f"\n{len(survivors)} signal(s) survived — the first validated stock edge(s).")
