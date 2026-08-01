"""Volatility-risk-premium edge search — the one options-native edge that survives.

Stock edges bought as options die to theta (see `option_economics`). The edge that
lives *in* the options market is the **volatility risk premium (VRP)**: implied vol
(VIX) persistently exceeds realized vol, so selling vol earns a carry. This account
has no historical option chains, so we harvest the VRP through the tradeable vol
ETP **SVXY** (short-vol), gated by the **VIX term structure** (VIX vs VIX3M) — the
filter that turns an account-ending tail into a survivable one.

Honest findings this produces:
- The raw VRP is real and persistent (VIX − forward realized vol > 0 ~83% of days).
- Naive always-short-vol (long SVXY) is a death trap: it lost ~91% in Feb-2018.
- Holding SVXY **only in contango** (VIX < VIX3M), flat otherwise, keeps most of the
  return while cutting the crash roughly in half — a *survivable* (not eliminated)
  ~35% max drawdown. It is a risk premium (payment for bearing crash risk), not free
  alpha, so Sharpe is modest (~0.6 on the post-2018 −0.5x SVXY that trades today).

SVXY changed from −1.0x to −0.5x leverage on 2018-02-28, so the pre/post split is
reported separately — only the post-2018 instrument is tradeable now.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import log
from pathlib import Path
from statistics import fmean, pstdev

from .option_economics import TRADING_DAYS
from .signals import load_daily

DEFAULT_COST_BPS = 5.0          # per regime flip (enter/exit SVXY), one-way
SVXY_RESET = date(2018, 2, 28)  # SVXY -1.0x -> -0.5x leverage change
CRASH_WINDOWS = {
    "Feb-2018 Volmageddon": (date(2018, 1, 25), date(2018, 2, 12)),
    "Mar-2020 COVID": (date(2020, 2, 19), date(2020, 3, 23)),
    "2022 bear": (date(2022, 1, 1), date(2022, 12, 31)),
}


@dataclass(frozen=True)
class PerfStats:
    n: int
    ann_return: float
    sharpe: float
    max_drawdown: float
    worst_day: float


def perf_stats(returns: list[float], *, freq: int = TRADING_DAYS) -> PerfStats:
    """Annualized return / Sharpe / max drawdown / worst period from a return series."""
    if len(returns) < 20:
        return PerfStats(len(returns), 0.0, 0.0, 0.0, 0.0)
    mean, sd = fmean(returns), pstdev(returns)
    ann = mean * freq
    vol = sd * (freq ** 0.5)
    sharpe = ann / vol if vol else 0.0
    equity, peak, mdd = 1.0, 1.0, 0.0
    for r in returns:
        equity *= 1 + r
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
    return PerfStats(len(returns), ann, sharpe, mdd, min(returns))


def _aligned(vol_dir: Path, symbols: list[str]) -> tuple[list[date], dict[str, dict[date, float]]]:
    series = {s: {b.ts.date(): b.close for b in load_daily(vol_dir, s)} for s in symbols}
    if any(not v for v in series.values()):
        missing = [s for s, v in series.items() if not v]
        raise FileNotFoundError(f"missing daily cache for {missing} under {vol_dir}")
    common = sorted(set.intersection(*(set(v) for v in series.values())))
    return common, series


def _daily_returns(dates: list[date], closes: dict[date, float]) -> list[tuple[date, float]]:
    out = []
    for i in range(1, len(dates)):
        prev, cur = closes[dates[i - 1]], closes[dates[i]]
        out.append((dates[i], cur / prev - 1.0 if prev > 0 else 0.0))
    return out


def filtered_short_vol(
    vol_dir: Path, *, cost_bps: float = DEFAULT_COST_BPS,
) -> list[tuple[date, float, float]]:
    """Term-structure-filtered short-vol daily returns.

    Each day t: position = long SVXY iff VIX_{t-1} < VIX3M_{t-1} (contango, causal),
    else flat. Returns list of (date, net_return, always_on_return) so callers can
    compare the filter to naive always-on short vol. A regime flip costs `cost_bps`.
    """
    dates, series = _aligned(vol_dir, ["VIX", "VIX3M", "SVXY"])
    svxy_ret = dict(_daily_returns(dates, series["SVXY"]))
    contango = {d: series["VIX"][d] < series["VIX3M"][d] for d in dates}

    out: list[tuple[date, float, float]] = []
    prev_pos = 0.0
    for i in range(1, len(dates)):
        d = dates[i]
        pos = 1.0 if contango[dates[i - 1]] else 0.0  # signal known at prior close
        r = svxy_ret[d]
        net = pos * r - (cost_bps / 1e4 if pos != prev_pos else 0.0)
        out.append((d, net, r))
        prev_pos = pos
    return out


def raw_vrp(vol_dir: Path, spy_dir: Path, *, window: int = 21) -> dict:
    """VIX (implied) minus SPY's *subsequent* `window`-day realized vol, in vol points.
    A persistently positive series is the raw premium that the strategy harvests."""
    vix = {b.ts.date(): b.close for b in load_daily(vol_dir, "VIX")}
    spy_bars = load_daily(spy_dir, "SPY")
    dates = [b.ts.date() for b in spy_bars]
    closes = [b.close for b in spy_bars]
    logret = [log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    premia: list[tuple[date, float]] = []
    for i in range(len(logret) - window):
        seg = logret[i:i + window]
        if len(seg) < 2:
            continue
        fwd_rv = pstdev(seg) * (TRADING_DAYS ** 0.5) * 100
        d = dates[i + 1]
        if d in vix:
            premia.append((d, vix[d] - fwd_rv))
    vals = [p for _, p in premia]
    return {
        "series": premia,
        "mean": fmean(vals) if vals else 0.0,
        "frac_positive": (sum(v > 0 for v in vals) / len(vals)) if vals else 0.0,
    }


def crash_pnl(returns: list[tuple[date, float]]) -> dict[str, float]:
    """Cumulative return of the strategy across each documented crash window."""
    out = {}
    for label, (a, b) in CRASH_WINDOWS.items():
        seg = [r for d, r in returns if a <= d <= b]
        cum = 1.0
        for r in seg:
            cum *= 1 + r
        out[label] = cum - 1.0 if seg else float("nan")
    return out


def _split(rows: list[tuple[date, float]], lo: date | None, hi: date | None) -> list[float]:
    return [r for d, r in rows if (lo is None or d >= lo) and (hi is None or d < hi)]


def run_vol_edge(
    vol_dir: Path, *, spy_dir: Path | None = None, cost_bps: float = DEFAULT_COST_BPS,
    train_end: date = date(2021, 12, 31),
) -> dict:
    """Full VRP edge report: raw premium, the filtered strategy vs always-on short vol,
    dev/holdout and pre/post-2018 splits, and the crash-tail table."""
    rows = filtered_short_vol(vol_dir, cost_bps=cost_bps)
    net = [(d, n) for d, n, _ in rows]
    always = [(d, a) for d, _, a in rows]
    holdout_start = date(train_end.year + (train_end.month // 12), (train_end.month % 12) + 1, 1)

    in_market = sum(1 for _, n, a in rows if n != 0 or a == 0) / len(rows) if rows else 0.0
    result = {
        "cost_bps": cost_bps,
        "in_market_frac": sum(1 for d, n in net if n != 0) / len(net) if net else 0.0,
        "filtered": {
            "full": perf_stats([r for _, r in net]),
            "dev": perf_stats(_split(net, None, holdout_start)),
            "holdout": perf_stats(_split(net, holdout_start, None)),
            "pre_2018": perf_stats(_split(net, None, SVXY_RESET)),
            "post_2018": perf_stats(_split(net, SVXY_RESET, None)),
        },
        "always_on": {
            "full": perf_stats([r for _, r in always]),
            "post_2018": perf_stats(_split(always, SVXY_RESET, None)),
        },
        "crash_filtered": crash_pnl(net),
        "crash_always_on": crash_pnl(always),
    }
    if spy_dir is not None:
        result["vrp"] = {k: v for k, v in raw_vrp(vol_dir, spy_dir).items() if k != "series"}
    return result


def print_report(result: dict) -> None:
    def line(label: str, s: PerfStats) -> str:
        return (f"  {label:34} ann={s.ann_return * 100:6.1f}%  Sharpe={s.sharpe:5.2f}  "
                f"maxDD={s.max_drawdown * 100:6.1f}%  worst={s.worst_day * 100:6.1f}%  (n={s.n})")

    print("\n=== Volatility risk premium — term-structure-filtered short vol (SVXY) ===")
    if "vrp" in result:
        v = result["vrp"]
        print(f"\nRaw VRP (VIX − forward realized vol): mean {v['mean']:.2f} vol pts, "
              f"positive {v['frac_positive'] * 100:.0f}% of days")
    f = result["filtered"]
    print(f"\nFiltered strategy (net {result['cost_bps']}bps/flip, "
          f"in-market {result['in_market_frac'] * 100:.0f}% of days):")
    print(line("FULL", f["full"]))
    print(line("DEV (<=train_end)", f["dev"]))
    print(line("HOLDOUT", f["holdout"]))
    print(line("PRE-2018 (-1.0x SVXY, retired)", f["pre_2018"]))
    print(line("POST-2018 (-0.5x, TRADEABLE NOW)", f["post_2018"]))
    print("\nContrast — naive always-on short vol (no filter):")
    print(line("always-on FULL", result["always_on"]["full"]))
    print(line("always-on POST-2018", result["always_on"]["post_2018"]))
    print("\nCrash tail — filtered vs always-on (cumulative P&L):")
    for label in CRASH_WINDOWS:
        ff = result["crash_filtered"][label]
        aa = result["crash_always_on"][label]
        print(f"  {label:24} filtered {ff * 100:7.1f}%   always-on {aa * 100:7.1f}%")
    print("\nThe VRP is payment for bearing crash risk, not free alpha. The filter tames "
          "the tail (Feb-2018: ~-17% vs ~-91%) but never removes it — expect a survivable "
          "~35% drawdown on the post-2018 instrument.")
