"""Monthly cross-sectional momentum basket — the tradeable form of the validated
edge, reported honestly.

Each rebalance, rank the universe by 12-1 momentum (return over the past ~year,
skipping the most recent month to avoid short-term reversal), hold an equal-weight
basket of the top decile until the next rebalance, and deduct turnover costs. We
report three equity curves:

- **momentum** — the long top-decile basket (raw return, includes market + survivorship),
- **benchmark** — equal-weight of the whole universe (same survivor set),
- **premium** — momentum minus benchmark, and a market-neutral **long-short** (top − bottom).

The premium and long-short are the honest edge: they net out the market and most of the
survivorship bias (both legs are drawn from the same universe), so their size — not the raw
return — is the number to trust.
"""

from __future__ import annotations

from pathlib import Path
from statistics import fmean, pstdev

from .signals import load_daily

REBALANCE_DAYS = 21     # ~monthly
LOOKBACK = 252          # ~12 months
SKIP = 21               # skip the most recent month (12-1 momentum)
DECILE = 0.1
COST_BPS = 2.0          # per share round-trip; applied to turnover


def _panel(symbols, daily_dir: Path):
    """Return (calendar, closes) where calendar is the sorted master date list and
    closes maps symbol -> {date: close}."""
    closes = {}
    dates = set()
    for s in symbols:
        bars = load_daily(daily_dir, s)
        if len(bars) < LOOKBACK + SKIP + REBALANCE_DAYS:
            continue
        closes[s] = {b.ts.date(): b.close for b in bars}
        dates.update(closes[s])
    return sorted(dates), closes


def _period_return(closes_s, d0, d1):
    a, b = closes_s.get(d0), closes_s.get(d1)
    return (b / a - 1.0) if a and b and a > 0 else None


def _momentum(closes_s, cal, ti):
    """12-1 momentum at calendar index ti: return from ti-LOOKBACK to ti-SKIP."""
    if ti - LOOKBACK < 0:
        return None
    return _period_return(closes_s, cal[ti - LOOKBACK], cal[ti - SKIP])


def _basket_return(members, closes, d0, d1):
    rets = [r for s in members if (r := _period_return(closes[s], d0, d1)) is not None]
    return fmean(rets) if rets else 0.0


def _curve_stats(period_rets, per_year):
    """Chain period returns into CAGR / annualized Sharpe / max drawdown."""
    equity, peak, max_dd = 1.0, 1.0, 0.0
    for r in period_rets:
        equity *= (1 + r)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
    n = len(period_rets)
    years = n / per_year if per_year else 0
    cagr = (equity ** (1 / years) - 1) if years > 0 and equity > 0 else 0.0
    mean, sd = (fmean(period_rets), pstdev(period_rets)) if n > 1 else (0.0, 0.0)
    sharpe = (mean / sd * (per_year ** 0.5)) if sd > 0 else 0.0
    return {
        "total_return_pct": round((equity - 1) * 100, 1),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "periods": n,
    }


def run_momentum(symbols, *, daily_dir: Path, rebalance_days=REBALANCE_DAYS,
                 decile=DECILE, cost_bps=COST_BPS) -> dict:
    cal, closes = _panel(symbols, daily_dir)
    if not closes:
        return {"error": "no symbols with enough history"}
    per_year = 252 / rebalance_days

    rebal_i = list(range(LOOKBACK + SKIP, len(cal) - 1, rebalance_days))
    mom_rets, bench_rets, ls_rets = [], [], []
    by_year: dict[int, list[float]] = {}
    prev_top = set()
    turnovers = []

    for ti in rebal_i:
        scored = [(m, s) for s in closes if (m := _momentum(closes[s], cal, ti)) is not None
                  and cal[ti] in closes[s]]
        if len(scored) < 10:
            continue
        scored.sort()
        k = max(1, int(len(scored) * decile))
        losers = {s for _, s in scored[:k]}
        winners = {s for _, s in scored[-k:]}
        d0, d1 = cal[ti], cal[min(ti + rebalance_days, len(cal) - 1)]

        turnover = len(winners - prev_top) / max(1, len(winners))
        turnovers.append(turnover)
        cost = turnover * cost_bps / 1e4 * 2  # enter + later exit the changed names
        prev_top = winners

        mom = _basket_return(winners, closes, d0, d1) - cost
        bench = _basket_return(list(closes), closes, d0, d1)
        short = _basket_return(losers, closes, d0, d1)
        mom_rets.append(mom)
        bench_rets.append(bench)
        ls_rets.append(mom - short)  # long winners, short losers (market-neutral)
        by_year.setdefault(d0.year, []).append(mom - bench)

    premium = [m - b for m, b in zip(mom_rets, bench_rets)]
    return {
        "symbols": len(closes),
        "rebalance_days": rebalance_days, "lookback": LOOKBACK, "skip": SKIP,
        "decile": decile, "cost_bps": cost_bps, "avg_turnover": round(fmean(turnovers), 3) if turnovers else 0,
        "momentum": _curve_stats(mom_rets, per_year),
        "benchmark": _curve_stats(bench_rets, per_year),
        "premium": _curve_stats(premium, per_year),
        "long_short": _curve_stats(ls_rets, per_year),
        "premium_by_year": {y: round(sum(v) * 100, 1) for y, v in sorted(by_year.items())},
    }


def print_momentum(r: dict) -> None:
    if "error" in r:
        print(r["error"])
        return
    print(
        f"\n--- monthly momentum basket: {r['symbols']} symbols, "
        f"top {r['decile']:.0%} by {r['lookback']}-{r['skip']}d momentum, "
        f"rebal {r['rebalance_days']}d, cost {r['cost_bps']}bps, avg turnover {r['avg_turnover']:.0%} ---\n"
    )
    print(f"{'leg':<12}{'total %':>10}{'CAGR %':>9}{'Sharpe':>8}{'max DD %':>10}")
    for leg in ("momentum", "benchmark", "premium", "long_short"):
        s = r[leg]
        print(f"{leg:<12}{s['total_return_pct']:>10.1f}{s['cagr_pct']:>9.2f}{s['sharpe']:>8.2f}{s['max_drawdown_pct']:>10.1f}")
    print("\npremium (momentum − benchmark) by year, %:")
    print("  " + "  ".join(f"{y}:{v:+.1f}" for y, v in r["premium_by_year"].items()))
    prem = r["premium"]
    if prem["cagr_pct"] > 0 and prem["sharpe"] > 0.3:
        print(
            f"\nHonest edge: the momentum PREMIUM is +{prem['cagr_pct']:.1f}%/yr "
            f"(Sharpe {prem['sharpe']:.2f}) net of turnover cost — modest and real, "
            "not the raw return. This is the number to trust (market + much survivorship netted out)."
        )
    else:
        print("\nThe premium over the universe is weak/negative once market is netted out — the raw "
              "momentum return was mostly market + survivorship, not a tradeable edge.")
