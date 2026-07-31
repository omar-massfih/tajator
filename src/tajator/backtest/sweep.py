"""Parameter sweep with an out-of-sample holdout: find a genuinely profitable
ORB variant instead of curve-fitting one.

Every variant is scored on a development window, ranked by pooled per-trade
underlying expectancy (with a minimum-trades floor to reject lucky small
samples), and the best are then re-run on an untouched holdout window. A variant
"survives" only if it stays positive out-of-sample. Runs fully offline against
the cached bars (``cached_only``, ``underlying_only``) — no IB connection.

Underlying-points expectancy is a *candidate* signal, not a proven options edge:
a survivor still has to beat option decay/spread before it is real profit.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path

from ..config import Settings
from .runner import run_backtest

log = logging.getLogger(__name__)

# Grid kept deliberately small so a parallel sweep finishes in minutes — each
# variant is a full-graph backtest over the whole window (~1s per trading day),
# so breadth is expensive. Each "exit" entry is an (exit_mode, runner_target_r) pair.
DEFAULT_GRID: dict[str, list] = {
    "orb_window_minutes": [15, 30],
    "orb_breakout_buffer_pct": [0.0005],
    "orb_min_relative_volume": [0.0, 1.3],
    "orb_min_breakout_range_atr": [0.0, 1.0],
    "exit": [("scale", 3.0), ("let_run", 2.0), ("let_run", 3.0), ("let_run", 5.0)],
}

_GRID_KEYS = [
    "orb_window_minutes", "orb_breakout_buffer_pct",
    "orb_min_relative_volume", "orb_min_breakout_range_atr", "exit",
]


def _variants(grid: dict[str, list]):
    for combo in itertools.product(*(grid[k] for k in _GRID_KEYS)):
        overrides = dict(zip(_GRID_KEYS, combo))
        exit_mode, target_r = overrides.pop("exit")
        overrides["exit_mode"] = exit_mode
        overrides["runner_target_r"] = target_r
        yield overrides


def _pooled_task(args: tuple) -> dict:
    """Picklable wrapper so a process pool can evaluate one variant/window."""
    settings, overrides, symbols, start, end, cache_dir = args
    return _pooled(settings, overrides, symbols, start, end, cache_dir)


def _run_phase(
    settings: Settings, variant_overrides: list[dict], symbols: list[str],
    start: date, end: date, cache_dir: Path, workers: int,
) -> list[dict]:
    """Evaluate every variant on one window, in parallel across processes."""
    tasks = [(settings, ov, symbols, start, end, cache_dir) for ov in variant_overrides]
    if workers <= 1 or len(tasks) == 1:
        return [_pooled_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_pooled_task, tasks))


def _pooled(
    settings: Settings, overrides: dict, symbols: list[str],
    start: date, end: date, cache_dir: Path,
) -> dict:
    """Run one variant across symbols on one window; pool into per-trade expectancy."""
    variant = settings.model_copy(update=overrides)
    total_points, total_trades = 0.0, 0
    per_symbol: dict[str, dict] = {}
    for symbol in symbols:
        report = run_backtest(
            symbol, start, end, variant, ib=None, cache_dir=cache_dir,
            underlying_only=True, cached_only=True, experiment="sweep",
        )
        total_points += report.total_underlying_points
        total_trades += report.total_trades
        per_symbol[symbol] = {
            "trades": report.total_trades,
            "points": report.total_underlying_points,
            "expectancy": report.underlying_expectancy,
            "max_dd": report.max_underlying_drawdown,
        }
    expectancy = round(total_points / total_trades, 4) if total_trades else 0.0
    return {
        "overrides": overrides,
        "trades": total_trades,
        "points": round(total_points, 4),
        "expectancy": expectancy,
        "per_symbol": per_symbol,
    }


def run_sweep(
    symbols: list[str],
    dev_start: date, dev_end: date,
    holdout_start: date, holdout_end: date,
    settings: Settings,
    cache_dir: Path,
    *,
    min_trades: int = 30,
    top_k: int = 8,
    grid: dict[str, list] | None = None,
    workers: int | None = None,
) -> dict:
    grid = grid or DEFAULT_GRID
    workers = workers if workers is not None else (os.cpu_count() or 1)
    variant_overrides = list(_variants(grid))
    log.info("sweep: %d variants x %d symbols on dev window (%d workers)",
             len(variant_overrides), len(symbols), workers)

    dev_results = _run_phase(
        settings, variant_overrides, symbols, dev_start, dev_end, cache_dir, workers
    )

    eligible = [r for r in dev_results if r["trades"] >= min_trades]
    eligible.sort(key=lambda r: r["expectancy"], reverse=True)

    top = eligible[:top_k]
    log.info("sweep: validating top %d on holdout window", len(top))
    holdout_results = _run_phase(
        settings, [r["overrides"] for r in top], symbols, holdout_start, holdout_end, cache_dir, workers
    )

    ranked = []
    for r, h in zip(top, holdout_results):
        ranked.append({
            "overrides": r["overrides"],
            "dev": {"trades": r["trades"], "points": r["points"], "expectancy": r["expectancy"]},
            "holdout": {"trades": h["trades"], "points": h["points"], "expectancy": h["expectancy"]},
            "survived": h["trades"] >= min_trades and h["expectancy"] > 0,
        })

    return {
        "symbols": symbols,
        "dev_window": [dev_start.isoformat(), dev_end.isoformat()],
        "holdout_window": [holdout_start.isoformat(), holdout_end.isoformat()],
        "min_trades": min_trades,
        "variants_tested": len(dev_results),
        "variants_eligible": len(eligible),
        "ranked": ranked,
    }


def _label(overrides: dict) -> str:
    return (
        f"win={overrides['orb_window_minutes']:>2} "
        f"buf={overrides['orb_breakout_buffer_pct']:.4f} "
        f"vol={overrides['orb_min_relative_volume']:.1f} "
        f"rng={overrides['orb_min_breakout_range_atr']:.1f} "
        f"exit={overrides['exit_mode']:>7}"
        + (f"@{overrides['runner_target_r']:.0f}R" if overrides["exit_mode"] == "let_run" else "    ")
    )


def print_sweep(result: dict) -> None:
    print(
        f"\n--- ORB sweep: {', '.join(result['symbols'])} ---\n"
        f"dev {result['dev_window'][0]}→{result['dev_window'][1]}  "
        f"holdout {result['holdout_window'][0]}→{result['holdout_window'][1]}  "
        f"(tested {result['variants_tested']}, {result['variants_eligible']} met min_trades={result['min_trades']})"
    )
    if not result["ranked"]:
        print("no variant produced enough trades to rank.")
        return
    print(f"\n{'variant':<52}{'dev exp':>9}{'dev n':>7}{'hold exp':>10}{'hold n':>8}  survives")
    for row in result["ranked"]:
        d, h = row["dev"], row["holdout"]
        mark = "  ✓" if row["survived"] else "  ✗"
        print(
            f"{_label(row['overrides']):<52}{d['expectancy']:>9.3f}{d['trades']:>7}"
            f"{h['expectancy']:>10.3f}{h['trades']:>8}{mark}"
        )
    survivors = [r for r in result["ranked"] if r["survived"]]
    if survivors:
        best = survivors[0]
        print(
            f"\nbest holdout survivor: {_label(best['overrides'])}  "
            f"(holdout expectancy {best['holdout']['expectancy']:+.3f} pts/trade over {best['holdout']['trades']} trades)"
        )
        print("Reminder: underlying-points edge must still beat option decay/spread to be real profit.")
    else:
        print("\nNo variant survived the holdout. There is no edge here to size — adding risk would only lose faster.")


def write_sweep(result: dict, log_dir: Path) -> Path:
    out_dir = log_dir / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"orb-sweep-{stamp}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    return path
