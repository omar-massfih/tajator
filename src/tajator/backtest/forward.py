"""Frozen forward-validation cohorts backed by timely TWS option capture."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config import Settings
from ..models import Bar
from .data import ET, ensure_underlying_bars
from .audit import _sample_stats
from .runner import _safe_experiment, _strategy_config, run_backtest

CAPTURE_PROTOCOL = {
    "version": 1,
    "chain": "actual TWS snapshot captured before replay",
    "base_pricing": "next option bar open with adverse costs",
    "panel": ["itm_1_near", "otm_1_near", "atm_next_expiry"],
    "panel_timing": "base timestamps and quantities",
}


def _execution_model(settings: Settings) -> dict:
    return {
        "price_source": "next option bar open (last bar close at EOD)",
        "modeled_half_spread_pct": settings.backtest_half_spread_pct,
        "slippage_cents_per_contract": settings.backtest_slippage_cents,
        "commission_per_contract": settings.backtest_commission_per_contract,
        "minimum_commission_per_order": settings.backtest_min_commission_per_order,
    }


def _source_fingerprint() -> str:
    """Fingerprint trading/backtest behavior, excluding reporting/orchestration code."""
    package_root = Path(__file__).resolve().parents[1]
    excluded = {
        "backtest/audit.py",
        "backtest/compare.py",
        "backtest/daily_signals.py",
        "backtest/forward.py",
        "backtest/historical_signals.py",
        "broker/shadow.py",
        "cli.py",
    }
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _definition(symbol: str, settings: Settings) -> dict:
    resolved = settings.for_symbol(symbol)
    definition = {
        "symbol": symbol,
        "strategy_config": _strategy_config(resolved),
        "execution_model": _execution_model(settings),
        "source_fingerprint": _source_fingerprint(),
        "capture_protocol": CAPTURE_PROTOCOL,
        "llm_mode": "disabled_deterministic",
    }
    definition["fingerprint"] = hashlib.sha256(
        json.dumps(definition, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return definition


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    """Recursively normalize values, including date-keyed report mappings."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def session_quality(bars: list[Bar], day: date) -> dict:
    """Prove a regular or standard 13:00 ET early-close session is complete."""
    regular = [
        bar for bar in bars
        if bar.ts.astimezone(ET).date() == day
        and (bar.ts.astimezone(ET).hour, bar.ts.astimezone(ET).minute) >= (9, 30)
        and (bar.ts.astimezone(ET).hour, bar.ts.astimezone(ET).minute) <= (16, 0)
    ]
    if not regular:
        return {"complete": False, "rth_bars": 0, "first": None, "last": None, "kind": "missing"}
    first = regular[0].ts.astimezone(ET)
    last = regular[-1].ts.astimezone(ET)
    first_ok = (first.hour, first.minute) <= (9, 31)
    full = len(regular) >= 370 and (last.hour, last.minute) >= (15, 59)
    early = (
        len(regular) >= 200
        and (12, 59) <= (last.hour, last.minute) <= (13, 5)
    )
    return {
        "complete": first_ok and (full or early),
        "rth_bars": len(regular),
        "first": first.isoformat(),
        "last": last.isoformat(),
        "kind": "full" if full else ("early_close" if early else "truncated"),
    }


def _build_cumulative(cohort_dir: Path, manifest: dict) -> dict:
    trades: list[dict] = []
    requested = covered = 0
    skipped: list[dict] = []
    chain_snapshots: dict[str, dict] = {}
    session_quality_by_day: dict[str, dict] = {}
    panel: dict[str, dict[str, list]] = {}
    for day in manifest["captured_days"]:
        record = json.loads((cohort_dir / "days" / f"{day}.json").read_text())
        report = record["report"]
        trades.extend(report.get("trades") or [])
        coverage = (report.get("metadata") or {}).get("data_coverage") or {}
        chain_model = (report.get("metadata") or {}).get("option_chain_model")
        if chain_model:
            chain_snapshots[day] = chain_model
        quality = (report.get("metadata") or {}).get("underlying_session_quality")
        if quality:
            session_quality_by_day[day] = quality
        requested += int(coverage.get("requested_weekdays") or 0)
        covered += int(coverage.get("days_with_underlying_bars") or 0)
        skipped.extend(coverage.get("skipped_missing_option_days") or [])
        for variant, variant_data in (report.get("option_panel") or {}).items():
            combined = panel.setdefault(variant, {"trades": [], "missing_contracts": []})
            combined["trades"].extend(variant_data.get("trades") or [])
            combined["missing_contracts"].extend(variant_data.get("missing_contracts") or [])

    definition = manifest["definition"]
    cumulative_panel = {}
    for variant, variant_data in panel.items():
        stats = dataclasses.asdict(_sample_stats(variant_data["trades"], "pnl"))
        cumulative_panel[variant] = {
            "trades": variant_data["trades"],
            "missing_contracts": variant_data["missing_contracts"],
            "complete": not variant_data["missing_contracts"],
            "stats": stats,
        }

    return {
        "symbol": manifest["symbol"],
        "start": manifest["captured_days"][0] if manifest["captured_days"] else None,
        "end": manifest["captured_days"][-1] if manifest["captured_days"] else None,
        "trades": trades,
        "option_panel": cumulative_panel,
        "metadata": {
            "research_mode": "historical_options",
            "experiment": manifest["name"],
            "config_fingerprint": definition["fingerprint"],
            "source_fingerprint": definition["source_fingerprint"],
            "capture_protocol": definition["capture_protocol"],
            "strategy_config": definition["strategy_config"],
            "execution_model": definition["execution_model"],
            "data_coverage": {
                "requested_weekdays": requested,
                "days_with_underlying_bars": covered,
                "skipped_missing_option_days": skipped,
                "skip_missing_option_data": False,
                "cached_only": False,
            },
            "validation_protocol": {
                "kind": "frozen_forward",
                "created_at": manifest["created_at"],
                "eligible_from": manifest.get("eligible_from"),
                "registration_mode": manifest.get("registration_mode", "legacy"),
                "captured_days": manifest["captured_days"],
                "rules": "one immutable definition per cohort; completed sessions only",
            },
            "option_chain_snapshots": chain_snapshots,
            "underlying_session_quality": session_quality_by_day,
        },
    }


def initialize_forward_cohort(
    *,
    name: str,
    symbol: str,
    settings: Settings,
    eligible_from: date | None = None,
) -> Path:
    """Lock a prospective cohort before its first observable session.

    Initialization is deliberately local and idempotent: it records the exact
    executable/configuration fingerprint without connecting to TWS or opening
    any historical day. Later captures must match this manifest.
    """
    symbol = symbol.upper()
    safe_name = _safe_experiment(name)
    manifest_path = settings.log_dir / "forward" / safe_name / "manifest.json"
    definition = _definition(symbol, settings)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("symbol") != symbol:
            raise ValueError(
                f"cohort {safe_name!r} is locked to {manifest.get('symbol')}, not {symbol}"
            )
        previous = (manifest.get("definition") or {}).get("fingerprint")
        if previous != definition["fingerprint"]:
            raise ValueError(
                f"cohort {safe_name!r} is frozen at {previous}; current definition is "
                f"{definition['fingerprint']}. Use a new cohort name instead of mixing rules."
            )
        return manifest_path

    created_at = datetime.now(timezone.utc)
    registration_mode = "pre_session" if eligible_from is None else "at_first_capture"
    eligible_from = eligible_from or (created_at.astimezone(ET).date() + timedelta(days=1))
    _write_json(
        manifest_path,
        {
            "name": safe_name,
            "symbol": symbol,
            "created_at": created_at.isoformat(),
            "eligible_from": eligible_from.isoformat(),
            "registration_mode": registration_mode,
            "definition": definition,
            "captured_days": [],
        },
    )
    return manifest_path


def capture_forward_day(
    *,
    name: str,
    symbol: str,
    day: date,
    settings: Settings,
    ib,
    cache_dir: Path,
) -> tuple[Path, Path]:
    """Capture one completed session and update its immutable validation cohort."""
    if day >= datetime.now(ET).date():
        raise ValueError("forward capture requires a completed session strictly before today")
    symbol = symbol.upper()
    safe_name = _safe_experiment(name)
    cohort_dir = settings.log_dir / "forward" / safe_name
    manifest_path = cohort_dir / "manifest.json"
    cumulative_path = cohort_dir / "cumulative.json"
    initialize_forward_cohort(
        name=safe_name, symbol=symbol, settings=settings, eligible_from=day,
    )
    manifest = json.loads(manifest_path.read_text())
    eligible_from = manifest.get("eligible_from")
    if eligible_from and day < date.fromisoformat(eligible_from):
        raise ValueError(
            f"cohort {safe_name!r} accepts sessions from {eligible_from}; "
            f"refusing retroactive capture of {day}"
        )

    fresh_bars = ensure_underlying_bars(ib, symbol, day, cache_dir, refresh=True)
    quality = session_quality(fresh_bars, day)
    if not quality["complete"]:
        raise ValueError(
            f"TWS underlying session for {symbol} {day} is not complete: "
            f"{quality['rth_bars']} RTH bars, last={quality['last']}, kind={quality['kind']}"
        )
    chain_snapshot = ib.get_option_chain(symbol)
    report = run_backtest(
        symbol,
        day,
        day,
        settings,
        use_llm=False,
        ib=ib,
        cache_dir=cache_dir,
        skip_missing_option_data=False,
        underlying_only=False,
        cached_only=False,
        experiment=f"forward-{safe_name}-{day.isoformat()}",
        chain_override=chain_snapshot,
        option_panel=True,
    )
    coverage = report.metadata.get("data_coverage") or {}
    if int(coverage.get("days_with_underlying_bars") or 0) != 1:
        raise ValueError(f"TWS returned no underlying session for {symbol} {day}")
    report.metadata["underlying_session_quality"] = quality

    record_path = cohort_dir / "days" / f"{day.isoformat()}.json"
    _write_json(
        record_path,
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "report": _jsonable(dataclasses.asdict(report)),
        },
    )
    manifest["captured_days"] = sorted(set(manifest["captured_days"] + [day.isoformat()]))
    _write_json(manifest_path, manifest)
    _write_json(cumulative_path, _build_cumulative(cohort_dir, manifest))
    return record_path, cumulative_path


def latest_completed_session(
    *,
    symbol: str,
    ib,
    cache_dir: Path,
    today: date | None = None,
    lookback_calendar_days: int = 7,
) -> date:
    """Find the newest prior session with real TWS stock bars.

    Capturing the newest actual session keeps its nearest non-0DTE contracts
    alive in the current chain, including across weekends and market holidays.
    """
    if lookback_calendar_days <= 0:
        raise ValueError("lookback_calendar_days must be positive")
    today = today or datetime.now(ET).date()
    for offset in range(1, lookback_calendar_days + 1):
        candidate = today - timedelta(days=offset)
        if candidate.weekday() >= 5:
            continue
        bars = ensure_underlying_bars(
            ib, symbol.upper(), candidate, cache_dir, refresh=True
        )
        if bars:
            quality = session_quality(bars, candidate)
            if not quality["complete"]:
                raise ValueError(
                    f"latest TWS session candidate {symbol.upper()} {candidate} is incomplete: "
                    f"{quality['rth_bars']} RTH bars, last={quality['last']}"
                )
            return candidate
    raise ValueError(
        f"no completed {symbol.upper()} session found in the prior "
        f"{lookback_calendar_days} calendar days"
    )
