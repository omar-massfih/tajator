"""Pre-flight safety checks for `tajator run`.

The session only manages positions it can prove it owns: a persisted position
that exactly matches what the broker account holds. Anything else at IB — a
kill-switch file, resting orders, unexplained positions — needs the operator,
so startup refuses instead of guessing (see the 2026-07-08 incident: untracked
contracts accumulated exactly because failures were papered over) — unless the
journal's fill records prove ownership exactly, which covers a crash between
an order fill and the post-tick state.json write (see recovery.py).
"""

from __future__ import annotations

import sys
from datetime import date, datetime

from .broker.base import BrokerOptionPosition
from .config import Settings
from .journal import ET, Journal
from .models import SelectedContract
from .notify import Notifier
from .recovery import RecoveredSession, recover_sessions
from .state_store import PersistedState, PersistedSession, StateStore


def check_kill_switch(settings: Settings) -> None:
    """Refuse to launch while the kill switch is set (call before connecting).
    Per-tick guardrails also honor it, but a file written because the account
    needs reconciling must stop a fresh launch outright, not just entries."""
    f = settings.kill_switch_file
    if f.exists():
        sys.exit(
            f"kill switch file present at {f}:\n  {f.read_text().strip()}\n"
            "Reconcile the account in IB Gateway, then delete the file and restart."
        )


def run_startup_checks(
    settings: Settings,
    broker,
    store: StateStore,
    journal: Journal,
    notifier: Notifier,
) -> dict[str, PersistedSession]:
    """Order/position preflight. Returns the per-symbol state to seed the
    sessions with (adopted positions and today's trade counts); exits with an
    operator message when the account holds anything tajator cannot explain."""
    orders = broker.open_option_orders(settings.symbols)
    if orders:
        sys.exit(
            "the IB account has resting orders in configured symbols:\n  "
            + "\n  ".join(orders)
            + "\ntajator will not trade around orders it did not place "
            "(cancelling them here could race a fill).\n"
            "Cancel them in IB Gateway/TWS, then restart."
        )

    try:
        persisted = store.load()
    except Exception as exc:  # noqa: BLE001 — a corrupt file must stop the launch
        sys.exit(
            f"state file {store.path} is unreadable ({exc}).\n"
            "Restore or delete it — without it any open position is unexplained "
            "and must be flattened manually — then restart."
        )

    today = broker.now().date()
    broker_positions = broker.option_positions(settings.symbols)
    adopt, warnings, refusals = reconcile_positions(
        persisted, broker_positions, settings.symbols, today
    )
    recovered_info: dict[str, RecoveredSession] = {}
    if refusals:
        # Second chance: a crash between an order fill and the post-tick
        # state.json write leaves the broker ahead of persisted state. The
        # journal's last fill record carries the full position, so replay it
        # and adopt only what it explains exactly.
        adopt, warnings, refusals, recovered_info = attempt_journal_recovery(
            settings, persisted, broker_positions, today, adopt, warnings, refusals
        )
    if refusals:
        sys.exit(
            "the IB account holds option positions tajator cannot explain:\n  "
            + "\n  ".join(refusals)
            + "\ntajator cannot adopt a position without its persisted plan and stop "
            "(journal replay could not explain these either).\n"
            "Flatten it manually in IB Gateway, or remove the symbol from SYMBOLS, then restart."
        )

    for warning in warnings:
        journal.write("startup_warning", warning=warning)
        notifier.notify_status(f"tajator startup: {warning}")
        print(f"!!! {warning}")
    for line in broker.other_positions_summary(settings.symbols):
        msg = f"unrelated position in the account (tajator will not touch it): {line}"
        journal.write("startup_warning", warning=msg)
        notifier.notify_status(f"tajator startup: {msg}")
        print(f"!   {msg}")

    for symbol, sess in adopt.items():
        if sess.position is not None:
            p = sess.position
            rec = recovered_info.get(symbol)
            if rec is not None:
                journal.write(
                    "position_recovered", symbol=symbol, position=p,
                    source_fill_ts=rec.last_fill_ts, trades_today=sess.trades_today,
                )
                notifier.notify_status(
                    f"{symbol}: RECOVERED {p.qty_remaining}x {p.contract.local_name} from journal "
                    "replay (state.json was stale — crash between fill and persist)"
                )
                print(
                    f"[{symbol}] recovered position {p.qty_remaining}x {p.contract.local_name} "
                    f"from the journal (stop {p.plan.stop_price})"
                )
            else:
                journal.write("position_adopted", symbol=symbol, position=p)
                notifier.notify_status(
                    f"{symbol}: adopted {p.qty_remaining}x {p.contract.local_name} from the previous run"
                )
                print(
                    f"[{symbol}] adopted position {p.qty_remaining}x {p.contract.local_name} "
                    f"(stop {p.plan.stop_price})"
                )
        # seed the file now so it reflects reality even if the first tick fails
        store.update(symbol, sess.position, sess.trades_today, today)
    return adopt


def attempt_journal_recovery(
    settings: Settings,
    persisted: PersistedState | None,
    broker_positions: list[BrokerOptionPosition],
    today: date,
    first_adopt: dict[str, PersistedSession],
    first_warnings: list[str],
    first_refusals: list[str],
) -> tuple[dict[str, PersistedSession], list[str], list[str], dict[str, RecoveredSession]]:
    """Second-pass reconcile with journal-derived state substituted for the
    symbols the first pass refused. All-or-nothing: if the amended state still
    leaves any refusal, the first pass's results are returned verbatim so the
    operator sees the original diagnostics. Never adopts a partial match."""
    recovered, reader_warnings = recover_sessions(settings.log_dir, settings.symbols, today)

    by_symbol: dict[str, list[BrokerOptionPosition]] = {}
    for bp in broker_positions:
        by_symbol.setdefault(bp.symbol, []).append(bp)

    # Normalize trades_today up front (the amended state is stamped with
    # today's date, so stale counts must not resurrect through the re-run).
    same_day = persisted is not None and persisted.trading_day == today
    amended_sessions = {
        sym: PersistedSession(
            position=s.position, trades_today=s.trades_today if same_day else 0
        )
        for sym, s in (persisted.sessions.items() if persisted else ())
    }

    substituted: dict[str, RecoveredSession] = {}
    for symbol in settings.symbols:
        _, _, sym_refusals = reconcile_positions(
            persisted, by_symbol.get(symbol, []), [symbol], today
        )
        if not sym_refusals:
            continue  # only refused symbols may be rewritten from the journal
        rec = recovered.get(symbol)
        if rec is None or rec.position is None:
            continue
        matches = [
            bp
            for bp in by_symbol.get(symbol, [])
            if _same_contract_strict(rec.position.contract, bp)
        ]
        if len(matches) != 1 or matches[0].qty != rec.position.qty_remaining:
            continue
        prev = amended_sessions.get(symbol)
        amended_sessions[symbol] = PersistedSession(
            position=rec.position,
            # max is the conservative direction for the daily trade cap
            trades_today=max(prev.trades_today if prev else 0, rec.trades_today),
        )
        substituted[symbol] = rec

    if not substituted:
        return first_adopt, first_warnings, first_refusals, {}

    amended = PersistedState(
        updated_at=persisted.updated_at if persisted else datetime.now(ET),
        trading_day=today,
        sessions=amended_sessions,
    )
    adopt, warnings, refusals = reconcile_positions(
        amended, broker_positions, settings.symbols, today
    )
    if refusals:
        return first_adopt, first_warnings, first_refusals, {}
    return adopt, warnings + reader_warnings, refusals, substituted


def reconcile_positions(
    persisted: PersistedState | None,
    broker_positions: list[BrokerOptionPosition],
    symbols: list[str],
    today: date,
) -> tuple[dict[str, PersistedSession], list[str], list[str]]:
    """Match persisted per-symbol state against the broker's option positions.

    Returns (adopt, warnings, refusals): every broker position must be
    explained by persisted state or it is a refusal; a persisted position the
    broker no longer holds was closed externally — a warning, and the session
    starts flat. trades_today only survives a same-day restart."""
    adopt: dict[str, PersistedSession] = {}
    warnings: list[str] = []
    refusals: list[str] = []
    by_symbol: dict[str, list[BrokerOptionPosition]] = {}
    for bp in broker_positions:
        by_symbol.setdefault(bp.symbol, []).append(bp)

    for symbol in symbols:
        sess = persisted.sessions.get(symbol) if persisted else None
        pos = sess.position if sess else None
        trades = (
            sess.trades_today
            if sess and persisted is not None and persisted.trading_day == today
            else 0
        )
        at_broker = by_symbol.pop(symbol, [])

        if pos is None:
            refusals.extend(
                f"{bp.qty:+d}x {bp.local_symbol} — not opened by tajator (no persisted state)"
                for bp in at_broker
            )
            adopt[symbol] = PersistedSession(trades_today=trades)
            continue

        matches = [bp for bp in at_broker if _same_contract(pos.contract, bp)]
        refusals.extend(
            f"{bp.qty:+d}x {bp.local_symbol} — not the persisted {symbol} position"
            for bp in at_broker
            if bp not in matches
        )
        if not matches:
            warnings.append(
                f"{symbol}: persisted position {pos.qty_remaining}x {pos.contract.local_name} "
                "is gone at the broker — closed externally; starting flat"
            )
            adopt[symbol] = PersistedSession(trades_today=trades)
            continue
        bp = matches[0]
        if bp.qty != pos.qty_remaining:
            refusals.append(
                f"{symbol}: broker holds {bp.qty:+d}x {bp.local_symbol} but the persisted plan "
                f"covers {pos.qty_remaining} — changed externally, the plan no longer matches"
            )
            continue
        adopt[symbol] = PersistedSession(position=pos, trades_today=trades)

    # option_positions() is already filtered to configured symbols, so
    # leftovers can only appear if the filter and this loop disagree — refuse.
    for leftovers in by_symbol.values():
        refusals.extend(
            f"{bp.qty:+d}x {bp.local_symbol} — unexpected symbol" for bp in leftovers
        )
    return adopt, warnings, refusals


def _same_contract(contract: SelectedContract, bp: BrokerOptionPosition) -> bool:
    if contract.con_id and bp.con_id and contract.con_id != bp.con_id:
        return False
    return (
        contract.symbol == bp.symbol
        and contract.expiry == bp.expiry
        and contract.strike == bp.strike
        and contract.right == bp.right
    )


def _same_contract_strict(contract: SelectedContract, bp: BrokerOptionPosition) -> bool:
    """Unlike _same_contract, a missing con_id is a mismatch, not a wildcard:
    journal-based recovery must never adopt from a stub/replay fill (those
    carry con_id null) or a hand-edited record."""
    return (
        bool(contract.con_id)
        and contract.con_id == bp.con_id
        and contract.symbol == bp.symbol
        and contract.expiry == bp.expiry
        and contract.strike == bp.strike
        and contract.right == bp.right
    )
