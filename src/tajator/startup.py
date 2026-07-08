"""Pre-flight safety checks for `tajator run`.

The session only manages positions it can prove it owns: a persisted position
that exactly matches what the broker account holds. Anything else at IB — a
kill-switch file, resting orders, unexplained positions — needs the operator,
so startup refuses instead of guessing (see the 2026-07-08 incident: untracked
contracts accumulated exactly because failures were papered over).
"""

from __future__ import annotations

import sys
from datetime import date

from .broker.base import BrokerOptionPosition
from .config import Settings
from .journal import Journal
from .models import SelectedContract
from .notify import Notifier
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
    adopt, warnings, refusals = reconcile_positions(
        persisted, broker.option_positions(settings.symbols), settings.symbols, today
    )
    if refusals:
        sys.exit(
            "the IB account holds option positions tajator cannot explain:\n  "
            + "\n  ".join(refusals)
            + "\ntajator cannot adopt a position without its persisted plan and stop.\n"
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
