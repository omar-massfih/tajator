"""CLI: tajator {run | check-ib | replay}."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import load_settings
from .journal import Journal
from .notify import NullNotifier, TelegramNotifier

ET = ZoneInfo("America/New_York")


def _runtime_policy_metadata(settings, deterministic: bool) -> dict:
    """Journal enough identity to keep incompatible execution samples separate."""
    from .backtest.forward import _definition, _source_fingerprint

    return {
        "policy_mode": "deterministic" if deterministic else "llm",
        "validation_compatible": deterministic,
        "source_fingerprint": _source_fingerprint(),
        "cohort_fingerprints": {
            symbol: _definition(symbol, settings)["fingerprint"]
            for symbol in settings.symbols
        } if deterministic else {},
        "symbols": settings.symbols,
        "llm_model": None if deterministic else settings.llm_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="tajator")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="live minute loop against IBKR (paper by default)")
    run_policy = run.add_mutually_exclusive_group()
    run_policy.add_argument(
        "--deterministic", dest="deterministic", action="store_true", default=True,
        help="use the frozen rule-follower (default; retained as an explicit compatibility flag)",
    )
    run_policy.add_argument(
        "--llm", dest="deterministic", action="store_false",
        help="opt into experimental LLM entry and management decisions",
    )
    shadow = sub.add_parser(
        "shadow",
        help="run deterministic live-quote simulation against TWS — never places orders",
    )
    shadow.add_argument("--symbol", default=None, help="defaults to the first configured symbol")
    shadow.add_argument(
        "--client-id", type=int, default=116,
        help="dedicated TWS market-data client ID (default: 116)",
    )
    check_ib = sub.add_parser(
        "check-ib", help="connectivity check: bars, chain, quote — no orders"
    )
    check_ib.add_argument(
        "--client-id", type=int, default=118,
        help="dedicated read-only diagnostic client ID (default: 118)",
    )
    sub.add_parser("prep", help="run pre-market prep now: levels + LLM briefing — no orders")

    test_order = sub.add_parser(
        "test-order",
        help="supervised paper diagnostic: buy 1 lot, watch the fill timeline, sell it back",
    )
    test_order.add_argument("--symbol", default=None, help="defaults to the first configured SYMBOLS entry")
    test_order.add_argument("--qty", type=int, default=1)
    test_order.add_argument(
        "--wait", type=int, default=None,
        help="override ORDER_TIMEOUT_S for this diagnostic",
    )
    test_order.add_argument(
        "--with-stop", action="store_true",
        help="also place, verify, and cancel a protective stop while the position is open",
    )

    replay = sub.add_parser("replay", help="step the graph through a recorded day (no IB orders)")
    src = replay.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="CSV of 1-min bars (ts,open,high,low,close,volume)")
    src.add_argument("--date", help="YYYY-MM-DD — fetch that day's bars from IB once")
    replay.add_argument("--symbol", default=None, help="defaults to the first configured SYMBOLS entry")
    replay.add_argument("--no-llm", action="store_true", help="deterministic rule-follower instead of the LLM")
    replay.add_argument("--prev-high", type=float, default=None)
    replay.add_argument("--prev-low", type=float, default=None)

    backtest = sub.add_parser(
        "backtest", help="step the graph over a date range from IB with real historical option fills"
    )
    backtest.add_argument("--symbol", default=None, help="defaults to the first configured SYMBOLS entry")
    backtest.add_argument("--start", required=True, help="YYYY-MM-DD")
    backtest.add_argument("--end", required=True, help="YYYY-MM-DD")
    backtest.add_argument("--no-llm", action="store_true", help="deterministic rule-follower instead of the LLM")
    backtest.add_argument("--cache-dir", type=Path, default=None, help="defaults to Settings.backtest_cache_dir")
    backtest.add_argument(
        "--skip-missing-option-data", action="store_true",
        help="exclude an entire day when any required historical option fill is unavailable",
    )
    backtest.add_argument(
        "--underlying-only", action="store_true",
        help="research signal outcomes using stock-price moves; no historical option data required",
    )
    backtest.add_argument(
        "--cached-only", action="store_true",
        help="never fetch missing historical bars; replay only files already in the cache",
    )
    backtest.add_argument("--experiment", default="baseline", help="report/journal experiment label")

    compare = sub.add_parser("backtest-compare", help="compare experiment-safe backtest JSON reports")
    compare.add_argument("reports", nargs="+", type=Path)

    strategy_compare = sub.add_parser(
        "strategy-compare",
        help="paired day-clustered comparison of baseline and candidate replays",
    )
    strategy_compare.add_argument("baseline", type=Path)
    strategy_compare.add_argument("candidate", type=Path)
    strategy_compare.add_argument("--min-trades", type=int, default=50)
    strategy_compare.add_argument("--min-positive-month-ratio", type=float, default=0.6)
    strategy_compare.add_argument(
        "--only-change", action="append", default=[],
        help="require this strategy setting to be the only change (repeatable)",
    )
    strategy_compare.add_argument("--output", type=Path, default=None)

    audit = sub.add_parser(
        "edge-audit",
        help="audit whether a backtest is stable and out-of-sample enough to support an edge",
    )
    audit.add_argument("report", type=Path)
    audit_role = audit.add_mutually_exclusive_group()
    audit_role.add_argument(
        "--validation-start",
        help="YYYY-MM-DD split; only trades on/after this date are judged as holdout",
    )
    audit_role.add_argument(
        "--validation-only", action="store_true",
        help="declare the entire report a frozen out-of-sample validation run",
    )
    audit.add_argument("--min-trades", type=int, default=50)

    forward_init = sub.add_parser(
        "forward-init",
        help="lock a prospective cohort definition before its first captured session",
    )
    forward_init.add_argument("--name", required=True, help="immutable cohort name")
    forward_init.add_argument("--symbol", default=None, help="defaults to the first configured symbol")

    forward = sub.add_parser(
        "forward-validate",
        help="capture one completed TWS session into a frozen options-validation cohort",
    )
    forward.add_argument("--name", required=True, help="immutable cohort name")
    forward.add_argument("--symbol", default=None, help="defaults to the first configured symbol")
    forward.add_argument("--date", required=True, help="completed session date, YYYY-MM-DD")
    forward.add_argument("--cache-dir", type=Path, default=None)
    forward.add_argument(
        "--client-id", type=int, default=117,
        help="dedicated read-only TWS API client ID (default: 117)",
    )

    latest = sub.add_parser(
        "forward-latest",
        help="discover and capture the latest completed TWS session into a frozen cohort",
    )
    latest.add_argument("--name", required=True, help="immutable cohort name")
    latest.add_argument("--symbol", default=None, help="defaults to the first configured symbol")
    latest.add_argument("--cache-dir", type=Path, default=None)
    latest.add_argument("--client-id", type=int, default=117)
    latest.add_argument("--lookback-days", type=int, default=7)

    panel = sub.add_parser(
        "option-panel-compare",
        help="compare captured ITM/ATM/OTM and expiry variants at identical signal times",
    )
    panel.add_argument("report", type=Path)
    panel.add_argument("--min-pairs", type=int, default=50)

    calibration = sub.add_parser(
        "execution-calibrate",
        help="compare journaled paper fills with historical option-bar execution assumptions",
    )
    calibration.add_argument("journal", type=Path)
    calibration.add_argument("--symbol", required=True)
    calibration.add_argument("--cache-dir", type=Path, default=None)
    calibration.add_argument("--client-id", type=int, default=117)
    calibration.add_argument("--output", type=Path, default=None)

    shadow_report = sub.add_parser(
        "shadow-report",
        help="build an edge-auditable report from no-order shadow journals",
    )
    shadow_report.add_argument(
        "path", type=Path, help="shadow journal JSONL file or directory"
    )
    shadow_report.add_argument("--symbol", required=True)
    shadow_report.add_argument("--output", type=Path, default=None)

    tournament = sub.add_parser(
        "historical-signal-tournament",
        help="select and validate preregistered intraday signals on cached TWS stock bars",
    )
    tournament.add_argument("--cache-dir", type=Path, default=None)
    tournament.add_argument("--output", type=Path, default=None)

    followup = sub.add_parser(
        "historical-signal-followup",
        help="validate the preregistered opening-drive fade historical follow-up",
    )
    followup.add_argument("--cache-dir", type=Path, default=None)
    followup.add_argument("--output", type=Path, default=None)

    daily_fetch = sub.add_parser(
        "historical-daily-fetch",
        help="fetch long-run TWS daily stock bars for the preregistered swing study",
    )
    daily_fetch.add_argument(
        "--symbols", default="AAPL,META,MSFT,SPY,AMZN,GOOGL,NVDA,QQQ"
    )
    daily_fetch.add_argument("--start", default="2017-01-01")
    daily_fetch.add_argument("--end", default="2026-06-30")
    daily_fetch.add_argument("--cache-dir", type=Path, default=None)
    daily_fetch.add_argument("--client-id", type=int, default=117)

    daily_tournament = sub.add_parser(
        "historical-daily-tournament",
        help="run the preregistered sequential swing-signal tournament",
    )
    daily_tournament.add_argument("--cache-dir", type=Path, default=None)
    daily_tournament.add_argument("--output", type=Path, default=None)

    focused = sub.add_parser(
        "historical-aapl-focus",
        help="run the preregistered AAPL-first temporal holdout on fresh TWS history",
    )
    focused.add_argument("--cache-dir", type=Path, default=Path("data/tws-focused"))
    focused.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "run":
        cmd_run(args)
    elif args.command == "shadow":
        cmd_shadow(args)
    elif args.command == "check-ib":
        cmd_check_ib(args)
    elif args.command == "test-order":
        cmd_test_order(args)
    elif args.command == "prep":
        cmd_prep()
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "backtest-compare":
        from .backtest.compare import print_comparison
        print_comparison(args.reports)
    elif args.command == "strategy-compare":
        cmd_strategy_compare(args)
    elif args.command == "edge-audit":
        cmd_edge_audit(args)
    elif args.command == "forward-init":
        cmd_forward_init(args)
    elif args.command == "forward-validate":
        cmd_forward_validate(args)
    elif args.command == "forward-latest":
        cmd_forward_latest(args)
    elif args.command == "option-panel-compare":
        cmd_option_panel_compare(args)
    elif args.command == "execution-calibrate":
        cmd_execution_calibrate(args)
    elif args.command == "shadow-report":
        cmd_shadow_report(args)
    elif args.command == "historical-signal-tournament":
        cmd_historical_signal_tournament(args)
    elif args.command == "historical-signal-followup":
        cmd_historical_signal_followup(args)
    elif args.command == "historical-daily-fetch":
        cmd_historical_daily_fetch(args)
    elif args.command == "historical-daily-tournament":
        cmd_historical_daily_tournament(args)
    elif args.command == "historical-aapl-focus":
        cmd_historical_aapl_focus(args)
    else:
        cmd_replay(args)


def _notifier(settings):
    return (
        TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
        if settings.telegram_bot_token and settings.telegram_chat_id
        else NullNotifier()
    )


def _ib_broker(settings=None, notifier=None):
    from .broker.ib import IBBroker

    settings = settings or load_settings()
    broker = IBBroker(settings, notifier=notifier)
    try:
        broker.connect()
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"could not connect to IB at {settings.ib_host}:{settings.ib_port} ({exc}).\n"
            "Is IB Gateway running with the API enabled?"
        )
    return settings, broker


def cmd_run(args) -> None:
    from .graph.nodes import RuntimeContext
    from .llm.decide import build_llm
    from .models import MorningBriefing
    from .runner import LiveRunner, TradingSession
    from .startup import check_execution_diagnostics, check_kill_switch, run_startup_checks
    from .state_store import StateStore

    settings = load_settings()
    notifier = _notifier(settings)
    check_kill_switch(settings, notifier)
    check_execution_diagnostics(settings)
    settings, broker = _ib_broker(settings, notifier)
    journal = Journal(settings.log_dir)
    broker.journal = journal  # order timelines land next to the trade records
    journal.write(
        "policy_start", ts=broker.now(),
        **_runtime_policy_metadata(settings, args.deterministic),
    )
    store = StateStore(settings.state_file)
    try:
        # Refuse on resting orders or positions that persisted state cannot
        # explain; adopt positions from a previous run that match exactly.
        adopted = run_startup_checks(settings, broker, store, journal, notifier)
    except SystemExit:
        broker.disconnect()
        raise
    llm = prep_llm = None
    if not args.deterministic:
        try:
            # fail fast on a missing/invalid API key instead of waiting all day
            llm = build_llm(settings.llm_model)
            prep_llm = build_llm(settings.llm_model, output_model=MorningBriefing)
        except Exception as exc:  # noqa: BLE001
            broker.disconnect()
            sys.exit(f"could not initialize LLM '{settings.llm_model}': {exc}")
    today = broker.now().date()
    sessions = [
        TradingSession(
            RuntimeContext(
                settings=settings, broker=broker, journal=journal, symbol=symbol,
                use_llm=not args.deterministic,
                notifier=notifier, _llm=llm, _prep_llm=prep_llm,
            ),
            store=store,
            restored=adopted.get(symbol),
            day=today,
        )
        for symbol in settings.symbols
    ]
    try:
        LiveRunner(sessions).run()
    finally:
        broker.disconnect()


def cmd_shadow(args) -> None:
    """Run the production graph on live TWS data with a broker that cannot order."""
    from .broker.shadow import ShadowBroker
    from .graph.nodes import RuntimeContext
    from .runner import LiveRunner, TradingSession
    from .state_store import PersistedSession, StateStore

    base = load_settings()
    shadow_dir = base.log_dir / "shadow"
    settings = base.model_copy(
        update={
            "ib_client_id": args.client_id,
            "protective_stop_enabled": False,
            "log_dir": shadow_dir,
            "state_file": shadow_dir / "state.json",
            "kill_switch_file": shadow_dir / "KILL",
        }
    )
    _, market = _ib_broker(settings, NullNotifier())
    journal = Journal(shadow_dir)
    broker = ShadowBroker(market, settings, journal)
    store = StateStore(settings.state_file)
    symbol = (args.symbol or settings.symbols[0]).upper()
    today = broker.now().date()
    restored = None
    try:
        persisted = store.load()
        if persisted is not None and persisted.trading_day == today:
            restored = persisted.sessions.get(symbol)
        elif persisted is not None:
            # An overnight shadow position is deliberately not adopted: this
            # strategy is intraday and a stale simulation must not contaminate
            # a new session's evidence.
            restored = PersistedSession()
        journal.write(
            "shadow_started", symbol=symbol, deterministic=True,
            client_id=args.client_id, no_order_placed=True,
        )
        print(
            "=== TAJATOR SHADOW | LIVE TWS DATA | DETERMINISTIC | NO ORDERS ===\n"
            f"symbol: {symbol}  journal/state: {shadow_dir}"
        )
        session = TradingSession(
            RuntimeContext(
                settings=settings,
                broker=broker,
                journal=journal,
                symbol=symbol,
                use_llm=False,
                notifier=NullNotifier(),
            ),
            store=store,
            restored=restored,
            day=today,
        )
        LiveRunner([session]).run()
    finally:
        market.disconnect()


def cmd_check_ib(args) -> None:
    from .trade.contracts import select_contract

    settings = load_settings().model_copy(update={"ib_client_id": args.client_id})
    settings, broker = _ib_broker(settings)
    try:
        print(f"connected: {broker.is_connected()}  (market data type {settings.market_data_type})")
        accounts = broker.ib.managedAccounts()
        print(f"accounts: {accounts}")

        for symbol in settings.symbols:
            print(f"\n=== {symbol} ===")
            bars = broker.get_bars(symbol, lookback_minutes=30)
            print(f"last {min(20, len(bars))} of {len(bars)} 1-min {symbol} bars:")
            for b in bars[-20:]:
                print(f"  {b.ts:%Y-%m-%d %H:%M}  O{b.open:.2f} H{b.high:.2f} L{b.low:.2f} C{b.close:.2f}")

            prev_high, prev_low = broker.get_prev_day_range(symbol)
            print(f"prev day range: high {prev_high} / low {prev_low}")

            chain = broker.get_option_chain(symbol)
            print(f"chain: {len(chain.strikes)} strikes, nearest expirations {chain.expirations[:4]}")

            if bars:
                contract = select_contract(chain, symbol, bars[-1].close, "call", broker.now())
                if contract:
                    started = time.monotonic()
                    quote, underlying = broker.get_entry_market_snapshot(contract)
                    elapsed = time.monotonic() - started
                    age = max(0.0, (broker.now() - quote.ts).total_seconds())
                    print(
                        f"atomic entry snapshot ({elapsed:.2f}s): underlying {underlying}; "
                        f"{contract.local_name} bid {quote.bid} / ask {quote.ask} / "
                        f"last {quote.last}; quote age {age:.2f}s"
                    )
        print("\ncheck-ib complete. No orders were placed.")
    finally:
        broker.disconnect()


def cmd_test_order(args) -> None:
    """Supervised paper round trip through the production market-order path."""
    from .trade.contracts import select_contract
    from .trade.execution import size_entry, validate_option_liquidity

    settings = load_settings()
    if settings.trading_mode != "paper":
        sys.exit("test-order is a paper diagnostic — refusing to run in live mode.")
    if args.wait is not None:
        settings = settings.model_copy(update={"order_timeout_s": args.wait})
    settings, broker = _ib_broker(settings)
    broker.journal = Journal(settings.log_dir)
    symbol = (args.symbol or settings.symbols[0]).upper()
    passed = False
    failure = "diagnostic did not complete"
    buy_fill = sell_fill = None
    contract = None
    diagnostic_failures: list[str] = []
    try:
        print(f"accounts: {broker.ib.managedAccounts()}")
        print(f"market data type requested: {settings.market_data_type}, "
              f"delayed fallback active: {broker.is_delayed_data}")
        bars = broker.get_bars(symbol, lookback_minutes=10)
        if not bars:
            sys.exit(f"no bars for {symbol} — is the market open?")
        spot = bars[-1].close
        chain = broker.get_option_chain(symbol)
        contract = select_contract(chain, symbol, spot, "call", broker.now())
        if contract is None:
            sys.exit(f"no usable {symbol} contract in the chain")
        quote, underlying = broker.get_entry_market_snapshot(contract)
        print(
            f"{contract.local_name}: bid {quote.bid} / ask {quote.ask} / last {quote.last} "
            f"({'DELAYED' if quote.delayed else 'live'} quotes)"
        )
        quote_failure = validate_option_liquidity(quote, settings)
        if quote_failure:
            raise RuntimeError(quote_failure)
        affordable = size_entry(
            quote.ask, settings, reserve_pct=settings.entry_budget_reserve_pct
        )
        if args.qty <= 0 or args.qty > affordable:
            raise RuntimeError(
                f"requested {args.qty} contract(s), but ask-plus-reserve budget allows {affordable}"
            )

        print(f"\nplacing BUY {args.qty}x {contract.local_name} through production market path ...")
        buy_fill = broker.buy_option_from_snapshot(
            contract, args.qty, quote, underlying,
        )
        print(
            f"filled {buy_fill.qty}x @ {buy_fill.premium:.2f} in "
            f"{buy_fill.execution_quality.latency_s:.1f}s"
        )
        diagnostic_failures.extend(buy_fill.execution_quality.breaches)
        if buy_fill.qty != args.qty:
            diagnostic_failures.append(f"entry filled only {buy_fill.qty}/{args.qty}")
        remaining = buy_fill.qty

        if args.with_stop:
            stop_price = round(spot - 1.00, 2)
            print(f"\nplacing protective stop (SELL if {symbol} <= {stop_price}, GTC) ...")
            stop = broker.place_protective_stop(
                contract, buy_fill.qty, stop_price, "call",
                f"{settings.order_ref_prefix}-stop:{symbol}",
            )
            print(f"placed order {stop.order_id} (ref {stop.order_ref}) — check it shows in TWS")
            status = broker.poll_protective_stop(contract, stop)
            print(f"poll: {status.state} (working {status.working_qty})")
            result = broker.cancel_protective_stop(contract, stop, expected_held=buy_fill.qty)
            print(f"cancel confirmed: cancelled={result.cancelled}, filled={result.filled_qty}")
            remaining -= result.filled_qty

        if remaining:
            print(f"\nselling {remaining}x back through production market path ...")
            sell_fill = broker.sell_option(contract, remaining)
            print(
                f"filled {sell_fill.qty}x @ {sell_fill.premium:.2f} in "
                f"{sell_fill.execution_quality.latency_s:.1f}s"
            )
            diagnostic_failures.extend(sell_fill.execution_quality.breaches)
            if sell_fill.qty != remaining:
                diagnostic_failures.append(f"exit filled only {sell_fill.qty}/{remaining}")
        passed = not diagnostic_failures
        failure = "; ".join(diagnostic_failures) if diagnostic_failures else ""
        print(
            f"\nround trip complete: PnL "
            f"${100 * (sell_fill.qty if sell_fill else 0) * ((sell_fill.premium if sell_fill else 0) - buy_fill.premium):+.0f}"
        )
    except Exception as exc:  # noqa: BLE001 — diagnostic must persist its failure reason
        failure = str(exc)
        print(f"\nexecution diagnostic FAILED: {failure}")
    finally:
        broker.journal.write(
            "execution_diagnostic",
            symbol=symbol,
            passed=passed,
            failure=failure,
            contract=contract,
            buy=buy_fill.execution_quality if buy_fill is not None else None,
            sell=sell_fill.execution_quality if sell_fill is not None else None,
        )
        broker.disconnect()
    if not passed:
        sys.exit(1)


def cmd_prep() -> None:
    from .graph.nodes import RuntimeContext
    from .llm.decide import build_llm
    from .models import MorningBriefing
    from .runner import TradingSession

    settings, broker = _ib_broker()
    try:
        llm = build_llm(settings.llm_model)
        prep_llm = build_llm(settings.llm_model, output_model=MorningBriefing)
    except Exception as exc:  # noqa: BLE001
        broker.disconnect()
        sys.exit(f"could not initialize LLM '{settings.llm_model}': {exc}")
    journal = Journal(settings.log_dir)
    try:
        for symbol in settings.symbols:
            ctx = RuntimeContext(
                settings=settings, broker=broker, journal=journal, symbol=symbol,
                _llm=llm, _prep_llm=prep_llm,
            )
            TradingSession(ctx).prep()
        print("\nprep complete. No orders were placed.")
    finally:
        broker.disconnect()


def cmd_replay(args) -> None:
    from .broker.stub import StubBroker
    from .graph.nodes import RuntimeContext
    from .runner import TradingSession

    settings = load_settings()
    symbol = args.symbol or settings.symbols[0]
    if args.csv:
        stub = StubBroker.from_csv(args.csv, args.prev_high, args.prev_low)
    else:
        _, ib = _ib_broker()
        try:
            requested_day = datetime.strptime(args.date, "%Y-%m-%d").date()
            day = datetime.combine(requested_day, datetime.min.time(), tzinfo=ET).replace(hour=20)
            raw = ib.ib.reqHistoricalData(
                ib._underlying(symbol),
                endDateTime=day,
                durationStr="1 D",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=2,
            )
            from .models import Bar

            bars = [
                Bar(ts=b.date.astimezone(ET), open=b.open, high=b.high, low=b.low,
                    close=b.close, volume=float(b.volume or 0))
                for b in raw
            ]
            if not bars:
                sys.exit(f"IB returned no bars for {args.date}")
            from .backtest.data import fetch_daily_series, prev_day_range_for

            daily_bars = fetch_daily_series(ib, symbol, requested_day, requested_day)
            prev_high, prev_low = prev_day_range_for(daily_bars, requested_day)
            stub = StubBroker(
                bars,
                args.prev_high or prev_high,
                args.prev_low or prev_low,
                daily_bars=daily_bars,
            )
        finally:
            ib.disconnect()

    ctx = RuntimeContext(
        settings=settings,
        broker=stub,
        # crash recovery replays logs/journal-*.jsonl, so those files must
        # stay live-only — replay's synthetic fills go in their own directory
        # (backtest already isolates itself the same way)
        journal=Journal(settings.log_dir / "replays"),
        symbol=symbol,
        use_llm=not args.no_llm,
    )
    TradingSession(ctx).run_replay(stub)


def cmd_backtest(args) -> None:
    from datetime import datetime as dt

    from .backtest.runner import print_summary, run_backtest

    # A cached-only research run must be genuinely offline; connecting to IB
    # would make the reproducible A/B gate depend on Gateway availability.
    if args.cached_only:
        settings, ib = load_settings(), None
    else:
        settings, ib = _ib_broker()
    symbol = args.symbol or settings.symbols[0]
    start = dt.strptime(args.start, "%Y-%m-%d").date()
    end = dt.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        sys.exit("--end must not be before --start")
    cache_dir = args.cache_dir or settings.backtest_cache_dir
    try:
        report = run_backtest(
            symbol, start, end, settings, use_llm=not args.no_llm, ib=ib, cache_dir=cache_dir,
            skip_missing_option_data=args.skip_missing_option_data,
            underlying_only=args.underlying_only,
            cached_only=args.cached_only,
            experiment=args.experiment,
        )
    except Exception as exc:  # noqa: BLE001 — e.g. bad LLM config; fail with a clean message
        if ib is not None:
            ib.disconnect()
        sys.exit(f"backtest failed: {exc}")
    if ib is not None:
        ib.disconnect()
    print_summary(report)


def cmd_edge_audit(args) -> None:
    from datetime import date

    from .backtest.audit import load_and_audit, print_audit

    try:
        validation_start = date.fromisoformat(args.validation_start) if args.validation_start else None
        audit = load_and_audit(
            args.report,
            validation_start=validation_start,
            validation_only=args.validation_only,
            min_trades=args.min_trades,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"edge audit failed: {exc}")
    print_audit(audit)


def cmd_forward_validate(args) -> None:
    from datetime import date

    from .backtest.forward import capture_forward_day

    settings = load_settings().model_copy(update={"ib_client_id": args.client_id})
    settings, ib = _ib_broker(settings)
    symbol = (args.symbol or settings.symbols[0]).upper()
    try:
        day = date.fromisoformat(args.date)
        record, cumulative = capture_forward_day(
            name=args.name,
            symbol=symbol,
            day=day,
            settings=settings,
            ib=ib,
            cache_dir=args.cache_dir or settings.backtest_cache_dir,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"forward validation failed: {exc}")
    finally:
        ib.disconnect()
    print(f"captured {symbol} {day}: {record}")
    print(f"cumulative validation report: {cumulative}")


def cmd_forward_init(args) -> None:
    from .backtest.forward import initialize_forward_cohort

    settings = load_settings()
    symbol = (args.symbol or settings.symbols[0]).upper()
    try:
        manifest_path = initialize_forward_cohort(
            name=args.name, symbol=symbol, settings=settings,
        )
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"forward initialization failed: {exc}")
    definition = manifest["definition"]
    print(
        f"locked prospective cohort {manifest['name']} for {symbol}: "
        f"config {definition['fingerprint']}, source {definition['source_fingerprint']}"
    )
    print(
        f"created at: {manifest['created_at']}  "
        f"eligible from: {manifest.get('eligible_from', 'legacy/unrestricted')}  "
        f"captured days: {len(manifest['captured_days'])}"
    )
    print(f"manifest: {manifest_path}")


def cmd_forward_latest(args) -> None:
    from .backtest.forward import capture_forward_day, latest_completed_session

    settings = load_settings().model_copy(update={"ib_client_id": args.client_id})
    settings, ib = _ib_broker(settings)
    symbol = (args.symbol or settings.symbols[0]).upper()
    cache_dir = args.cache_dir or settings.backtest_cache_dir
    try:
        day = latest_completed_session(
            symbol=symbol,
            ib=ib,
            cache_dir=cache_dir,
            lookback_calendar_days=args.lookback_days,
        )
        record, cumulative = capture_forward_day(
            name=args.name,
            symbol=symbol,
            day=day,
            settings=settings,
            ib=ib,
            cache_dir=cache_dir,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"forward latest failed: {exc}")
    finally:
        ib.disconnect()
    print(f"latest completed session: {symbol} {day}")
    print(f"captured: {record}")
    print(f"cumulative validation report: {cumulative}")


def cmd_option_panel_compare(args) -> None:
    from .backtest.audit import paired_panel_rows, print_option_panel

    try:
        report = json.loads(args.report.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"option panel comparison failed: {exc}")
    try:
        paired_panel_rows(report, min_pairs=args.min_pairs)
    except ValueError as exc:
        sys.exit(f"option panel comparison failed: {exc}")
    print_option_panel(report, min_pairs=args.min_pairs)


def cmd_strategy_compare(args) -> None:
    from .backtest.audit import compare_strategy_reports, print_strategy_comparison

    try:
        baseline = json.loads(args.baseline.read_text())
        candidate = json.loads(args.candidate.read_text())
        result = compare_strategy_reports(
            baseline,
            candidate,
            min_trades=args.min_trades,
            min_positive_month_ratio=args.min_positive_month_ratio,
            expected_config_changes=tuple(args.only_change),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"strategy comparison failed: {exc}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print_strategy_comparison(result)
    if args.output is not None:
        print(f"paired comparison report: {args.output}")


def cmd_execution_calibrate(args) -> None:
    from .backtest.audit import calibrate_execution_journal, print_execution_calibration

    settings = load_settings().model_copy(update={"ib_client_id": args.client_id})
    settings, ib = _ib_broker(settings)
    try:
        calibration = calibrate_execution_journal(
            args.journal,
            symbol=args.symbol,
            ib=ib,
            cache_dir=args.cache_dir or settings.backtest_cache_dir,
            settings=settings,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        sys.exit(f"execution calibration failed: {exc}")
    finally:
        ib.disconnect()
    output = args.output or (
        settings.log_dir / "calibrations" /
        f"{args.journal.stem}_{args.symbol.upper()}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, indent=2))
    print_execution_calibration(calibration)
    print(f"calibration report: {output}")


def cmd_shadow_report(args) -> None:
    from .backtest.audit import build_shadow_report, print_shadow_report, write_report

    try:
        report = build_shadow_report(args.path, symbol=args.symbol)
        output = args.output or (
            args.path if args.path.is_dir() else args.path.parent
        ) / f"{args.symbol.upper()}_shadow_report.json"
        write_report(report, output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"shadow report failed: {exc}")
    print_shadow_report(report)
    print(f"shadow report: {output}")


def cmd_historical_signal_tournament(args) -> None:
    from .backtest.historical_signals import (
        print_tournament,
        run_tournament,
        write_tournament,
    )

    settings = load_settings()
    cache_dir = args.cache_dir or settings.backtest_cache_dir
    output = args.output or settings.log_dir / "research" / "historical-signal-tournament-v1.json"
    try:
        report = run_tournament(cache_dir)
        write_tournament(report, output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"historical signal tournament failed: {exc}")
    print_tournament(report)
    print(f"tournament report: {output}")


def cmd_historical_signal_followup(args) -> None:
    from .backtest.historical_signals import (
        print_followup,
        run_opening_drive_fade_followup,
        write_tournament,
    )

    settings = load_settings()
    cache_dir = args.cache_dir or settings.backtest_cache_dir
    output = args.output or settings.log_dir / "research" / "opening-drive-fade-followup-v1.json"
    try:
        report = run_opening_drive_fade_followup(cache_dir)
        write_tournament(report, output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"historical signal follow-up failed: {exc}")
    print_followup(report)
    print(f"follow-up report: {output}")


def cmd_historical_daily_fetch(args) -> None:
    from datetime import date

    from .backtest.daily_signals import fetch_daily_history

    settings = load_settings().model_copy(update={"ib_client_id": args.client_id})
    settings, ib = _ib_broker(settings)
    cache_dir = args.cache_dir or settings.backtest_cache_dir
    symbols = tuple(symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip())
    try:
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
        if end < start:
            raise ValueError("--end must not be before --start")
        for symbol in symbols:
            path = fetch_daily_history(ib, symbol, start, end, cache_dir)
            print(f"cached {symbol}: {path}")
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"historical daily fetch failed: {exc}")
    finally:
        ib.disconnect()


def cmd_historical_daily_tournament(args) -> None:
    from .backtest.daily_signals import (
        print_daily_tournament,
        run_daily_tournament,
        write_daily_tournament,
    )

    settings = load_settings()
    cache_dir = args.cache_dir or settings.backtest_cache_dir
    output = args.output or settings.log_dir / "research" / "historical-daily-tournament-v1.json"
    try:
        report = run_daily_tournament(cache_dir)
        write_daily_tournament(report, output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"historical daily tournament failed: {exc}")
    print_daily_tournament(report)
    print(f"daily tournament report: {output}")


def cmd_historical_aapl_focus(args) -> None:
    from .backtest.daily_signals import (
        print_aapl_focused,
        run_aapl_focused_holdout,
        write_daily_tournament,
    )

    settings = load_settings()
    output = args.output or settings.log_dir / "research" / "aapl-focused-holdout-v1.json"
    try:
        report = run_aapl_focused_holdout(args.cache_dir)
        write_daily_tournament(report, output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"AAPL-focused historical holdout failed: {exc}")
    print_aapl_focused(report)
    print(f"AAPL-focused report: {output}")


if __name__ == "__main__":
    main()
