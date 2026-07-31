"""CLI: tajator {run | check-ib | replay}."""

from __future__ import annotations

import argparse
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


def _validate_tws_chain_snapshot(args, start, end, now: datetime) -> None:
    """Keep the live-chain replay diagnostic narrow and strictly post-session."""
    if args.cached_only or args.underlying_only:
        raise ValueError("--tws-chain-snapshot requires an online exact-option backtest")
    if start != end or start != now.date():
        raise ValueError(
            "--tws-chain-snapshot is restricted to a single current-day diagnostic"
        )
    if (now.hour, now.minute) < (16, 0):
        raise ValueError("--tws-chain-snapshot requires the current session to be complete")


def _runtime_policy_metadata(settings) -> dict:
    """Journal enough identity to describe the run."""
    return {
        "policy_mode": "deterministic",
        "strategy": "orb",
        "symbols": settings.symbols,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="tajator")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="live minute loop against IBKR (paper by default)")

    check_ib = sub.add_parser(
        "check-ib", help="connectivity check: bars, chain, quote — no orders"
    )
    check_ib.add_argument(
        "--client-id", type=int, default=118,
        help="dedicated read-only diagnostic client ID (default: 118)",
    )
    check_ib.add_argument(
        "--entry-samples", type=int, choices=range(1, 6), default=1,
        help="paired entry-data samples per symbol, 1-5 (recommended: 3)",
    )

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
    replay.add_argument("--prev-high", type=float, default=None)
    replay.add_argument("--prev-low", type=float, default=None)

    backtest = sub.add_parser(
        "backtest", help="step the graph over a date range from IB with real historical option fills"
    )
    backtest.add_argument("--symbol", default=None, help="defaults to the first configured SYMBOLS entry")
    backtest.add_argument("--start", required=True, help="YYYY-MM-DD")
    backtest.add_argument("--end", required=True, help="YYYY-MM-DD")
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
    backtest.add_argument(
        "--tws-chain-snapshot", action="store_true",
        help="single current-day diagnostic using the actual TWS expiration/strike chain",
    )
    backtest.add_argument("--experiment", default="baseline", help="report/journal experiment label")

    compare = sub.add_parser("backtest-compare", help="compare experiment-safe backtest JSON reports")
    compare.add_argument("reports", nargs="+", type=Path)

    sweep = sub.add_parser(
        "orb-sweep",
        help="search ORB variants on cached bars, validate the best out-of-sample (offline)",
    )
    sweep.add_argument("--symbols", default=None, help="comma-separated; defaults to SYMBOLS")
    sweep.add_argument("--dev-start", required=True, help="YYYY-MM-DD")
    sweep.add_argument("--dev-end", required=True, help="YYYY-MM-DD")
    sweep.add_argument("--holdout-start", required=True, help="YYYY-MM-DD")
    sweep.add_argument("--holdout-end", required=True, help="YYYY-MM-DD")
    sweep.add_argument("--min-trades", type=int, default=30)
    sweep.add_argument("--top-k", type=int, default=8)
    sweep.add_argument("--cache-dir", type=Path, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "run":
        cmd_run(args)
    elif args.command == "check-ib":
        cmd_check_ib(args)
    elif args.command == "test-order":
        cmd_test_order(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "backtest-compare":
        from .backtest.compare import print_comparison
        print_comparison(args.reports)
    elif args.command == "orb-sweep":
        cmd_orb_sweep(args)
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
    journal.write("policy_start", ts=broker.now(), **_runtime_policy_metadata(settings))
    store = StateStore(settings.state_file)
    try:
        # Refuse on resting orders or positions that persisted state cannot
        # explain; adopt positions from a previous run that match exactly.
        adopted = run_startup_checks(settings, broker, store, journal, notifier)
    except SystemExit:
        broker.disconnect()
        raise
    today = broker.now().date()
    sessions = [
        TradingSession(
            RuntimeContext(
                settings=settings, broker=broker, journal=journal, symbol=symbol,
                notifier=notifier,
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


def _streaming_entry_market_diagnostic(broker, contract, timeout_s: float = 5.0):
    """Read-only trial of concurrent temporary streams; never submits an order."""
    opt = broker._option(contract)
    stock = broker._underlying(contract.symbol)
    subscriptions = []
    started = time.monotonic()
    bid = ask = underlying = None
    cleanup_errors = []
    try:
        option_ticker = broker.ib.reqMktData(opt, snapshot=False)
        subscriptions.append(opt)
        stock_ticker = broker.ib.reqMktData(stock, snapshot=False)
        subscriptions.append(stock)
        while True:
            bid = broker._positive_price(getattr(option_ticker, "bid", None))
            ask = broker._positive_price(getattr(option_ticker, "ask", None))
            underlying = broker._positive_price(stock_ticker.marketPrice())
            if bid is not None and ask is not None and underlying is not None:
                return (
                    broker._option_quote_from_ticker(option_ticker),
                    underlying,
                    time.monotonic() - started,
                )
            remaining = timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError(
                    f"no complete option/stock quote after {timeout_s:.1f}s "
                    f"(bid={bid}, ask={ask}, underlying={underlying})"
                )
            broker.ib.waitOnUpdate(timeout=min(0.25, remaining))
    finally:
        for subscribed in subscriptions:
            try:
                broker.ib.cancelMktData(subscribed)
            except Exception as exc:  # noqa: BLE001 — diagnostic cleanup is best effort
                cleanup_errors.append(str(exc))
                logging.getLogger(__name__).warning(
                    "could not cancel diagnostic market-data stream: %s", exc,
                )
        if cleanup_errors:
            raise RuntimeError(
                "temporary market-data cleanup failed: " + "; ".join(cleanup_errors)
            )


def _run_entry_market_data_pair(
    broker, settings, diagnostics, contract, symbol: str, sample_index: int,
) -> None:
    """Persist one no-order temporary-stream/production-snapshot pair."""
    from .trade.execution import validate_option_liquidity

    diagnostic_started_at = broker.now()
    diagnostic_id = (
        f"{symbol}:{diagnostic_started_at.isoformat()}:sample-{sample_index}"
    )
    entry_window = (9, 30) <= (
        diagnostic_started_at.hour, diagnostic_started_at.minute,
    ) <= (14, 0)
    common = {
        "symbol": symbol,
        "diagnostic_id": diagnostic_id,
        "sample_index": sample_index,
        "regular_entry_window": entry_window,
        "contract": contract,
        "no_order_placed": True,
    }
    stream_started = time.monotonic()
    try:
        quote, underlying, elapsed = _streaming_entry_market_diagnostic(
            broker, contract,
        )
        diagnostics.write(
            "entry_market_data_diagnostic",
            ts=broker.now(),
            method="temporary_streams",
            elapsed_seconds=round(elapsed, 4),
            complete_bid_ask=quote.bid is not None and quote.ask is not None,
            liquidity_reason=validate_option_liquidity(
                quote, settings, now=broker.now(),
            ),
            option_quote=quote,
            underlying_price=underlying,
            **common,
        )
        print(
            f"sample {sample_index} experimental streams ({elapsed:.2f}s): "
            f"underlying {underlying}; {contract.local_name} "
            f"bid {quote.bid} / ask {quote.ask} / last {quote.last}"
        )
    except Exception as exc:  # noqa: BLE001 — production check must still run
        elapsed = time.monotonic() - stream_started
        diagnostics.write(
            "entry_market_data_diagnostic",
            ts=broker.now(),
            method="temporary_streams",
            elapsed_seconds=round(elapsed, 4),
            complete_bid_ask=False,
            error=str(exc),
            **common,
        )
        print(f"sample {sample_index} experimental streams unavailable: {exc}")

    started = time.monotonic()
    try:
        quote, underlying = broker.get_entry_market_snapshot(contract)
    except Exception as exc:  # noqa: BLE001 — keep checking later samples
        elapsed = time.monotonic() - started
        diagnostics.write(
            "entry_market_data_diagnostic",
            ts=broker.now(),
            method="production_snapshot",
            elapsed_seconds=round(elapsed, 4),
            complete_bid_ask=False,
            error=str(exc),
            **common,
        )
        print(
            f"sample {sample_index} production snapshot failed "
            f"after {elapsed:.2f}s: {exc}"
        )
    else:
        elapsed = time.monotonic() - started
        age = max(0.0, (broker.now() - quote.ts).total_seconds())
        diagnostics.write(
            "entry_market_data_diagnostic",
            ts=broker.now(),
            method="production_snapshot",
            elapsed_seconds=round(elapsed, 4),
            complete_bid_ask=quote.bid is not None and quote.ask is not None,
            liquidity_reason=validate_option_liquidity(
                quote, settings, now=broker.now(),
            ),
            option_quote=quote,
            underlying_price=underlying,
            **common,
        )
        print(
            f"sample {sample_index} production snapshot ({elapsed:.2f}s): "
            f"underlying {underlying}; {contract.local_name} "
            f"bid {quote.bid} / ask {quote.ask} / last {quote.last}; "
            f"quote age {age:.2f}s"
        )


def cmd_check_ib(args) -> None:
    from .trade.contracts import select_contract

    settings = load_settings().model_copy(update={"ib_client_id": args.client_id})
    settings, broker = _ib_broker(settings)
    diagnostics = Journal(settings.log_dir / "diagnostics")
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
                    for sample_index in range(1, args.entry_samples + 1):
                        _run_entry_market_data_pair(
                            broker, settings, diagnostics, contract, symbol, sample_index,
                        )
                        if sample_index < args.entry_samples:
                            broker.ib.sleep(5.0)
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

    replay_journal = Journal(settings.log_dir / "replays")
    ctx = RuntimeContext(
        settings=settings,
        broker=stub,
        # crash recovery replays logs/journal-*.jsonl, so those files must
        # stay live-only — replay's synthetic fills go in their own directory
        # (backtest already isolates itself the same way)
        journal=replay_journal,
        symbol=symbol,
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
    chain_override = None
    try:
        if args.tws_chain_snapshot:
            _validate_tws_chain_snapshot(args, start, end, datetime.now(ET))
            chain_override = ib.get_option_chain(symbol)
        report = run_backtest(
            symbol, start, end, settings, ib=ib, cache_dir=cache_dir,
            skip_missing_option_data=args.skip_missing_option_data,
            underlying_only=args.underlying_only,
            cached_only=args.cached_only,
            experiment=args.experiment,
            chain_override=chain_override,
        )
    except Exception as exc:  # noqa: BLE001 — fail with a clean message
        if ib is not None:
            ib.disconnect()
        sys.exit(f"backtest failed: {exc}")
    if ib is not None:
        ib.disconnect()
    print_summary(report)


def cmd_orb_sweep(args) -> None:
    from datetime import datetime as dt

    from .backtest.sweep import print_sweep, run_sweep, write_sweep

    settings = load_settings()
    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else settings.symbols
    )
    cache_dir = args.cache_dir or settings.backtest_cache_dir
    windows = {
        name: dt.strptime(getattr(args, name.replace("-", "_")), "%Y-%m-%d").date()
        for name in ("dev-start", "dev-end", "holdout-start", "holdout-end")
    }
    result = run_sweep(
        symbols,
        windows["dev-start"], windows["dev-end"],
        windows["holdout-start"], windows["holdout-end"],
        settings, cache_dir,
        min_trades=args.min_trades, top_k=args.top_k,
    )
    print_sweep(result)
    path = write_sweep(result, settings.log_dir)
    print(f"\nwrote {path}")


