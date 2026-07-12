"""CLI: tajator {run | check-ib | replay}."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import load_settings
from .journal import Journal
from .notify import NullNotifier, TelegramNotifier

ET = ZoneInfo("America/New_York")


def main() -> None:
    parser = argparse.ArgumentParser(prog="tajator")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="live minute loop against IBKR (paper by default)")
    sub.add_parser("check-ib", help="connectivity check: bars, chain, quote — no orders")
    sub.add_parser("prep", help="run pre-market prep now: levels + LLM briefing — no orders")

    test_order = sub.add_parser(
        "test-order",
        help="supervised paper diagnostic: buy 1 lot, watch the fill timeline, sell it back",
    )
    test_order.add_argument("--symbol", default=None, help="defaults to the first configured SYMBOLS entry")
    test_order.add_argument("--qty", type=int, default=1)
    test_order.add_argument("--wait", type=int, default=180, help="seconds to wait for each fill")
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

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "run":
        cmd_run()
    elif args.command == "check-ib":
        cmd_check_ib()
    elif args.command == "test-order":
        cmd_test_order(args)
    elif args.command == "prep":
        cmd_prep()
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "backtest-compare":
        from .backtest.compare import print_comparison
        print_comparison(args.reports)
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


def cmd_run() -> None:
    from .graph.nodes import RuntimeContext
    from .llm.decide import build_llm
    from .models import MorningBriefing
    from .runner import LiveRunner, TradingSession
    from .startup import check_kill_switch, run_startup_checks
    from .state_store import StateStore

    settings = load_settings()
    notifier = _notifier(settings)
    check_kill_switch(settings, notifier)
    settings, broker = _ib_broker(settings, notifier)
    journal = Journal(settings.log_dir)
    broker.journal = journal  # order timelines land next to the trade records
    store = StateStore(settings.state_file)
    try:
        # Refuse on resting orders or positions that persisted state cannot
        # explain; adopt positions from a previous run that match exactly.
        adopted = run_startup_checks(settings, broker, store, journal, notifier)
    except SystemExit:
        broker.disconnect()
        raise
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


def cmd_check_ib() -> None:
    from .trade.contracts import select_contract

    settings, broker = _ib_broker()
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
                    premium = broker.get_option_premium(contract)
                    print(f"nearest-strike call: {contract.local_name} — premium {premium}")
        print("\ncheck-ib complete. No orders were placed.")
    finally:
        broker.disconnect()


def cmd_test_order(args) -> None:
    """Supervised paper diagnostic (see the 2026-07-08 incident): observe the
    natural fill latency of a market order WITHOUT the timeout-cancel path, so
    slow paper-sim fills can be told apart from lost events. Watch TWS while
    this runs. Exit code 0 only when the full round trip completed."""
    import time as time_mod

    from ib_async import MarketOrder

    from .trade.contracts import select_contract

    settings = load_settings()
    if settings.trading_mode != "paper":
        sys.exit("test-order is a paper diagnostic — refusing to run in live mode.")
    settings, broker = _ib_broker(settings)
    broker.journal = Journal(settings.log_dir)
    symbol = (args.symbol or settings.symbols[0]).upper()

    def stream(trade, wait_s: int) -> float | None:
        """Print status/log lines as they arrive; returns seconds-to-done or None."""
        start = time_mod.monotonic()
        seen = 0
        while True:
            for e in trade.log[seen:]:
                print(f"    {e.time.astimezone(ET):%H:%M:%S}  {e.status:<14} {e.message}")
            seen = len(trade.log)
            if trade.isDone():
                return time_mod.monotonic() - start
            if time_mod.monotonic() - start >= wait_s:
                return None
            broker.ib.waitOnUpdate(timeout=1.0)

    ok = False
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
        opt = broker._option(contract)
        [ticker] = broker.ib.reqTickers(opt)
        print(f"{contract.local_name}: bid {ticker.bid} / ask {ticker.ask} / last {ticker.last} "
              f"({'DELAYED' if broker.is_delayed_data else 'live'} quotes)")
        if broker.is_delayed_data:
            print("!!! quotes are DELAYED — expect slow/none paper fills; fix the market "
                  "data subscription before trusting entry timing.")

        print(f"\nplacing BUY {args.qty}x {contract.local_name} (market, NO timeout cancel) ...")
        buy = MarketOrder("BUY", args.qty)
        buy.orderRef = f"{settings.order_ref_prefix}-test:{symbol}"
        buy_trade = broker.ib.placeOrder(opt, buy)
        took = stream(buy_trade, args.wait)
        broker._journal_order_timeline(buy_trade, f"test-order BUY {args.qty}x {contract.local_name}")
        if took is None or buy_trade.orderStatus.status != "Filled":
            print(f"\nNOT FILLED within {args.wait}s (status {buy_trade.orderStatus.status}).")
            print("The order was NOT auto-cancelled — watch it in TWS and cancel it there "
                  "once you have seen how long the paper engine takes.")
            sys.exit(1)
        filled_qty = int(buy_trade.orderStatus.filled)
        print(f"filled {filled_qty}x @ {buy_trade.orderStatus.avgFillPrice} in {took:.1f}s")

        if args.with_stop:
            stop_price = round(spot - 1.00, 2)
            print(f"\nplacing protective stop (SELL if {symbol} <= {stop_price}, GTC) ...")
            stop = broker.place_protective_stop(
                contract, filled_qty, stop_price, "call",
                f"{settings.order_ref_prefix}-stop:{symbol}",
            )
            print(f"placed order {stop.order_id} (ref {stop.order_ref}) — check it shows in TWS")
            status = broker.poll_protective_stop(contract, stop)
            print(f"poll: {status.state} (working {status.working_qty})")
            result = broker.cancel_protective_stop(contract, stop, expected_held=filled_qty)
            print(f"cancel confirmed: cancelled={result.cancelled}, filled={result.filled_qty}")

        print(f"\nselling {filled_qty}x back (market, NO timeout cancel) ...")
        sell = MarketOrder("SELL", filled_qty)
        sell.orderRef = f"{settings.order_ref_prefix}-test:{symbol}"
        sell_trade = broker.ib.placeOrder(opt, sell)
        took = stream(sell_trade, args.wait)
        broker._journal_order_timeline(sell_trade, f"test-order SELL {filled_qty}x {contract.local_name}")
        if took is None or sell_trade.orderStatus.status != "Filled":
            print(f"\nSELL NOT FILLED within {args.wait}s (status {sell_trade.orderStatus.status}) — "
                  "the position is still open; close it in TWS.")
            sys.exit(1)
        buy_px = float(buy_trade.orderStatus.avgFillPrice)
        sell_px = float(sell_trade.orderStatus.avgFillPrice)
        print(f"filled {int(sell_trade.orderStatus.filled)}x @ {sell_px} in {took:.1f}s")
        print(f"\nround trip complete: PnL ${100 * filled_qty * (sell_px - buy_px):+.0f} "
              "(timelines journaled as order_timeline events)")
        ok = True
    finally:
        broker.disconnect()
    if not ok:
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
            day = datetime.strptime(args.date, "%Y-%m-%d").replace(hour=20, tzinfo=ET)
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
            prev_high, prev_low = ib.get_prev_day_range(symbol)
            stub = StubBroker(bars, args.prev_high or prev_high, args.prev_low or prev_low)
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
        ib.disconnect()
        sys.exit(f"backtest failed: {exc}")
    ib.disconnect()
    print_summary(report)


if __name__ == "__main__":
    main()
