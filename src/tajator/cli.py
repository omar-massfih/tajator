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

ET = ZoneInfo("America/New_York")


def main() -> None:
    parser = argparse.ArgumentParser(prog="tajator")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="live minute loop against IBKR (paper by default)")
    sub.add_parser("check-ib", help="connectivity check: bars, chain, quote — no orders")
    sub.add_parser("prep", help="run pre-market prep now: levels + LLM briefing — no orders")

    replay = sub.add_parser("replay", help="step the graph through a recorded day (no IB orders)")
    src = replay.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="CSV of 1-min bars (ts,open,high,low,close,volume)")
    src.add_argument("--date", help="YYYY-MM-DD — fetch that day's bars from IB once")
    replay.add_argument("--symbol", default=None, help="defaults to the first configured SYMBOLS entry")
    replay.add_argument("--no-llm", action="store_true", help="deterministic rule-follower instead of the LLM")
    replay.add_argument("--prev-high", type=float, default=None)
    replay.add_argument("--prev-low", type=float, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "run":
        cmd_run()
    elif args.command == "check-ib":
        cmd_check_ib()
    elif args.command == "prep":
        cmd_prep()
    else:
        cmd_replay(args)


def _ib_broker():
    from .broker.ib import IBBroker

    settings = load_settings()
    broker = IBBroker(settings)
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

    settings, broker = _ib_broker()
    try:
        # fail fast on a missing/invalid API key instead of waiting all day
        llm = build_llm(settings.llm_model)
        prep_llm = build_llm(settings.llm_model, output_model=MorningBriefing)
    except Exception as exc:  # noqa: BLE001
        broker.disconnect()
        sys.exit(f"could not initialize LLM '{settings.llm_model}': {exc}")
    journal = Journal(settings.log_dir)
    sessions = [
        TradingSession(
            RuntimeContext(
                settings=settings, broker=broker, journal=journal, symbol=symbol,
                _llm=llm, _prep_llm=prep_llm,
            )
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
        journal=Journal(settings.log_dir),
        symbol=symbol,
        use_llm=not args.no_llm,
    )
    TradingSession(ctx).run_replay(stub)


if __name__ == "__main__":
    main()
