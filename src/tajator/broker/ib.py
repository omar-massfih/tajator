"""Interactive Brokers via ib_async (maintained fork of ib_insync).

Requires IB Gateway running with the API enabled
(Configure → API → Settings → Enable ActiveX and Socket Clients).
Port 4002 = IB Gateway paper, 4001 = IB Gateway live.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from ib_async import IB, MarketOrder, Option, Stock
from ib_async.order import PriceCondition

from ..config import Settings
from ..journal import Journal
from ..models import (
    Bar,
    Direction,
    ExecutionQuality,
    OptionQuote,
    ProtectiveStop,
    SelectedContract,
)
from ..notify import Notifier, NullNotifier
from .base import (
    Broker,
    BrokerOpenOrder,
    BrokerOptionPosition,
    ChainParams,
    Fill,
    OrderFailed,
    StopCancelResult,
    StopStatus,
)

ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)

NO_SUBSCRIPTION_CODES = {354, 10167, 10197}
CANCEL_TIMEOUT_S = 10
STOP_ACK_TIMEOUT_S = 5  # wait for a protective stop to acknowledge at IB


class IBBroker(Broker):
    def __init__(self, settings: Settings, notifier: Notifier | None = None):
        self.settings = settings
        self.ib = IB()
        self.notifier = notifier or NullNotifier()
        self.journal: Journal | None = None  # set by callers that want order timelines
        self._stocks: dict[str, Stock] = {}
        self._chain_cache: dict[str, tuple[str, ChainParams]] = {}
        self._daily_bars_cache: dict[str, tuple[str, list[Bar]]] = {}
        self._qualified: dict[str, Option] = {}
        self._delayed = False
        # subscribe before connect() so connect-time errors are seen too
        self.ib.errorEvent += self._on_error
        # always-on order diagnostics — the 07-08 incident was undiagnosable
        # without a record of when statuses and executions actually arrived
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec_details
        self.ib.commissionReportEvent += self._on_commission_report

    # -- lifecycle ---------------------------------------------------------------

    @property
    def uses_live_execution_guards(self) -> bool:
        return True

    def connect(self) -> None:
        s = self.settings
        self.ib.connect(s.ib_host, s.ib_port, clientId=s.ib_client_id, timeout=10)
        self.ib.reqMarketDataType(3 if self._delayed else s.market_data_type)
        log.info("connected to IB %s:%s (mode=%s)", s.ib_host, s.ib_port, s.trading_mode)

    def ensure_connected(self) -> bool:
        """Reconnect after a dropped session (e.g. the Gateway's nightly restart).

        Called once per tick; a failed attempt raises and is retried on the
        next tick, so the minute cadence doubles as the retry backoff."""
        if self.ib.isConnected():
            return False
        log.warning("IB connection lost — reconnecting to %s:%s",
                    self.settings.ib_host, self.settings.ib_port)
        self.ib.disconnect()  # clear any half-open client state before redialing
        self.connect()
        return True

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def _on_error(self, reqId, errorCode, errorString, *args) -> None:
        if errorCode in NO_SUBSCRIPTION_CODES and not self._delayed:
            self._delayed = True
            self.ib.reqMarketDataType(3)
            log.warning(
                "NO MARKET DATA SUBSCRIPTION (IB error %s) — falling back to DELAYED data. "
                "Fine for plumbing tests, NOT for timing real entries.", errorCode,
            )

    def _on_order_status(self, trade) -> None:
        o, s = trade.order, trade.orderStatus
        log.info(
            "order status #%s/%s [%s] %s %gx %s: %s (filled %g @ %s)",
            o.orderId, o.permId or "-", o.orderRef or "-", o.action, o.totalQuantity,
            getattr(trade.contract, "localSymbol", "") or trade.contract.symbol,
            s.status, s.filled or 0, s.avgFillPrice or 0,
        )

    def _on_exec_details(self, trade, fill) -> None:
        ex = fill.execution
        log.info(
            "execution #%s/%s: %s %gx @ %s (%s)",
            ex.orderId, ex.permId or "-", ex.side, ex.shares, ex.price, ex.time,
        )

    def _on_commission_report(self, trade, fill, report) -> None:
        if self.journal is None:
            return
        execution = getattr(fill, "execution", None)
        self.journal.write(
            "commission_report",
            symbol=getattr(trade.contract, "symbol", ""),
            order_id=getattr(execution, "orderId", None),
            exec_id=getattr(execution, "execId", ""),
            commission=float(getattr(report, "commission", 0.0) or 0.0),
            currency=getattr(report, "currency", ""),
            realized_pnl=getattr(report, "realizedPNL", None),
        )

    def _journal_order_timeline(self, trade, label: str) -> None:
        """Persist the order's full status history — when each state actually
        arrived is the evidence that separates a slow fill from a lost one."""
        if self.journal is None:
            return
        o = trade.order
        self.journal.write(
            "order_timeline",
            symbol=getattr(trade.contract, "symbol", ""),
            label=label,
            order_id=o.orderId,
            perm_id=o.permId or None,
            order_ref=o.orderRef or "",
            timeline=[
                {"t": e.time.isoformat(), "status": e.status, "msg": e.message}
                for e in getattr(trade, "log", [])
            ],
        )

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    @property
    def is_delayed_data(self) -> bool:
        return self._delayed

    def option_positions(self, symbols: list[str]) -> list[BrokerOptionPosition]:
        """Existing option positions in the account for the given underlyings.
        Startup reconciles these against persisted state before trading."""
        found = []
        for pos in self.ib.positions():
            c = pos.contract
            if c.secType == "OPT" and c.symbol in symbols and pos.position:
                found.append(
                    BrokerOptionPosition(
                        symbol=c.symbol,
                        expiry=getattr(c, "lastTradeDateOrContractMonth", "") or "",
                        strike=float(getattr(c, "strike", 0) or 0),
                        right=(getattr(c, "right", "") or "?")[:1],
                        con_id=int(getattr(c, "conId", 0) or 0),
                        local_symbol=c.localSymbol or c.symbol,
                        qty=int(pos.position),
                        avg_cost=float(pos.avgCost or 0),
                    )
                )
        return found

    def open_option_positions(self, symbols: list[str]) -> list[str]:
        """Same, as human-readable strings for operator messages."""
        return [
            f"{p.qty:+g}x {p.local_symbol} (avg cost {p.avg_cost:.2f})"
            for p in self.option_positions(symbols)
        ]

    def open_option_orders_detailed(self, symbols: list[str]) -> list[BrokerOpenOrder]:
        """Resting orders at IB for the given underlyings. reqAllOpenOrders sees
        orders from other client IDs and manual TWS orders too — e.g. one left
        working by a crashed session — and re-binds our own GTC orders."""
        self.ib.reqAllOpenOrders()
        found = []
        for trade in self.ib.openTrades():
            c = trade.contract
            if c.secType == "OPT" and c.symbol in symbols:
                o = trade.order
                found.append(
                    BrokerOpenOrder(
                        order_id=int(o.orderId or 0),
                        perm_id=int(o.permId) if o.permId else None,
                        order_ref=o.orderRef or "",
                        action=o.action,
                        qty=int(o.totalQuantity),
                        order_type=o.orderType,
                        status=trade.orderStatus.status,
                        con_id=int(getattr(c, "conId", 0) or 0),
                        local_symbol=c.localSymbol or c.symbol,
                        symbol=c.symbol,
                        expiry=getattr(c, "lastTradeDateOrContractMonth", "") or "",
                        strike=float(getattr(c, "strike", 0) or 0),
                        right=(getattr(c, "right", "") or "")[:1],
                    )
                )
        return found

    def open_option_orders(self, symbols: list[str]) -> list[str]:
        """Same, as human-readable strings for operator messages."""
        return [
            f"{o.action} {o.qty:g}x {o.local_symbol} ({o.order_type}, status {o.status})"
            for o in self.open_option_orders_detailed(symbols)
        ]

    def other_positions_summary(self, symbols: list[str]) -> list[str]:
        """Nonzero positions the startup guard does not gate on — stock, or
        options on unconfigured underlyings — reported so nothing sits unseen."""
        found = []
        for pos in self.ib.positions():
            c = pos.contract
            if pos.position and not (c.secType == "OPT" and c.symbol in symbols):
                found.append(f"{pos.position:+g}x {c.secType} {c.localSymbol or c.symbol}")
        return found

    def _underlying(self, symbol: str) -> Stock:
        if symbol not in self._stocks:
            stock = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(stock)
            self._stocks[symbol] = stock
        return self._stocks[symbol]

    # -- Broker interface -----------------------------------------------------------

    def now(self) -> datetime:
        return datetime.now(ET)

    def get_bars(self, symbol: str, lookback_minutes: int = 390) -> list[Bar]:
        raw = self.ib.reqHistoricalData(
            self._underlying(symbol),
            endDateTime="",
            durationStr=f"{lookback_minutes * 60} S",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=False,  # premarket bars feed premarket levels
            formatDate=2,
        )
        return [
            Bar(
                ts=b.date.astimezone(ET),
                open=b.open, high=b.high, low=b.low, close=b.close,
                volume=float(b.volume) if b.volume and not math.isnan(b.volume) else 0.0,
            )
            for b in raw
        ]

    def get_prev_day_range(self, symbol: str) -> tuple[float | None, float | None]:
        daily = self.ib.reqHistoricalData(
            self._underlying(symbol),
            endDateTime="",
            durationStr="5 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
        )
        if not daily:
            return None, None
        today = self.now().date()
        prior = [b for b in daily if _bar_date(b.date) < today]
        if not prior:
            return None, None
        prev = prior[-1]
        return float(prev.high), float(prev.low)

    def get_daily_bars(self, symbol: str, lookback_days: int = 90) -> list[Bar]:
        """Fetch daily history once per symbol/session; consumers exclude today causally."""
        today = self.now().date().isoformat()
        cached = self._daily_bars_cache.get(symbol)
        if cached and cached[0] == today:
            return cached[1][-lookback_days:]
        # 140 calendar days comfortably covers 90 US trading sessions.
        raw = self.ib.reqHistoricalData(
            self._underlying(symbol),
            endDateTime="",
            durationStr="140 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
        )
        bars = [
            Bar(
                ts=(b.date.astimezone(ET) if isinstance(b.date, datetime)
                    and b.date.tzinfo is not None else
                    datetime.combine(_bar_date(b.date), datetime.min.time(), tzinfo=ET)),
                open=float(b.open), high=float(b.high), low=float(b.low), close=float(b.close),
                volume=float(b.volume) if b.volume and not math.isnan(b.volume) else 0.0,
            )
            for b in raw
        ]
        self._daily_bars_cache[symbol] = (today, bars)
        return bars[-lookback_days:]

    def get_option_chain(self, symbol: str) -> ChainParams:
        today = self.now().date().isoformat()
        cached = self._chain_cache.get(symbol)
        if cached and cached[0] == today:
            return cached[1]
        stock = self._underlying(symbol)
        params = self.ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
        if not params:
            return ChainParams(expirations=[], strikes=[])
        # A symbol can return several entries per exchange (one per trading
        # class, some with only a couple of strikes). Prefer SMART entries,
        # then take the richest chain among them.
        smarts = [p for p in params if p.exchange == "SMART"] or list(params)
        best = max(smarts, key=lambda p: (len(p.strikes), len(p.expirations)))
        chain = ChainParams(
            expirations=sorted(best.expirations), strikes=sorted(best.strikes)
        )
        self._chain_cache[symbol] = (today, chain)
        return chain

    def _option(self, contract: SelectedContract, include_expired: bool = False) -> Option:
        # include_expired is a backtest-only escape hatch: IB won't resolve an
        # expired option's security definition (needed for its historical bars)
        # unless the request opts in. Live trading never touches expired
        # contracts, so it keeps the default and this path stays unchanged.
        key = contract.local_name
        if key not in self._qualified:
            opt = Option(
                contract.symbol, contract.expiry, contract.strike, contract.right,
                "SMART", currency="USD",
            )
            if include_expired:
                opt.includeExpired = True
            qualified = self.ib.qualifyContracts(opt)
            if not qualified:
                raise RuntimeError(f"could not qualify option {key}")
            self._qualified[key] = opt
        opt = self._qualified[key]
        if contract.con_id is None and opt.conId:
            contract.con_id = opt.conId  # persisted state matches on conId
        return opt

    def get_option_premium(self, contract: SelectedContract) -> float | None:
        opt = self._option(contract)
        [ticker] = self.ib.reqTickers(opt)
        price = ticker.marketPrice()
        if price is None or math.isnan(price) or price <= 0:
            price = ticker.close
            if price is not None and not math.isnan(price) and price > 0:
                log.warning(
                    "no live quote for %s — using last close %.2f for sizing "
                    "(may be stale)", contract.local_name, price,
                )
        if price is None or math.isnan(price) or price <= 0:
            return None
        return float(price)

    @staticmethod
    def _positive_price(value) -> float | None:
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        return price if math.isfinite(price) and price > 0 else None

    def get_option_quote(self, contract: SelectedContract) -> OptionQuote:
        opt = self._option(contract)
        [ticker] = self.ib.reqTickers(opt)
        quote_ts = getattr(ticker, "time", None)
        if isinstance(quote_ts, datetime):
            quote_ts = (
                quote_ts.replace(tzinfo=ET)
                if quote_ts.tzinfo is None
                else quote_ts.astimezone(ET)
            )
        else:
            quote_ts = self.now()
        return OptionQuote(
            bid=self._positive_price(getattr(ticker, "bid", None)),
            ask=self._positive_price(getattr(ticker, "ask", None)),
            last=self._positive_price(getattr(ticker, "last", None)),
            ts=quote_ts,
            delayed=self._delayed,
            source="ib_delayed" if self._delayed else "ib_live",
        )

    def get_underlying_price(self, symbol: str) -> float | None:
        [ticker] = self.ib.reqTickers(self._underlying(symbol))
        return self._positive_price(ticker.marketPrice())

    def record_execution_preflight(self, **payload: object) -> None:
        if self.journal is not None:
            self.journal.write("execution_preflight", **payload)

    def buy_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._place(contract, "BUY", qty)

    def sell_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._place(contract, "SELL", qty)

    def _place(self, contract: SelectedContract, side: str, qty: int) -> Fill:
        opt = self._option(contract)
        qty_before = self._snapshot_position(opt)
        try:
            arrival_quote = self.get_option_quote(contract)
        except Exception:  # exits must never fail merely because a quote request failed
            log.exception("could not capture arrival quote for %s", contract.local_name)
            arrival_quote = None
        try:
            underlying_submit = self.get_underlying_price(contract.symbol)
        except Exception:
            log.exception("could not capture underlying at submission for %s", contract.symbol)
            underlying_submit = None
        order = MarketOrder(side, qty)
        order.tif = "DAY"
        order.orderRef = f"{self.settings.order_ref_prefix}:{contract.symbol}"
        submitted_at = self.now()
        started = time.monotonic()
        trade = self.ib.placeOrder(opt, order)
        deadline = time.monotonic() + self.settings.order_timeout_s
        while not trade.isDone() and time.monotonic() < deadline:
            self.ib.waitOnUpdate(timeout=1.0)
        if not trade.isDone():
            # Never leave a market order working untracked: cancel, then wait
            # briefly for the terminal status so partial fills are reported.
            self.ib.cancelOrder(trade.order)
            cancel_deadline = time.monotonic() + CANCEL_TIMEOUT_S
            while not trade.isDone() and time.monotonic() < cancel_deadline:
                self.ib.waitOnUpdate(timeout=1.0)
        try:
            if trade.orderStatus.status == "Filled":
                fill = Fill(
                premium=float(trade.orderStatus.avgFillPrice), qty=qty, ts=self.now()
                )
            else:
                fill = self._resolve_unfilled(trade, opt, contract, side, qty, qty_before)
        except Exception:
            self._journal_order_timeline(
                trade, f"{side} {qty}x {contract.local_name} failed"
            )
            raise
        self._journal_order_timeline(
            trade, f"{side} {qty}x {contract.local_name} filled {fill.qty}/{qty}"
        )
        return self._finalize_execution_quality(
            fill, trade, contract, side, qty, arrival_quote,
            submitted_at, started, underlying_submit,
        )

    def _finalize_execution_quality(
        self, fill: Fill, trade, contract: SelectedContract, side: str, requested: int,
        quote: OptionQuote | None, submitted_at: datetime, started: float,
        underlying_submit: float | None,
    ) -> Fill:
        filled_at = fill.ts
        latency = max(0.0, time.monotonic() - started)
        try:
            underlying_fill = self.get_underlying_price(contract.symbol)
        except Exception:
            log.exception("could not capture underlying at fill for %s", contract.symbol)
            underlying_fill = None
        reference = None
        if quote is not None and quote.valid:
            reference = quote.ask if side == "BUY" else quote.bid
        slippage = None
        slippage_pct = None
        if reference is not None:
            slippage = max(0.0, fill.premium - reference) if side == "BUY" else max(0.0, reference - fill.premium)
            slippage_pct = slippage / reference if reference > 0 else None
        breaches: list[str] = []
        if latency > self.settings.max_acceptable_fill_latency_s:
            breaches.append(
                f"fill latency {latency:.1f}s exceeds {self.settings.max_acceptable_fill_latency_s:g}s"
            )
        if slippage is not None:
            allowed = min(
                self.settings.max_execution_slippage_cents / 100,
                self.settings.max_execution_slippage_pct * reference,
            )
            if slippage > allowed:
                breaches.append(f"adverse slippage {slippage:.2f} exceeds {allowed:.2f}")
        if side == "BUY" and fill.premium * fill.qty * 100 > self.settings.max_premium_usd:
            breaches.append(
                f"filled cost ${fill.premium * fill.qty * 100:.0f} exceeds "
                f"${self.settings.max_premium_usd:.0f} budget"
            )
        quality = ExecutionQuality(
            side=side,
            order_id=int(getattr(trade.order, "orderId", 0) or 0) or None,
            perm_id=int(getattr(trade.order, "permId", 0) or 0) or None,
            status=trade.orderStatus.status,
            requested_qty=requested,
            filled_qty=fill.qty,
            submitted_at=submitted_at,
            filled_at=filled_at,
            quote=quote,
            underlying_submit=underlying_submit,
            underlying_fill=underlying_fill,
            fill_price=fill.premium,
            latency_s=round(latency, 3),
            reference_price=reference,
            slippage=round(slippage, 4) if slippage is not None else None,
            slippage_pct=round(slippage_pct, 6) if slippage_pct is not None else None,
            breaches=breaches,
        )
        fill.execution_quality = quality
        if self.journal is not None:
            self.journal.write(
                "execution_quality", symbol=contract.symbol,
                contract=contract, execution_quality=quality,
            )
        if breaches:
            self._halt_new_entries(
                f"execution quality breach on {side} {fill.qty}x {contract.local_name}: "
                + "; ".join(breaches)
            )
        return fill

    def _resolve_unfilled(
        self, trade, opt: Option, contract: SelectedContract, side: str, qty: int,
        qty_before: int | None,
    ) -> Fill:
        """Terminal non-Filled order. orderStatus.filled alone cannot be trusted:
        a cancel can race the fill, so IB reports Cancelled (filled 0) for an
        order that actually executed. Reconcile against execution reports and
        the account position, adopt whatever really filled, and halt new
        entries when the fill is partial or unconfirmed."""
        grace = time.monotonic() + self.settings.fill_grace_s  # let late execution reports land
        while time.monotonic() < grace:
            self.ib.waitOnUpdate(timeout=1.0)
        status = trade.orderStatus.status
        reported = int(trade.orderStatus.filled or 0)
        from_fills = int(sum(f.execution.shares for f in trade.fills))
        delta = self._position_delta(opt, side, qty_before)
        confirmed = delta is not None and delta == from_fills
        filled = from_fills if confirmed else max(reported, from_fills, delta or 0)
        label = f"{side} {qty}x {contract.local_name} ended {status} (filled {filled}/{qty})"
        premium = self._fill_premium(trade, filled)
        if filled and premium is not None:
            full_confirmed_fill = filled == qty and confirmed
            if not full_confirmed_fill:
                self._halt_new_entries(
                    f"{label} — the {filled} filled contract(s) WERE adopted into the "
                    "session and will be managed normally"
                )
            return Fill(premium=premium, qty=filled, ts=self.now())
        if filled == 0 and confirmed:
            if side == "BUY":
                self._halt_new_entries(f"{label} — nothing filled, but entry orders are failing")
            raise OrderFailed(
                f"{label}. No contracts filled.",
                side=side, requested=qty, filled=0, suspect=False,
            )
        self._halt_new_entries(f"{label} — TRUE FILL COUNT UNCONFIRMED")
        raise OrderFailed(
            f"{label}. True fill count unconfirmed — reconcile the position in IB Gateway.",
            side=side, requested=qty, filled=filled, suspect=True,
        )

    def _snapshot_position(self, opt: Option) -> int | None:
        """Account position for the contract before an order, or None if unreadable."""
        if not opt.conId:
            return None
        try:
            return self._position_qty(opt.conId)
        except Exception:  # noqa: BLE001 — a failed snapshot only weakens reconciliation
            log.exception("could not snapshot position for conId %s", opt.conId)
            return None

    def _position_qty(self, con_id: int) -> int:
        for pos in self.ib.positions():
            if pos.contract.conId == con_id:
                return int(pos.position or 0)
        return 0

    def _position_delta(self, opt: Option, side: str, qty_before: int | None) -> int | None:
        """Contracts gained (BUY) or shed (SELL) per the account's position since
        the pre-order snapshot, or None when the position could not be read."""
        if qty_before is None or not opt.conId:
            return None
        try:
            self.ib.reqPositions()  # refresh — the cancel may have raced the fill
            after = self._position_qty(opt.conId)
        except Exception:  # noqa: BLE001 — fall back to execution reports only
            log.exception("could not refresh positions to reconcile conId %s", opt.conId)
            return None
        sign = 1 if side == "BUY" else -1
        return max(0, sign * (after - qty_before))

    @staticmethod
    def _fill_premium(trade, filled: int) -> float | None:
        """Average per-contract price of what filled, from execution reports
        first (survives the cancel/fill race), else the order's avgFillPrice."""
        if filled <= 0:
            return None
        shares = sum(f.execution.shares for f in trade.fills)
        if shares:
            total = sum(f.execution.shares * f.execution.price for f in trade.fills)
            return float(total / shares)
        avg = trade.orderStatus.avgFillPrice
        return float(avg) if avg else None

    # -- broker-side protective stop ------------------------------------------

    def place_protective_stop(
        self,
        contract: SelectedContract,
        qty: int,
        stop_price: float,
        direction: Direction,
        order_ref: str,
    ) -> ProtectiveStop:
        opt = self._option(contract)
        under = self._underlying(contract.symbol)
        order = MarketOrder("SELL", qty)
        order.tif = "GTC"  # must survive overnight and the Gateway's restart
        order.orderRef = order_ref
        # The stop is on the UNDERLYING price (the plan's stop is an equity
        # price): calls stop out when the stock falls through it, puts when
        # it rises through it.
        order.conditions = [
            PriceCondition(
                price=stop_price, conId=under.conId, exch="SMART",
                isMore=(direction == "put"),
            )
        ]
        order.conditionsIgnoreRth = False  # a weird pre-market print must not trigger it
        trade = self.ib.placeOrder(opt, order)
        deadline = time.monotonic() + STOP_ACK_TIMEOUT_S
        while time.monotonic() < deadline:
            status = trade.orderStatus.status
            if status in ("PreSubmitted", "Submitted", "Filled"):
                break
            if status in ("Cancelled", "ApiCancelled", "Inactive"):
                self._journal_order_timeline(trade, f"protective stop for {contract.local_name} rejected")
                raise RuntimeError(
                    f"IB rejected the protective stop for {qty}x {contract.local_name}: {status}"
                )
            self.ib.waitOnUpdate(timeout=1.0)
        # An unacknowledged order may still be live at IB — record it rather
        # than raise, so it is never left working untracked.
        log.info(
            "protective stop placed: SELL %dx %s if %s %s %.2f (order %s, ref %s)",
            qty, contract.local_name, contract.symbol,
            ">=" if direction == "put" else "<=", stop_price,
            trade.order.orderId, order_ref,
        )
        return ProtectiveStop(
            order_id=int(trade.order.orderId),
            perm_id=int(trade.order.permId) if trade.order.permId else None,
            order_ref=order_ref,
            qty=qty,
            stop_price=stop_price,
        )

    def cancel_protective_stop(
        self,
        contract: SelectedContract,
        stop: ProtectiveStop,
        expected_held: int | None = None,
    ) -> StopCancelResult:
        opt = self._option(contract)
        trade = self._find_stop_trade(stop)
        if trade is not None and not trade.isDone():
            self.ib.cancelOrder(trade.order)
            deadline = time.monotonic() + CANCEL_TIMEOUT_S
            while not trade.isDone() and time.monotonic() < deadline:
                self.ib.waitOnUpdate(timeout=1.0)
            if not trade.isDone():
                # The stop may still execute — the caller must NOT sell.
                self._halt_new_entries(
                    f"protective stop {stop.order_id} for {contract.local_name} "
                    "cancel never confirmed — reconcile in IB Gateway"
                )
                raise OrderFailed(
                    f"cancel of protective stop {stop.order_id} for {contract.local_name} "
                    "never reached a terminal state.",
                    side="SELL", requested=stop.qty, filled=0, suspect=True,
                )
            grace = time.monotonic() + self.settings.fill_grace_s
            while time.monotonic() < grace:  # a cancel can race the stop's fill
                self.ib.waitOnUpdate(timeout=1.0)
        if trade is not None:
            self._journal_order_timeline(trade, f"protective stop cancel for {contract.local_name}")
            filled = int(sum(f.execution.shares for f in trade.fills))
            avg = self._fill_premium(trade, filled)
        else:
            filled, avg = self._stop_executions(stop)
        # Cross-check the account: it should hold exactly what the caller
        # tracked, minus what the stop sold. Anything else means fills this
        # session cannot see.
        held = self._snapshot_position(opt) if expected_held is not None else None
        if held is not None and held != expected_held - filled:
            self._halt_new_entries(
                f"protective stop {stop.order_id} for {contract.local_name}: account holds "
                f"{held} but reports explain {expected_held - filled} — reconcile in IB Gateway"
            )
            raise OrderFailed(
                f"protective stop {stop.order_id} for {contract.local_name}: position/execution "
                "mismatch after cancel — true fill count unconfirmed.",
                side="SELL", requested=stop.qty, filled=filled, suspect=True,
            )
        if filled and avg is None:
            self._halt_new_entries(
                f"protective stop {stop.order_id} for {contract.local_name} filled {filled} "
                "with no priced executions — reconcile in IB Gateway"
            )
            raise OrderFailed(
                f"protective stop {stop.order_id} filled {filled}x {contract.local_name} "
                "but no execution prices arrived.",
                side="SELL", requested=stop.qty, filled=filled, suspect=True,
            )
        return StopCancelResult(cancelled=True, filled_qty=filled, avg_price=avg)

    def poll_protective_stop(
        self, contract: SelectedContract, stop: ProtectiveStop
    ) -> StopStatus:
        trade = self._find_stop_trade(stop)
        if trade is None:
            filled, avg = self._stop_executions(stop)
            if filled >= stop.qty:
                return StopStatus(state="filled", filled_qty=filled, avg_price=avg)
            if filled:
                return StopStatus(state="partial", filled_qty=filled, avg_price=avg)
            return StopStatus(state="gone")
        filled = int(sum(f.execution.shares for f in trade.fills)) or int(
            trade.orderStatus.filled or 0
        )
        avg = self._fill_premium(trade, filled)
        if trade.orderStatus.status == "Filled" or filled >= stop.qty:
            self._journal_order_timeline(trade, f"protective stop for {contract.local_name} FILLED")
            return StopStatus(state="filled", filled_qty=max(filled, stop.qty), avg_price=avg)
        if filled:
            working = 0 if trade.isDone() else stop.qty - filled
            return StopStatus(state="partial", filled_qty=filled, avg_price=avg, working_qty=working)
        if trade.isDone():  # cancelled externally, nothing executed
            return StopStatus(state="gone")
        return StopStatus(state="working", working_qty=stop.qty)

    def _find_stop_trade(self, stop: ProtectiveStop):
        """The live Trade for a protective stop, or None. reqAllOpenOrders
        re-binds GTC orders after a restart, so persisted stops stay visible."""
        self.ib.reqAllOpenOrders()
        for trade in self.ib.trades():
            o = trade.order
            if stop.perm_id and getattr(o, "permId", 0) == stop.perm_id:
                return trade
            if int(o.orderId or 0) == stop.order_id and (o.orderRef or "") == stop.order_ref:
                return trade
        return None

    def _stop_executions(self, stop: ProtectiveStop) -> tuple[int, float | None]:
        """(shares, avg price) executed under a stop order this session, for
        the case where the order object itself is no longer visible."""
        total, cost = 0, 0.0
        for f in self.ib.fills():
            ex = f.execution
            if (stop.perm_id and ex.permId == stop.perm_id) or ex.orderId == stop.order_id:
                total += int(ex.shares)
                cost += ex.shares * ex.price
        return total, (cost / total if total else None)

    def _halt_new_entries(self, reason: str) -> None:
        log.error("%s — activating kill switch %s", reason, self.settings.kill_switch_file)
        text = (
            f"{reason} at {self.now().isoformat()}\n"
            "Reconcile the position in IB Gateway, then delete this file to resume entries.\n"
        )
        self.settings.kill_switch_file.write_text(text)
        self.notifier.notify_status(
            f"tajator KILL switch activated at {self.settings.kill_switch_file}:\n{text.strip()}"
        )


def _bar_date(d) -> object:
    return d.date() if isinstance(d, datetime) else d
