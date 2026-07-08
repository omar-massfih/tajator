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

from ..config import Settings
from ..models import Bar, SelectedContract
from .base import Broker, BrokerOptionPosition, ChainParams, Fill, OrderFailed

ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)

NO_SUBSCRIPTION_CODES = {354, 10167, 10197}
ORDER_TIMEOUT_S = 30
CANCEL_TIMEOUT_S = 10
FILL_GRACE_S = 5  # after a cancel, wait this long for late execution reports


class IBBroker(Broker):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ib = IB()
        self._stocks: dict[str, Stock] = {}
        self._chain_cache: dict[str, tuple[str, ChainParams]] = {}
        self._qualified: dict[str, Option] = {}
        self._delayed = False
        # subscribe before connect() so connect-time errors are seen too
        self.ib.errorEvent += self._on_error

    # -- lifecycle ---------------------------------------------------------------

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

    def is_connected(self) -> bool:
        return self.ib.isConnected()

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

    def open_option_orders(self, symbols: list[str]) -> list[str]:
        """Resting orders at IB for the given underlyings, as human-readable
        strings. reqAllOpenOrders sees orders from other client IDs and manual
        TWS orders too — e.g. one left working by a crashed session."""
        self.ib.reqAllOpenOrders()
        found = []
        for trade in self.ib.openTrades():
            c = trade.contract
            if c.secType == "OPT" and c.symbol in symbols:
                o = trade.order
                found.append(
                    f"{o.action} {o.totalQuantity:g}x {c.localSymbol or c.symbol} "
                    f"({o.orderType}, status {trade.orderStatus.status})"
                )
        return found

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

    def _option(self, contract: SelectedContract) -> Option:
        key = contract.local_name
        if key not in self._qualified:
            opt = Option(
                contract.symbol, contract.expiry, contract.strike, contract.right,
                "SMART", currency="USD",
            )
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

    def buy_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._place(contract, "BUY", qty)

    def sell_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._place(contract, "SELL", qty)

    def _place(self, contract: SelectedContract, side: str, qty: int) -> Fill:
        opt = self._option(contract)
        qty_before = self._snapshot_position(opt)
        trade = self.ib.placeOrder(opt, MarketOrder(side, qty))
        deadline = time.monotonic() + ORDER_TIMEOUT_S
        while not trade.isDone() and time.monotonic() < deadline:
            self.ib.waitOnUpdate(timeout=1.0)
        if not trade.isDone():
            # Never leave a market order working untracked: cancel, then wait
            # briefly for the terminal status so partial fills are reported.
            self.ib.cancelOrder(trade.order)
            cancel_deadline = time.monotonic() + CANCEL_TIMEOUT_S
            while not trade.isDone() and time.monotonic() < cancel_deadline:
                self.ib.waitOnUpdate(timeout=1.0)
        if trade.orderStatus.status == "Filled":
            return Fill(
                premium=float(trade.orderStatus.avgFillPrice), qty=qty, ts=self.now()
            )
        return self._resolve_unfilled(trade, opt, contract, side, qty, qty_before)

    def _resolve_unfilled(
        self, trade, opt: Option, contract: SelectedContract, side: str, qty: int,
        qty_before: int | None,
    ) -> Fill:
        """Terminal non-Filled order. orderStatus.filled alone cannot be trusted:
        a cancel can race the fill, so IB reports Cancelled (filled 0) for an
        order that actually executed. Reconcile against execution reports and
        the account position, adopt whatever really filled, and halt new
        entries — any non-Filled outcome means the account needs a look."""
        grace = time.monotonic() + FILL_GRACE_S  # let late execution reports land
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

    def _halt_new_entries(self, reason: str) -> None:
        log.error("%s — activating kill switch %s", reason, self.settings.kill_switch_file)
        self.settings.kill_switch_file.write_text(
            f"{reason} at {self.now().isoformat()}\n"
            "Reconcile the position in IB Gateway, then delete this file to resume entries.\n"
        )


def _bar_date(d) -> object:
    return d.date() if isinstance(d, datetime) else d
