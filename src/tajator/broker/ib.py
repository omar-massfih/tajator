"""Interactive Brokers via ib_async (maintained fork of ib_insync).

Requires IB Gateway running with the API enabled
(Configure → API → Settings → Enable ActiveX and Socket Clients).
Port 4002 = IB Gateway paper, 4001 = IB Gateway live.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from zoneinfo import ZoneInfo

from ib_async import IB, MarketOrder, Option, Stock

from ..config import Settings
from ..models import Bar, SelectedContract
from .base import Broker, ChainParams, Fill

ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)

NO_SUBSCRIPTION_CODES = {354, 10167, 10197}
ORDER_TIMEOUT_S = 30


class IBBroker(Broker):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ib = IB()
        self._stock: Stock | None = None
        self._chain_cache: tuple[str, ChainParams] | None = None
        self._qualified: dict[str, Option] = {}
        self._delayed = False

    # -- lifecycle ---------------------------------------------------------------

    def connect(self) -> None:
        s = self.settings
        self.ib.connect(s.ib_host, s.ib_port, clientId=s.ib_client_id, timeout=10)
        self.ib.reqMarketDataType(s.market_data_type)
        self.ib.errorEvent += self._on_error
        log.info("connected to IB %s:%s (mode=%s)", s.ib_host, s.ib_port, s.trading_mode)

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

    def _underlying(self, symbol: str) -> Stock:
        if self._stock is None or self._stock.symbol != symbol:
            stock = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(stock)
            self._stock = stock
        return self._stock

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
        if self._chain_cache and self._chain_cache[0] == f"{symbol}:{today}":
            return self._chain_cache[1]
        stock = self._underlying(symbol)
        params = self.ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
        smart = next((p for p in params if p.exchange == "SMART"), params[0] if params else None)
        if smart is None:
            return ChainParams(expirations=[], strikes=[])
        chain = ChainParams(
            expirations=sorted(smart.expirations), strikes=sorted(smart.strikes)
        )
        self._chain_cache = (f"{symbol}:{today}", chain)
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
        return self._qualified[key]

    def get_option_premium(self, contract: SelectedContract) -> float | None:
        opt = self._option(contract)
        [ticker] = self.ib.reqTickers(opt)
        price = ticker.marketPrice()
        if price is None or math.isnan(price) or price <= 0:
            price = ticker.close
        if price is None or math.isnan(price) or price <= 0:
            return None
        return float(price)

    def buy_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._place(contract, "BUY", qty)

    def sell_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._place(contract, "SELL", qty)

    def _place(self, contract: SelectedContract, side: str, qty: int) -> Fill:
        opt = self._option(contract)
        trade = self.ib.placeOrder(opt, MarketOrder(side, qty))
        waited = 0.0
        while not trade.isDone() and waited < ORDER_TIMEOUT_S:
            self.ib.waitOnUpdate(timeout=1.0)
            waited += 1.0
        if not trade.isDone():
            raise RuntimeError(
                f"{side} {qty}x {contract.local_name} not filled within {ORDER_TIMEOUT_S}s "
                f"(status={trade.orderStatus.status}) — check IB Gateway"
            )
        if trade.orderStatus.status != "Filled":
            raise RuntimeError(
                f"{side} {qty}x {contract.local_name} ended {trade.orderStatus.status}"
            )
        return Fill(
            premium=float(trade.orderStatus.avgFillPrice), qty=qty, ts=self.now()
        )


def _bar_date(d) -> object:
    return d.date() if isinstance(d, datetime) else d
