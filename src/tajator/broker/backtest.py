"""Backtest broker: real historical option fills only — no synthetic pricing.

Subclasses StubBroker (same cursor/bar-serving/chain logic) and overrides only
option pricing: real historical 1-min option bars, fetched (and cached) via IB.
If IB has no historical data for a specific contract/day (illiquid strike, or
an expired contract that fails to qualify), pricing fails loudly instead of
silently substituting a synthetic estimate — mixing real and guessed prices
into the same backtest would make its PnL numbers meaningless.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from ..backtest.data import ET, ensure_option_bars
from ..models import Bar, SelectedContract
from .base import ChainParams, Fill
from .stub import StubBroker

# A fill priced from an option bar further than this from the decision time is
# treated as no data: better to abort than to fill off a stale/illiquid print.
MAX_FILL_STALENESS = timedelta(minutes=5)


class BacktestBroker(StubBroker):
    def __init__(
        self,
        bars: list[Bar],
        prev_day_high: float | None = None,
        prev_day_low: float | None = None,
        daily_bars: list[Bar] | None = None,
        *,
        ib=None,
        cache_dir: Path | None = None,
        chain_override: ChainParams | None = None,
        half_spread_pct: float = 0.0,
        slippage_cents: float = 0.0,
        commission_per_contract: float = 0.0,
        min_commission_per_order: float = 0.0,
    ):
        super().__init__(bars, prev_day_high, prev_day_low, daily_bars=daily_bars)
        self._ib = ib
        self._cache_dir = cache_dir
        self._chain_override = chain_override
        self._option_series: dict[tuple, list[Bar] | None] = {}
        self.half_spread_pct = half_spread_pct
        self.slippage_cents = slippage_cents
        self.commission_per_contract = commission_per_contract
        self.min_commission_per_order = min_commission_per_order

    def get_option_chain(self, symbol: str) -> ChainParams:
        """Real listed strikes from IB, but expirations generated mechanically
        for the *backtest* day.

        StubBroker's default chain uses a half-dollar strike grid; for a symbol
        like SPY (whole-dollar strikes) that makes the selector pick contracts
        that were never listed, so every fill fails to price and the backtest
        aborts. The real strike set comes from IB's chain (cached by day inside
        IBBroker). Expirations still have to be synthesized — today's chain no
        longer lists the historical expirations a past session traded — so we
        keep StubBroker's near-Friday logic anchored on the backtest day."""
        if self._chain_override is not None:
            return self._chain_override
        if self._ib is None:
            return super().get_option_chain(symbol)
        strikes = self._ib.get_option_chain(symbol).strikes or self._chain.strikes
        day = self.now().astimezone(ET).date()
        friday = day.toordinal() + (4 - day.weekday()) % 7
        this_friday = datetime.fromordinal(friday).strftime("%Y%m%d")
        next_friday = datetime.fromordinal(friday + 7).strftime("%Y%m%d")
        return ChainParams(expirations=[this_friday, next_friday], strikes=sorted(strikes))

    def _series_for(self, contract: SelectedContract) -> list[Bar] | None:
        day = self.now().astimezone(ET).date()
        key = (contract.symbol, contract.expiry, contract.strike, contract.right, day)
        if key not in self._option_series:
            self._option_series[key] = ensure_option_bars(self._ib, contract, day, self._cache_dir)
        return self._option_series[key]

    @staticmethod
    def _fill_price(series: list[Bar], ts: datetime) -> float | None:
        """Price a fill decided at `ts` without look-ahead.

        The decision was made on the close of the bar ending at `ts`, so the
        order can only fill afterwards: use the NEXT option bar's open. At the
        end of the series (e.g. the day's forced flatten) fall back to the
        decision bar's close. Anything further than MAX_FILL_STALENESS from
        `ts` counts as no data."""
        prev = None
        for bar in series:
            if bar.ts > ts:
                if bar.ts - ts <= MAX_FILL_STALENESS:
                    return bar.open
                break
            prev = bar
        if prev is not None and ts - prev.ts <= MAX_FILL_STALENESS:
            return prev.close
        return None

    def get_option_premium(self, contract: SelectedContract) -> float | None:
        series = self._series_for(contract)
        price = self._fill_price(series, self.now()) if series else None
        if price is None:
            day = self.now().astimezone(ET).date()
            raise RuntimeError(
                f"no historical option data for {contract.local_name} on {day} — "
                "cannot price this fill from real data, aborting backtest"
            )
        return round(price, 2)

    def _execution_fill(self, contract: SelectedContract, qty: int, side: str) -> Fill:
        reference = self.get_option_premium(contract)
        adverse = reference * self.half_spread_pct + self.slippage_cents / 100
        premium = reference + adverse if side == "BUY" else max(0.01, reference - adverse)
        fee = max(self.min_commission_per_order, qty * self.commission_per_contract)
        fill = Fill(premium=round(premium, 2), qty=qty, ts=self.now(), fee=round(fee, 2))
        self.fills.append((side, contract, fill))
        return fill

    def _counterfactual_fill(
        self, contract: SelectedContract, side: str, template: Fill
    ) -> Fill:
        """Price another contract at the base fill's exact decision timestamp."""
        series = self._series_for(contract)
        reference = self._fill_price(series, template.ts) if series else None
        if reference is None:
            day = self.now().astimezone(ET).date()
            raise RuntimeError(
                f"no historical option data for {contract.local_name} on {day} at "
                f"{template.ts.isoformat()}"
            )
        adverse = reference * self.half_spread_pct + self.slippage_cents / 100
        premium = reference + adverse if side == "BUY" else max(0.01, reference - adverse)
        fee = max(self.min_commission_per_order, template.qty * self.commission_per_contract)
        return template.model_copy(update={"premium": round(premium, 2), "fee": round(fee, 2)})

    @staticmethod
    def _panel_contracts(
        base: SelectedContract, chain: ChainParams
    ) -> dict[str, SelectedContract | None]:
        strikes = sorted(set(chain.strikes))
        expirations = sorted(set(chain.expirations))
        if not strikes:
            return {"itm_1_near": None, "otm_1_near": None, "atm_next_expiry": None}
        index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - base.strike))
        itm_index = index - 1 if base.right == "C" else index + 1
        otm_index = index + 1 if base.right == "C" else index - 1
        later = [expiry for expiry in expirations if expiry > base.expiry]

        def at_strike(candidate_index: int) -> SelectedContract | None:
            if not 0 <= candidate_index < len(strikes):
                return None
            return base.model_copy(update={"strike": strikes[candidate_index], "con_id": None})

        return {
            "itm_1_near": at_strike(itm_index),
            "otm_1_near": at_strike(otm_index),
            "atm_next_expiry": (
                base.model_copy(update={"expiry": later[0], "con_id": None}) if later else None
            ),
        }

    def reprice_option_panel(
        self,
        fills: list[tuple[str, SelectedContract, Fill]],
        chain: ChainParams,
    ) -> tuple[dict[str, list[tuple[str, SelectedContract, Fill]]], dict[str, list[dict]]]:
        """Reprice fixed contract variants without changing signals, times, or quantities."""
        variants = {name: [] for name in ("itm_1_near", "otm_1_near", "atm_next_expiry")}
        missing = {name: [] for name in variants}
        grouped: dict[tuple, list[tuple[str, SelectedContract, Fill]]] = {}
        for item in fills:
            _, contract, _ = item
            key = (contract.symbol, contract.expiry, contract.strike, contract.right)
            grouped.setdefault(key, []).append(item)

        for group in grouped.values():
            base = group[0][1]
            contracts = self._panel_contracts(base, chain)
            for name, alternative in contracts.items():
                if alternative is None:
                    missing[name].append({
                        "base_contract": base.local_name,
                        "reason": "no listed adjacent strike or later expiration",
                    })
                    continue
                try:
                    repriced = [
                        (side, alternative, self._counterfactual_fill(alternative, side, fill))
                        for side, _, fill in group
                    ]
                except RuntimeError as exc:
                    missing[name].append({
                        "base_contract": base.local_name,
                        "alternative_contract": alternative.local_name,
                        "reason": str(exc),
                    })
                    continue
                variants[name].extend(repriced)
        return variants, missing

    def buy_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._execution_fill(contract, qty, "BUY")

    def sell_option(self, contract: SelectedContract, qty: int) -> Fill:
        return self._execution_fill(contract, qty, "SELL")
