"""Settings loaded from environment / .env. Paper trading is the hard default."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .risk.guardrails import STOP_COOLDOWN_MINUTES, STOP_MAX_CENTS, STOP_MIN_CENTS

AGENT_DIR = Path(__file__).resolve().parents[2]
LIVE_PORTS = {4001, 7496}  # 4001 = IB Gateway live, 7496 = TWS live
PAPER_PORTS = {4002, 7497}  # 4002 = IB Gateway paper, 7497 = TWS paper


class SymbolStrategyOverride(BaseModel):
    """Per-symbol overrides for the small set of knobs that vary by name."""

    no_new_entries_before: time | None = None
    no_new_entries_after: time | None = None


def _default_symbol_strategy_overrides() -> dict[str, SymbolStrategyOverride]:
    """No per-symbol overrides by default — the ORB strategy is symbol-agnostic."""
    return {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Interactive Brokers
    ib_host: str = "127.0.0.1"
    ib_port: int = 4002
    ib_client_id: int = 17
    trading_mode: Literal["paper", "live"] = "paper"
    market_data_type: int = 1
    # Paper-sim fills can take minutes without live option data (see the
    # 2026-07-08 incident) — a working market order is not a failure, and
    # cancelling it is what creates the cancel/fill race.
    order_timeout_s: int = 120
    fill_grace_s: int = 15  # post-cancel window for late execution reports
    block_entries_on_delayed_data: bool = True
    max_option_spread_pct: float = 0.08
    max_option_spread_cents: int = 30
    max_option_quote_age_s: float = 5.0
    entry_budget_reserve_pct: float = 0.05
    max_entry_drift_atr: float = 0.5
    max_entry_drift_min_cents: int = 10
    max_execution_slippage_pct: float = 0.03
    max_execution_slippage_cents: int = 10
    max_acceptable_fill_latency_s: float = 10.0
    execution_diagnostic_max_age_days: int = 7
    execution_live_confirmed: bool = False

    # Strategy — Opening-Range Breakout
    symbols: Annotated[list[str], NoDecode] = ["SPY"]
    orb_window_minutes: int = 15  # opening range = first N minutes after 09:30 ET
    orb_breakout_buffer_pct: float = 0.0005  # close must clear the range by this fraction
    # Entry-quality filters (0 disables). Fewer, higher-conviction breakouts.
    orb_min_relative_volume: float = 0.0  # breakout-bar volume / mean session volume
    orb_min_breakout_range_atr: float = 0.0  # breakout-bar range >= this x ATR
    # Exit style: "scale" (scale out at ema50/vwap then hod/lod) or "let_run"
    # (single position, trail an ATR chandelier stop toward a large R-multiple).
    exit_mode: Literal["scale", "let_run"] = "scale"
    runner_target_r: float = 3.0  # let_run target as a multiple of entry-to-stop risk
    runner_trail_atr_mult: float = 1.5  # let_run trailing-stop distance in ATRs
    max_trades_per_day: int = 2
    max_contracts: int = 10  # raised (aggressive but capped)
    max_premium_usd: float = 2000.0  # raised (aggressive but capped)
    stop_buffer_cents: int = 40  # fallback stop when a candidate carries none
    no_new_entries_after: time = time(15, 30)
    no_new_entries_before: time = time(9, 30)
    atr_window_bars: int = 14
    # Broker-side protective stop: a GTC market sell resting at IB, triggered
    # by the underlying crossing the plan's stop price. Backstop for the
    # in-loop mental stop — protects the position when tajator is down.
    protective_stop_enabled: bool = Field(
        default=False, validation_alias=AliasChoices("PROTECTIVE_STOP", "protective_stop_enabled")
    )
    order_ref_prefix: str = "tajator"  # provenance tag on every order we place
    symbol_strategy_overrides: dict[str, SymbolStrategyOverride] = Field(
        default_factory=_default_symbol_strategy_overrides
    )

    # Stop-distance rule (defaults live next to the gate in risk/guardrails.py)
    stop_min_cents: int = STOP_MIN_CENTS
    stop_max_cents: int = STOP_MAX_CENTS

    # A level that stopped us out is dead for this long; 0 disables the cooldown.
    stop_cooldown_minutes: int = STOP_COOLDOWN_MINUTES

    # Stop protecting the runner after the first scale-out: "breakeven" (entry
    # price — gives the runner room toward hod/lod) or "first_target" (the
    # strategy notes' tighter lock; in backtests it ended every runner within
    # two bars of the scale-out, so breakeven is the default).
    runner_stop: Literal["breakeven", "first_target"] = "breakeven"

    # Conservative backtest execution model. Historical option OHLC bars do
    # not contain a bid/ask pair, so the modeled half-spread and slippage are
    # applied adversely to each side and disclosed in report metadata.
    backtest_half_spread_pct: float = 0.01
    backtest_slippage_cents: float = 1.0
    backtest_commission_per_contract: float = 0.65
    backtest_min_commission_per_order: float = 1.0

    # Telegram trade notifications (optional — leave blank to disable)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Paths
    kill_switch_file: Path = AGENT_DIR / "KILL"
    state_file: Path = AGENT_DIR / "state.json"  # live session state, adopted on restart
    log_dir: Path = AGENT_DIR / "logs"
    backtest_cache_dir: Path = AGENT_DIR / "data" / "historical"

    @field_validator("no_new_entries_after", "no_new_entries_before", mode="before")
    @classmethod
    def _parse_time(cls, v: object) -> object:
        if isinstance(v, str) and ":" in v:
            hh, mm = v.split(":")
            return time(int(hh), int(mm))
        return v

    @field_validator("symbol_strategy_overrides", mode="before")
    @classmethod
    def _uppercase_override_symbols(cls, v: object) -> object:
        if isinstance(v, dict):
            return {str(k).upper(): value for k, value in v.items()}
        return v

    @field_validator("symbols", mode="before")
    @classmethod
    def _parse_symbols(cls, v: object) -> object:
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return v

    @field_validator(
        "backtest_half_spread_pct", "backtest_slippage_cents",
        "backtest_commission_per_contract", "backtest_min_commission_per_order",
        "max_option_spread_pct", "entry_budget_reserve_pct", "max_entry_drift_atr",
        "max_execution_slippage_pct", "max_acceptable_fill_latency_s", "max_option_quote_age_s",
    )
    @classmethod
    def _nonnegative_execution_values(cls, v: float) -> float:
        if v < 0:
            raise ValueError("execution and backtest values cannot be negative")
        return v

    @model_validator(mode="after")
    def _live_requires_live_port(self) -> "Settings":
        # Going live is a deliberate two-field change: TRADING_MODE and IB_PORT
        # must both say "live", otherwise refuse to start.
        if self.trading_mode == "live" and self.ib_port not in LIVE_PORTS:
            raise ValueError(
                "TRADING_MODE=live but IB_PORT is not a live port. "
                "Set IB_PORT=4001 (IB Gateway) or 7496 (TWS) to confirm live trading."
            )
        if self.trading_mode == "paper" and self.ib_port in LIVE_PORTS:
            raise ValueError(
                "TRADING_MODE=paper but IB_PORT is a LIVE port. "
                "Use 4002 (IB Gateway paper) or 7497 (TWS paper)."
            )
        if self.no_new_entries_before >= self.no_new_entries_after:
            raise ValueError("NO_NEW_ENTRIES_BEFORE must be before NO_NEW_ENTRIES_AFTER")
        if self.atr_window_bars < 2:
            raise ValueError("ATR_WINDOW_BARS must be at least 2")
        if self.orb_window_minutes < 1:
            raise ValueError("ORB_WINDOW_MINUTES must be at least 1")
        if self.orb_breakout_buffer_pct < 0:
            raise ValueError("ORB_BREAKOUT_BUFFER_PCT cannot be negative")
        if self.orb_min_relative_volume < 0:
            raise ValueError("ORB_MIN_RELATIVE_VOLUME cannot be negative")
        if self.orb_min_breakout_range_atr < 0:
            raise ValueError("ORB_MIN_BREAKOUT_RANGE_ATR cannot be negative")
        if self.runner_target_r <= 0:
            raise ValueError("RUNNER_TARGET_R must be positive")
        if self.runner_trail_atr_mult <= 0:
            raise ValueError("RUNNER_TRAIL_ATR_MULT must be positive")
        for name in (
            "max_option_spread_cents", "max_entry_drift_min_cents",
            "max_execution_slippage_cents", "execution_diagnostic_max_age_days",
            "stop_min_cents", "stop_max_cents", "max_contracts", "max_trades_per_day",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name.upper()} must be positive")
        if self.stop_min_cents > self.stop_max_cents:
            raise ValueError("STOP_MIN_CENTS must not exceed STOP_MAX_CENTS")
        if self.max_premium_usd <= 0:
            raise ValueError("MAX_PREMIUM_USD must be positive")
        return self

    def for_symbol(self, symbol: str) -> "Settings":
        override = self.symbol_strategy_overrides.get(symbol.upper())
        if override is None:
            return self
        updates = {k: v for k, v in override.model_dump().items() if v is not None}
        resolved = self.model_copy(update=updates)
        if resolved.no_new_entries_before >= resolved.no_new_entries_after:
            raise ValueError(
                f"{symbol.upper()} no_new_entries_before must be before no_new_entries_after"
            )
        return resolved


def load_settings() -> Settings:
    return Settings()
