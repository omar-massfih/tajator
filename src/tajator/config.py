"""Settings loaded from environment / .env. Paper trading is the hard default."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .market.levels import (
    CLUSTER_TOL,
    DOUBLE_MIN_PULLBACK_PCT,
    DOUBLE_MIN_TOUCH_SEPARATION_BARS,
    SWING_WINDOW,
)
from .market.setups import (
    APPROACH_BAND,
    ENTRY_CONFIRMATION,
    FAST_APPROACH_SPEED_MULT,
    MIN_LEVEL_DIST_FROM_OPEN_PCT,
    MIN_SPEED_PCT,
    OVERSHOOT_BAND,
    REJECTION_WICK_MIN_FRAC,
    SPEED_WINDOW,
    TRADE_FLIPPED_LEVELS,
)
from .market.price_action import LONG_WICK_MIN_FRAC, REACTION_LOOKBACK_BARS
from .risk.guardrails import STOP_COOLDOWN_MINUTES, STOP_MAX_CENTS, STOP_MIN_CENTS

AGENT_DIR = Path(__file__).resolve().parents[2]
LIVE_PORTS = {4001, 7496}  # 4001 = IB Gateway live, 7496 = TWS live
PAPER_PORTS = {4002, 7497}  # 4002 = IB Gateway paper, 7497 = TWS paper
REGIMES = {"unknown", "range", "trend_up", "trend_down", "high_volatility"}


class SymbolStrategyOverride(BaseModel):
    multi_timeframe_context: bool | None = None
    entry_confirmation: Literal["immediate", "touch_rejection"] | None = None
    max_entry_to_stop_cents: int | None = None
    no_new_entries_before: time | None = None
    no_new_entries_after: time | None = None
    opening_confirmation_until: time | None = None
    stop_atr_multiplier: float | None = None
    allowed_regimes: list[str] | None = None
    blocked_direction_regimes: list[str] | None = None
    min_level_quality_score: float | None = None
    reaction_lookback_bars: int | None = None
    long_wick_min_frac: float | None = None

    @field_validator("blocked_direction_regimes")
    @classmethod
    def _valid_direction_regimes(cls, values):
        _validate_direction_regimes(values or [])
        return values

    @model_validator(mode="after")
    def _valid_values(self):
        for name in ("max_entry_to_stop_cents", "stop_atr_multiplier", "min_level_quality_score"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.reaction_lookback_bars is not None and self.reaction_lookback_bars < 2:
            raise ValueError("reaction_lookback_bars must be at least 2")
        if self.long_wick_min_frac is not None and not 0 <= self.long_wick_min_frac <= 1:
            raise ValueError("long_wick_min_frac must be between 0 and 1")
        return self


def _validate_direction_regimes(values: list[str]) -> None:
    valid = {f"{direction}:{regime}" for direction in ("call", "put") for regime in REGIMES}
    invalid = sorted(set(values) - valid)
    if invalid:
        raise ValueError(f"invalid direction/regime blocks: {', '.join(invalid)}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_model: str = "openai:gpt-5.1"

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

    # Strategy
    symbols: Annotated[list[str], NoDecode] = ["SPY"]
    multi_timeframe_context: bool = False
    max_trades_per_day: int = 2
    max_contracts: int = 4
    max_premium_usd: float = 500.0
    stop_buffer_cents: int = 40
    no_new_entries_after: time = time(15, 30)
    no_new_entries_before: time = time(9, 30)
    # Broker-side protective stop: a GTC market sell resting at IB, triggered
    # by the underlying crossing the plan's stop price. Backstop for the
    # in-loop mental stop — protects the position when tajator is down.
    protective_stop_enabled: bool = Field(
        default=False, validation_alias=AliasChoices("PROTECTIVE_STOP", "protective_stop_enabled")
    )
    order_ref_prefix: str = "tajator"  # provenance tag on every order we place

    # Level quality gates (defaults live next to the algorithms in market/)
    double_min_touch_separation_bars: int = DOUBLE_MIN_TOUCH_SEPARATION_BARS
    double_min_pullback_pct: float = DOUBLE_MIN_PULLBACK_PCT
    min_level_dist_from_open_pct: float = MIN_LEVEL_DIST_FROM_OPEN_PCT
    swing_window_bars: int = SWING_WINDOW
    level_cluster_tol_pct: float = CLUSTER_TOL

    # Setup detection tuning (defaults live next to the algorithm in market/setups.py)
    approach_band_pct: float = APPROACH_BAND
    overshoot_band_pct: float = OVERSHOOT_BAND
    speed_window_bars: int = SPEED_WINDOW
    min_speed_pct: float = MIN_SPEED_PCT
    fast_approach_speed_mult: float = FAST_APPROACH_SPEED_MULT
    rejection_wick_min_frac: float = REJECTION_WICK_MIN_FRAC
    reaction_lookback_bars: int = REACTION_LOOKBACK_BARS
    long_wick_min_frac: float = LONG_WICK_MIN_FRAC
    # Role-reversed levels (a broken support retested as resistance, and vice
    # versa) are chart context, not trades — the worst entry class in backtests.
    trade_flipped_levels: bool = TRADE_FLIPPED_LEVELS
    entry_confirmation: Literal["immediate", "touch_rejection"] = ENTRY_CONFIRMATION
    opening_confirmation_until: time | None = None
    max_entry_to_stop_cents: int | None = None
    stop_atr_multiplier: float | None = None
    atr_window_bars: int = 14
    allowed_regimes: list[str] = Field(default_factory=list)
    blocked_direction_regimes: list[str] = Field(default_factory=list)
    min_level_quality_score: float | None = None
    symbol_strategy_overrides: dict[str, SymbolStrategyOverride] = Field(default_factory=dict)

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

    @field_validator("no_new_entries_after", "no_new_entries_before", "opening_confirmation_until", mode="before")
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

    @field_validator("blocked_direction_regimes")
    @classmethod
    def _valid_global_direction_regimes(cls, values: list[str]) -> list[str]:
        _validate_direction_regimes(values)
        return values

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
        for name in ("max_entry_to_stop_cents", "stop_atr_multiplier", "min_level_quality_score"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.atr_window_bars < 2:
            raise ValueError("ATR_WINDOW_BARS must be at least 2")
        if self.reaction_lookback_bars < 2:
            raise ValueError("REACTION_LOOKBACK_BARS must be at least 2")
        if not 0 <= self.long_wick_min_frac <= 1:
            raise ValueError("LONG_WICK_MIN_FRAC must be between 0 and 1")
        for name in (
            "max_option_spread_cents", "max_entry_drift_min_cents",
            "max_execution_slippage_cents", "execution_diagnostic_max_age_days",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name.upper()} must be positive")
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
