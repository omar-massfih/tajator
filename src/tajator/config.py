"""Settings loaded from environment / .env. Paper trading is the hard default."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .market.levels import (
    CLUSTER_TOL,
    DOUBLE_MIN_PULLBACK_PCT,
    DOUBLE_MIN_TOUCH_SEPARATION_BARS,
    SWING_WINDOW,
)
from .market.setups import (
    APPROACH_BAND,
    MIN_LEVEL_DIST_FROM_OPEN_PCT,
    MIN_SPEED_PCT,
    OVERSHOOT_BAND,
    SPEED_WINDOW,
)
from .risk.guardrails import STOP_MAX_CENTS, STOP_MIN_CENTS

AGENT_DIR = Path(__file__).resolve().parents[2]
LIVE_PORTS = {4001, 7496}  # 4001 = IB Gateway live, 7496 = TWS live
PAPER_PORTS = {4002, 7497}  # 4002 = IB Gateway paper, 7497 = TWS paper


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

    # Strategy
    symbols: Annotated[list[str], NoDecode] = ["SPY"]
    max_trades_per_day: int = 2
    max_contracts: int = 4
    max_premium_usd: float = 500.0
    stop_buffer_cents: int = 40
    no_new_entries_after: time = time(15, 30)
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

    # Stop-distance rule (defaults live next to the gate in risk/guardrails.py)
    stop_min_cents: int = STOP_MIN_CENTS
    stop_max_cents: int = STOP_MAX_CENTS

    # Telegram trade notifications (optional — leave blank to disable)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Paths
    kill_switch_file: Path = AGENT_DIR / "KILL"
    state_file: Path = AGENT_DIR / "state.json"  # live session state, adopted on restart
    log_dir: Path = AGENT_DIR / "logs"
    backtest_cache_dir: Path = AGENT_DIR / "data" / "historical"

    @field_validator("no_new_entries_after", mode="before")
    @classmethod
    def _parse_time(cls, v: object) -> object:
        if isinstance(v, str) and ":" in v:
            hh, mm = v.split(":")
            return time(int(hh), int(mm))
        return v

    @field_validator("symbols", mode="before")
    @classmethod
    def _parse_symbols(cls, v: object) -> object:
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
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
        return self


def load_settings() -> Settings:
    return Settings()
