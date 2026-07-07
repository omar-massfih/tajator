"""Settings loaded from environment / .env. Paper trading is the hard default."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

AGENT_DIR = Path(__file__).resolve().parents[2]
LIVE_PORTS = {4001}  # IB Gateway live; 4002 = IB Gateway paper


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

    # Strategy
    symbols: Annotated[list[str], NoDecode] = ["SPY"]
    max_trades_per_day: int = 2
    max_contracts: int = 4
    max_premium_usd: float = 500.0
    stop_buffer_cents: int = 40
    no_new_entries_after: time = time(15, 30)

    # Paths
    kill_switch_file: Path = AGENT_DIR / "KILL"
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
                "TRADING_MODE=live but IB_PORT is a paper port. "
                "Set IB_PORT=4001 (IB Gateway live) to confirm live trading."
            )
        if self.trading_mode == "paper" and self.ib_port in LIVE_PORTS:
            raise ValueError(
                "TRADING_MODE=paper but IB_PORT is a LIVE port. Use 4002 (IB Gateway paper)."
            )
        return self


def load_settings() -> Settings:
    return Settings()
