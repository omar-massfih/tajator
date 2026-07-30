# Tajator

> **Name origin:** *Tajator* is shaped from Arabic *tājir* (تاجر, "trader") and
> Latin *mercator* ("merchant"), carrying the idea of trade, markets, and craft.

A deterministic intraday options day-trader built with **LangGraph**, executing a
classic **Opening-Range Breakout (ORB)** strategy through **Interactive Brokers**
(paper account by default).

The strategy in one line: after the first `ORB_WINDOW_MINUTES` of the regular
session, lock the opening range `[or_low, or_high]`. When a completed 1-minute bar
**closes above `or_high`** (plus a small buffer), buy **calls**; when it **closes
below `or_low`**, buy **puts**. The stop is the opposite side of the range; the
position scales out in pieces and rides a runner toward the high/low of day, and is
force-flattened before the close (`RUNNER_STOP=first_target` restores the tighter
lock at the first target). This is a *momentum* strategy — it enters *with* the
move, above/below the level, not fading into it.

> **This is an experimental system for paper trading. Options trading involves
> substantial risk of loss, and an all-or-nothing sizing policy maximizes the
> probability of ruin. Do not point it at real money you cannot afford to lose.**

## Design: deterministic, one path

```
fetch_data → compute_context ─┬─ (position open) → manage_position → stop/runner exit | scale out
                              └─ (flat) → detect_setups → decide → risk_gate → enter
```

- **`detect_setups`** computes the opening range and emits a breakout candidate
  (`market/orb.py`). No breakout → nothing happens.
- **`risk_gate`** (`risk/guardrails.py`) is a non-negotiable veto layer: market
  hours, `MAX_TRADES_PER_DAY`, one position at a time, a stop on the correct side
  within the configured cent band, and a kill-switch file.
- **Sizing** caps each entry by `MAX_PREMIUM_USD` and `MAX_CONTRACTS` (raised but
  finite — a single fill can't zero the account).
- **Stops** (`trade/position.py`) are enforced in code on every tick: the mental
  stop at the opposite side of the range, break-even protection after the first
  scale-out, and a VWAP runner exit.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically):

```bash
uv sync
cp .env.example .env   # then fill it in
```

**IBKR:** install [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php),
log into your **paper** account, and enable the API (Configure → API → Settings →
*Enable ActiveX and Socket Clients*; port 4002 for paper, or 7497 for TWS paper).

## Commands

```bash
uv run tajator check-ib     # connectivity check: bars, chain, quote — places NO orders
uv run tajator test-order   # paper diagnostic: buy 1 lot, watch the fill timeline, sell it back
uv run tajator test-order --with-stop   # + place/verify/cancel a protective stop mid-trade
uv run tajator replay --csv tests/data/spy_sample_day.csv --symbol SPY \
    --prev-high 503.5 --prev-low 497.0   # bundled ORB day, no IB needed
uv run tajator replay --date 2026-07-02          # fetch a real day from IB, replay it
uv run tajator backtest --symbol SPY --start 2026-04-01 --end 2026-06-30
uv run tajator backtest --symbol SPY --start 2026-04-01 --end 2026-06-30 \
    --underlying-only --experiment baseline  # long-window stock-signal research
uv run tajator backtest-compare logs/backtests/*_baseline.json logs/backtests/*_variant.json
uv run tajator run          # live minute loop (paper by default)
uv run pytest               # full test suite
```

Everything is deterministic — there is no LLM in the trade path. `replay` steps the
same graph through a recorded day with instant synthetic option fills (plumbing
validation, not a backtest). `backtest` steps the same graph over a date range,
fetching real underlying bars and, for every fill, the real historical option quote
for that exact contract/day (cached under `data/historical/`); `--underlying-only`
replays the identical detector against stock bars and reports direction-adjusted
underlying points for long windows where IB no longer exposes expired options. Any
position open at a recorded day's close is force-flattened so ledgers count every
trade (live trading never auto-flattens; it warns and journals).

`test-order` is the supervised acceptance check after any execution change. It uses
the same quote validation, budget sizing, market-order timeout, fill reconciliation,
and execution telemetry as production, then immediately sells the confirmed paper
position. Live mode additionally requires a recent pass for every configured symbol
plus `EXECUTION_LIVE_CONFIRMED=true`.

## Configuration

Key `.env` settings (see `.env.example` for the full list):

- `SYMBOLS` — comma-separated watchlist (default `SPY`); each runs its own
  independent session sharing one IB connection and journal.
- `ORB_WINDOW_MINUTES` (default 15), `ORB_BREAKOUT_BUFFER_PCT` (default 0.0005).
- `MAX_CONTRACTS` (default 10), `MAX_PREMIUM_USD` (default 2000) — raised but capped.
- `MAX_TRADES_PER_DAY` (default 2), `NO_NEW_ENTRIES_BEFORE`/`AFTER`.
- `STOP_MIN_CENTS`/`STOP_MAX_CENTS` (default 5/400) — the sane-distance band the
  opposite-side stop must fall inside.
- `RUNNER_STOP` — `breakeven` (default) or `first_target`.
- `PROTECTIVE_STOP=true` also rests a GTC market sell at IB, triggered by the
  underlying crossing the plan's stop — a backstop when tajator is down.

## Safety

- Paper by default. Going live requires changing **both** `TRADING_MODE=live` and
  `IB_PORT` to a live port (4001 for IB Gateway, 7496 for TWS) — one without the
  other refuses to start; paper mode refuses to connect to any live port.
- Operator-owned kill switch: `touch KILL` in the repo root blocks all new entries
  immediately (existing positions are still managed and can exit). Tajator reads
  this file but never creates it.
- `run` refuses to start if the IB account already holds option positions in a
  configured symbol — it only manages positions it opened itself.
- If the IB connection drops, the loop reconnects on the next tick and journals it.
- A partial, unconfirmed, or execution-quality-breaching order halts new entries
  inside the running process and notifies the operator (without creating `KILL`).
- Ctrl-C during `run` offers to flatten any open position.
- Everything is journaled to `logs/journal-YYYY-MM-DD.jsonl`: snapshots, candidates,
  decisions, risk vetoes, fills, quote preflights, order timelines, and execution quality.

## Out of scope

Multi-symbol scanning (the watchlist is a fixed list), dashboards, greeks/IV
modeling, option-spread strategies, limit orders, holiday calendar. Market orders only.
