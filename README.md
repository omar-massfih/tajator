# Tajator

> **Name origin:** *Tajator* is shaped from Arabic *tājir* (تاجر, "trader") and
> Latin *mercator* ("merchant"), carrying the idea of trade, markets, and craft.


An agentic AI options trader built with **LangGraph + LangChain**, executing the
support/resistance strategy from `../trading/strategy-notes/` through **Interactive Brokers**
(paper account by default).

The strategy in one line: buy **calls** as price moves down into support, buy **puts**
as price moves up into resistance (incl. double tops/bottoms), mental stop ~40 cents
beyond the level on the equity, scale out in pieces at the 9 EMA → 50 EMA/VWAP →
high/low of day, and protect the final runner at break-even.

## Design: the LLM proposes, code disposes

- **Deterministic code** computes indicators (9/50 EMA, session VWAP), detects levels
  (prev-day H/L, premarket H/L, double tops/bottoms) and pre-filters setups
  ("price approaching a level with speed"). No candidate → the LLM isn't even asked.
- **The LLM** (via `init_chat_model`, so any provider works) makes only the judgment
  calls: take this setup or wait; scale this piece now or hold one more bar. It returns
  a structured `Decision` and its reasoning is journaled verbatim.
- **Hard guardrails** veto anything else: market hours, max 2 trades/day, one position
  at a time, stop required on the correct side (20–60 cents), the LLM cannot invent
  trades the detector didn't flag, premium budget caps, and a kill-switch file.
- **Stops are never LLM-negotiable**: the mental stop, runner break-even, and
  VWAP runner rules are enforced in code on every tick, before anything else.

Graph per 1-minute tick:

```
fetch_data → compute_context ─┬─ (position open) → manage_position → stop/runner exit | llm_manage → scale out
                              └─ (flat) → detect_setups → llm_decide → risk_gate → select contract → enter
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically):

```bash
uv sync
cp .env.example .env   # then fill it in
```

**LLM backend** (`LLM_MODEL` in `.env`):

- `codex` (default) — uses the [Codex CLI](https://github.com/openai/codex) with your
  ChatGPT subscription: `brew install codex` and `codex login` once, no API key needed.
  Each decision runs `codex exec` in a read-only sandbox from an empty scratch dir,
  with the `Decision` JSON schema enforced via `--output-schema`. Optionally pin a
  model with `codex:<model>`.
- `openai:gpt-5.1` (or any langchain `init_chat_model` string, e.g.
  `anthropic:claude-sonnet-5`) — direct API access; needs that provider's key
  (`OPENAI_API_KEY`, etc.). Note a ChatGPT subscription is *not* an OpenAI API key.

**IBKR:** install [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php),
log into your **paper** account, and enable the API (Configure → API → Settings →
*Enable ActiveX and Socket Clients*; port 4002 for paper). Without a market-data
subscription the agent falls back to delayed data (fine for plumbing tests,
not for live timing).

## Commands

```bash
uv run tajator check-ib     # connectivity: bars, chain, quote — places NO orders
uv run tajator replay --csv tests/data/spy_sample_day.csv --no-llm \
    --prev-high 503.5 --prev-low 497.0   # bundled synthetic day, no IB/LLM needed
uv run tajator replay --date 2026-07-02          # fetch a real day from IB, replay with the LLM
uv run tajator backtest --symbol SPY --start 2026-04-01 --end 2026-06-30 --no-llm
uv run tajator run          # live minute loop (paper) during market hours
uv run pytest                     # full test suite
```

Replay steps the *same graph* through a recorded day with instant synthetic option
fills — it validates plumbing and decision flow, it is not a backtest. Any position
still open at the recorded day's close is force-flattened at the last bar so
replay/backtest ledgers count every trade (live trading never auto-flattens; it
warns and journals instead).

`backtest` steps the same graph over every trading day in a date range, fetching
(and caching under `data/historical/`) real underlying bars from IB and, for every
fill, the real historical option quote for that exact contract/day. There is no
synthetic-price fallback: candidate strikes/expirations are generated the same
mechanical way `replay` does (nearest strike to spot, nearest non-0DTE Friday),
but the instant a specific fill can't be priced from real IB data (illiquid
strike, or an expired contract that fails to qualify) the backtest aborts
immediately with the offending contract/day named in the error, rather than
quietly mixing in a guessed price — a partially-synthetic PnL number would be
worse than no number. It prints an aggregate win-rate/PnL/drawdown summary and
writes a per-trade ledger + daily equity curve to
`logs/backtests/<symbol>_<start>_<end>.json`. Known limitation: no market-holiday
calendar (holidays are just days with no bars, silently skipped).

## Safety

- Paper by default. Going live requires changing **both** `TRADING_MODE=live` and
  `IB_PORT` to a live port (4001 for IB Gateway, 7496 for TWS) — one without the
  other refuses to start; paper mode refuses to connect to any live port.
- Kill switch: `touch KILL` in the repo root blocks all new entries immediately
  (existing positions are still managed and can exit).
- Ctrl-C during `run` offers to flatten any open position.
- Everything is journaled to `logs/journal-YYYY-MM-DD.jsonl`: snapshots, candidates,
  every LLM decision + reasoning, risk vetoes, and fills.

## Out of scope (v1)

Multi-symbol scanning (the watchlist is a fixed list, not a scanner), dashboards,
greeks/IV modeling, spreads, limit orders, broker-side stops (the mental stop is
enforced by the loop), holiday calendar. Market orders only.

Fixed watchlist (`SYMBOLS=SPY,AAPL,MSFT,NVDA`, one comma-separated env var) — each
symbol runs its own independent `TradingSession` (own position, own daily trade
counter) sharing one IB connection, journal, and LLM client. Sessions tick
sequentially, so with many symbols and a slow LLM one pass can exceed the 60s
cadence and delay stop checks for later symbols — keep the watchlist short when
running with the LLM. `check-ib` checks
connectivity for every configured symbol; `replay` still exercises one symbol per
run via `--symbol` (defaults to the first configured symbol).

**This is an experimental system for paper trading. Options trading involves
substantial risk of loss. Do not point it at real money.**
