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
uv run tajator run          # live minute loop (paper) during market hours
uv run pytest                     # full test suite
```

Replay steps the *same graph* through a recorded day with instant synthetic option
fills — it validates plumbing and decision flow, **it is not a backtest**.

## Safety

- Paper by default. Going live requires changing **both** `TRADING_MODE=live` and
  `IB_PORT=4001` — one without the other refuses to start.
- Kill switch: `touch KILL` in the repo root blocks all new entries immediately
  (existing positions are still managed and can exit).
- Ctrl-C during `run` offers to flatten any open position.
- Everything is journaled to `logs/journal-YYYY-MM-DD.jsonl`: snapshots, candidates,
  every LLM decision + reasoning, risk vetoes, and fills.

## Out of scope (v1)

Backtesting/PnL analytics, multi-symbol scanning, dashboards, greeks/IV modeling,
spreads, limit orders, broker-side stops (the mental stop is enforced by the loop),
holiday calendar. Single symbol (`SYMBOL=SPY`), market orders only.

**This is an experimental system for paper trading. Options trading involves
substantial risk of loss. Do not point it at real money.**
