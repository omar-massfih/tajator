# Tajator

> **Name origin:** *Tajator* is shaped from Arabic *tājir* (تاجر, "trader") and
> Latin *mercator* ("merchant"), carrying the idea of trade, markets, and craft.

A deterministic intraday options day-trader built with **LangGraph**, executing a
**support/resistance fade** through **Interactive Brokers** (paper account by default),
**plus** an offline research toolkit for hunting and out-of-sample-validating stock edges.

The strategy in one line: buy **calls** as price falls *into* support and **puts** as
price rises *into* resistance (prev-day / premarket highs & lows, qualified double
tops/bottoms), with a mental stop ~40¢ beyond the level, scale out in pieces toward the
50-EMA/VWAP then the high/low of day, and protect the runner at break-even. Discipline
gates: an approach must have speed, a *very* fast approach must print a rejection wick,
role-reversed levels stay chart context, and a stopped-out level is untradable for a
cooldown. Per-symbol overrides tune it (AAPL ships a frozen touch-rejection/$1-risk-cap/
14:00-cutoff policy).

> **Experimental, paper-only. Options trading involves substantial risk of loss.
> Do not point it at real money you cannot afford to lose.**

## Honest findings (what this project actually established)

Extensive out-of-sample research in this repo reached a clear, tested conclusion:

- **Intraday single-name options strategies have no durable edge here** — the S/R fade
  and an Opening-Range Breakout both backtest at noise-level expectancy (~0.02–0.09
  underlying pts/trade), and every "improvement" that looked good in-sample (parameter
  sweeps, trend-alignment filters) **failed out-of-sample**. Option spread + theta then
  turns those coin-flips into net losers.
- **The one directional edge that survived every gate is cross-sectional / time-series
  momentum in *stocks*** — long the strongest-momentum names, ~monthly rebalance, validated
  across 112 symbols, unseen names, and hostile regimes (a ~10–14 %/yr *premium*, Sharpe ~0.8,
  its magnitude inflated by survivorship). It is a **stock** strategy; a ~1 %/month edge
  cannot be bought as options (see `edge-search` + `option-economics`). Live via `momentum-rebalance`.
- **The one *options-native* edge is the volatility risk premium** — implied vol (VIX) exceeds
  realized vol ~85 % of days (~4 vol pts), so *selling* vol earns a carry. Naive short vol is a
  death trap (SVXY lost ~91 % in Feb-2018); gating it on the **VIX term structure** (hold SVXY
  only when VIX < VIX3M) tames the tail to a *survivable* ~35 % drawdown for ~13–18 %/yr on the
  post-2018 instrument. It is a **risk premium** (payment for bearing crash risk), not free alpha —
  Sharpe ~0.6. See `vol-edge-search`.

The fade trades cleanly and is the default bot; the momentum toolkit is where a real,
tradeable edge lives. Neither is a money printer — treat the numbers honestly.

## Design: deterministic, one path

```
fetch_data → compute_context ─┬─ (position open) → manage_position → stop/runner exit | scale out
                              └─ (flat) → detect_setups → decide → risk_gate → enter
```

- **`compute_context` / `detect_setups`** detect levels (`market/levels.py`) and
  "price approaching a level with speed" candidates (`market/setups.py`), rank and
  regime/quality/cooldown-filter them. No candidate → nothing happens. No LLM in the path.
- **`risk_gate`** (`risk/guardrails.py`) is a non-negotiable veto: market hours,
  `MAX_TRADES_PER_DAY`, one position at a time, the decision must match a *detected*
  candidate, a stop on the correct side within the 20–60¢ band, an actual entry-to-stop
  risk cap, and a kill-switch file.
- **Stops** (`trade/position.py`) run in code every tick: mental stop beyond the level,
  break-even after the first scale-out, VWAP runner exit; flat before the close.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 installed automatically):

```bash
uv sync
cp .env.example .env   # then fill it in
```

**IBKR:** install IB Gateway or TWS, log into your **paper** account, enable the API
(port 4002 for IB Gateway paper, 7497 for TWS paper).

## Commands

Live / diagnostics:
```bash
uv run tajator run          # live minute loop, S/R fade (paper by default)
uv run tajator check-ib     # connectivity: bars, chain, quote — places NO orders
uv run tajator test-order   # supervised paper diagnostic: buy 1 lot, watch fills, sell back
uv run tajator replay --csv tests/data/spy_sample_day.csv --symbol SPY \
    --prev-high 503.5 --prev-low 497.0   # bundled day, no IB needed
uv run tajator backtest --symbol AAPL --start 2026-04-01 --end 2026-06-30 --underlying-only
uv run tajator backtest-compare logs/backtests/*_a.json logs/backtests/*_b.json
```

Edge research (offline, on the cached daily/intraday bars):
```bash
uv run tajator daily-fetch                       # paced IB fetch of ~100 names' daily bars
uv run tajator daily-fetch --symbols VIX,VIX3M,VXX,SVXY --daily-dir data/historical/vol  # vol data
uv run tajator edge-search --horizon daily       # test documented signals, OOS-validated
uv run tajator momentum-backtest                 # monthly momentum stock basket, cost-aware
uv run tajator vol-edge-search                   # volatility risk premium: term-structure short vol
```

Live momentum basket (the one validated edge — a **stock** book, not options):
```bash
uv run tajator momentum-rebalance --capital 100000 --offline   # preview from cache, no IB
uv run tajator momentum-rebalance --capital 100000             # fetch fresh, show plan (dry-run)
uv run tajator momentum-rebalance --capital 100000 --execute   # place paper stock orders
```
`momentum-rebalance` ranks the universe by 12-1 momentum, targets an equal-weight top-decile
basket, reconciles it against current IB stock positions, and prints the order plan. It is
**dry-run by default** (nothing is placed without `--execute`), paper-by-default, honours the
KILL switch, fetches split/dividend-adjusted bars, and logs each run to
`logs/rebalances/`. Run it ~monthly.

`backtest`/`replay` step the *same* graph; `--underlying-only` reports direction-adjusted
stock points (this account has no expired-option data, so that is the usable research mode).
`edge-search` gates every signal on a temporal holdout, a cross-symbol holdout, a Bonferroni
correction, and day-clustered errors; `option-economics` prices whether a signal survives as
stock / ITM / ATM / spread. `test-order` is the supervised acceptance check before live.

## Configuration

Key `.env` settings (see `.env.example`):

- `SYMBOLS` — watchlist (currently `AAPL`; each runs an independent session).
- Level/setup detection: `APPROACH_BAND_PCT`, `MIN_SPEED_PCT`, `REJECTION_WICK_MIN_FRAC`,
  `DOUBLE_MIN_*`, `SWING_WINDOW_BARS`, `ENTRY_CONFIRMATION` (`immediate`/`touch_rejection`).
- Stops: `STOP_MIN_CENTS`/`STOP_MAX_CENTS` (20/60), `STOP_ATR_MULTIPLIER`,
  `MAX_ENTRY_TO_STOP_CENTS`, `STOP_COOLDOWN_MINUTES`, `RUNNER_STOP`.
- Sizing: `MAX_CONTRACTS`, `MAX_PREMIUM_USD`, `MAX_TRADES_PER_DAY`, entry windows.
- Filters (opt-in): `ALLOWED_REGIMES`, `BLOCKED_DIRECTION_REGIMES`, `MIN_LEVEL_QUALITY_SCORE`.
- `SYMBOL_STRATEGY_OVERRIDES` — per-symbol tuning; AAPL ships a frozen policy by default.
- `PROTECTIVE_STOP=true` rests a broker-side GTC stop as a backstop.

## Safety

- Paper by default. Live requires changing **both** `TRADING_MODE=live` and `IB_PORT` to a
  live port — one without the other refuses to start.
- Kill switch: `touch KILL` blocks all new entries (open positions still managed). Tajator
  reads it, never creates it.
- `run` refuses to start on foreign option positions in a configured symbol.
- A partial / unconfirmed / quality-breaching order halts new entries in-process.
- Ctrl-C offers to flatten. Everything is journaled to `logs/journal-YYYY-MM-DD.jsonl`.

## Out of scope

Multi-symbol *options* scanning, dashboards, greeks/IV modeling, option spreads, limit
orders, holiday calendar. Market orders only. (The momentum basket trades **stocks**, not
options — `momentum-rebalance` is a separate long-only book from the intraday options bot.)
