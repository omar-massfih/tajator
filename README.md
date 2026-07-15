# Tajator

> **Name origin:** *Tajator* is shaped from Arabic *tājir* (تاجر, "trader") and
> Latin *mercator* ("merchant"), carrying the idea of trade, markets, and craft.


An agentic AI options trader built with **LangGraph + LangChain**, executing the
support/resistance strategy from `../trading/strategy-notes/` through **Interactive Brokers**
(paper account by default).

The strategy in one line: buy **calls** as price moves down into support, buy **puts**
as price moves up into resistance (incl. double tops/bottoms), mental stop ~40 cents
beyond the level on the equity, use EMA9 as chart context only, scale out in
pieces at the 50 EMA/VWAP → high/low of day, and protect the final runner at
break-even (`RUNNER_STOP=first_target` restores the notes' tighter lock at the
first target). Discipline gates added after the first live-data backtests:
a level that stops us out is untradable for `STOP_COOLDOWN_MINUTES`, a *very*
fast approach must print a rejection wick on the entry bar before it can
become a candidate, and role-reversed levels (a broken support retested as
resistance, or vice versa) stay chart context instead of trades
(`TRADE_FLIPPED_LEVELS` re-admits them).

## Design: the LLM proposes, code disposes

- **Deterministic code** computes indicators (9/50 EMA, session VWAP), detects levels
  (prev-day H/L, premarket H/L, double tops/bottoms) and pre-filters setups
  ("price approaching a level with speed"). No candidate → the LLM isn't even asked.
- **The LLM** (via `init_chat_model`, so any provider works) makes only the judgment
  calls: take this setup or wait; scale this piece now or hold one more bar. It returns
  a structured `Decision` and its reasoning is journaled verbatim.
- **Pattern-data mode** is a separate, opt-in paper policy. Every five completed
  bars it serializes the latest 120 one-minute OHLCV bars plus objective swing pivots
  and asks the model to classify only double tops/bottoms, head-and-shoulders variants,
  or confirmed triangle breakouts. Code independently checks direction, confidence, a recent
  completed-bar breakout, visible levels, and the no-chase band before creating a
  normal setup candidate.
- **Hard guardrails** veto anything else: market hours, max 2 trades/day, one position
  at a time, stop required on the correct side (20–60 cents), the LLM cannot invent
  trades the detector didn't flag, premium budget caps, and a kill-switch file.
- **Stops are never LLM-negotiable**: the mental stop, first-target runner stop, and
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
uv run tajator check-ib     # compare entry market-data latency on client 118 — places NO orders
uv run tajator check-ib --entry-samples 3  # recommended collection run on each of 2 sessions
uv run tajator entry-data-report  # evaluate the frozen paired latency/validity gate
uv run tajator test-order   # paper diagnostic: buy 1 lot, watch the fill timeline, sell it back
uv run tajator test-order --with-stop   # + place/verify/cancel a protective stop mid-trade
uv run tajator replay --csv tests/data/spy_sample_day.csv --no-llm \
    --prev-high 503.5 --prev-low 497.0   # bundled synthetic day, no IB/LLM needed
uv run tajator replay --date 2026-07-02          # fetch a real day from IB, replay with the LLM
uv run tajator replay --date 2026-07-02 --pattern-data  # TWS bars → numerical-pattern policy
uv run tajator backtest --symbol SPY --start 2026-04-01 --end 2026-06-30 --no-llm
uv run tajator backtest --symbol AAPL --start 2026-06-01 --end 2026-06-30 \
    --pattern-data --underlying-only --experiment aapl-pattern-data-v1
uv run tajator backtest --symbol AAPL --start 2026-07-14 --end 2026-07-14 \
    --no-llm --tws-chain-snapshot --experiment exact-current-day  # completed day only
uv run tajator backtest --symbol SPY --start 2026-04-01 --end 2026-06-30 \
    --no-llm --underlying-only --experiment baseline  # long-window signal research
uv run tajator backtest-compare logs/backtests/*_baseline.json logs/backtests/*_risk-cap.json
uv run tajator strategy-compare \
    logs/backtests/MSFT_2024-07-01_2025-12-31_msft-risk-cap-holdout-current-v1.json \
    logs/backtests/MSFT_2024-07-01_2025-12-31_msft-risk-cap-holdout-candidate-v1.json \
    --min-trades 250 --only-change max_entry_to_stop_cents \
    --output logs/research/msft-risk-cap-holdout-v1.json
uv run tajator edge-audit logs/backtests/AAPL_2025-07-01_2026-06-30_baseline.json \
    --validation-start 2026-01-01  # judge only a pre-declared holdout
uv run tajator forward-init --name aapl-panel-v4 --symbol AAPL  # lock before observation
uv run tajator forward-validate --name aapl-rejection-v1 --symbol AAPL \
    --date 2026-07-13  # capture a completed session before its options expire
uv run tajator forward-latest --name msft-panel-v2 --symbol MSFT  # preferred daily capture
uv run tajator run          # deterministic live minute loop (paper by default)
uv run tajator run --llm    # explicitly opt into experimental LLM decisions
uv run tajator run --pattern-data  # experimental numerical-pattern policy; paper mode only
uv run tajator shadow --symbol MSFT  # live TWS quotes, deterministic simulated fills, NO orders
uv run tajator shadow-report logs/shadow --symbol MSFT
uv run tajator edge-audit logs/shadow/MSFT_shadow_report.json --validation-only
uv run tajator option-panel-compare logs/forward/msft-forward-v1/cumulative.json
uv run tajator execution-calibrate logs/journal-2026-07-13.jsonl --symbol MSFT
uv run tajator historical-signal-tournament  # cached historical TWS stock bars
uv run tajator historical-signal-followup    # preregistered opening-drive fade
uv run tajator historical-daily-fetch        # long-run historical TWS daily bars
uv run tajator historical-daily-tournament   # gated swing-signal study
uv run tajator historical-aapl-focus         # AAPL-first temporal holdout; MSFT conditional
uv run pytest                     # full test suite
```

`test-order` is the supervised acceptance check after any incident or execution
change. It uses the same quote validation, budget sizing, market-order timeout,
fill reconciliation, and execution telemetry as production, then immediately
sells the confirmed paper position. A pass requires valid live bid/ask data,
both fills within `MAX_ACCEPTABLE_FILL_LATENCY_S`, and no slippage or budget
breach. Watch TWS while it runs. Live mode additionally requires a recent pass
for every configured symbol plus `EXECUTION_LIVE_CONFIRMED=true`.

`strategy-compare` is the promotion gate for a baseline and one locked
candidate replayed over the same stock-price window. It aggregates each replay
by active trading day because vetoing a setup can expose a different later
trade; absent trades on either side count as zero for that paired day. The
report checks the candidate's absolute confidence interval, paired daily
improvement interval, coverage, positive months and half-years, drawdown, and
prints every changed strategy setting. Use a fresh isolated TWS cache for the
baseline and `--cached-only` for the candidate so both see identical bars.

Ordinary long-window backtests synthesize Friday expirations around the strike
grid available from TWS; that is appropriate for stock-signal research but not
for auditing today's selected contract. After the current session is complete,
`--tws-chain-snapshot` refreshes the full stock session, refuses incomplete
coverage, captures the actual TWS expirations and strikes, and prices the exact
contract from its historical option bars. It is deliberately restricted to one
current-day online diagnostic and must never be presented as prospective
evidence unless the cohort itself was registered beforehand.

Live entries are guarded before submission: the option needs a fresh, narrow
bid/ask; sizing uses ask plus a reserve; and the underlying must still be inside
the setup zone without having crossed the stop or already moved away. TWS
requests both facts in one synchronous snapshot and the accepted snapshot is
carried directly into submission, avoiding a second 10+ second snapshot wait.
`check-ib` also measures an experimental five-second temporary-stream path
before the production snapshot and persists paired no-order records under
`logs/diagnostics/`. After-hours missing option bid/ask is reported but cannot
qualify the stream candidate; production remains unchanged until the frozen
multi-session latency and quote-validity gate passes. `entry-data-report`
excludes records outside 09:30-14:00 ET, requires paired records for both AAPL
and MSFT, and applies fixed thresholds that are intentionally not CLI options.
Use `--entry-samples 3` during 09:30-14:00 ET on two separate sessions: samples
are bounded to 1-5, separated by five seconds after each completed pair, and
cannot bypass the audit's independent two-session requirement.
Orders remain DAY market orders. Risk-removing exits submit immediately and are
never blocked or delayed by a missing or wide quote. Entries journal their
accepted snapshot; every order journals its status timeline and fill latency. A
slow, slipped, or over-budget confirmed fill is adopted and managed but
activates the kill switch against new entries.

With `PROTECTIVE_STOP=true`, every entry also rests a GTC market sell at IB,
triggered by the *underlying* crossing the plan's stop price. The in-loop mental
stop stays primary; the broker-side order is the backstop that protects the
position when tajator is down, disconnected, or has lost track of it. The agent
never sells while a stop might still be working (cancel-and-confirm first), and
startup recognizes its own stops by `orderRef`, re-placing/cancelling as needed —
anything foreign still refuses to launch.

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
worse than no number. Fills are priced at the *next* option bar's open (never
the bar whose close triggered the decision), so there is no look-ahead bias;
the day's forced flatten uses the last bar's close. It prints an aggregate win-rate/PnL/drawdown summary and
writes a per-trade ledger + daily equity curve to
`logs/backtests/<symbol>_<start>_<end>.json`. Known limitation: no market-holiday
calendar (holidays are just days with no bars, silently skipped).

Backtest fills include a configurable adverse half-spread, slippage, and
commission model (see `BACKTEST_*` in `.env.example`). Reports preserve the
exact strategy settings, execution assumptions, LLM mode, and git revision,
and separate gross PnL, fees, and net PnL. Each trade also records profit
factor inputs, return on premium, planned underlying stop distance, exit
reason, and underlying maximum favorable/adverse excursion over its lifetime.

`--underlying-only` replays the identical detector, risk gates, stops, and
targets using historical stock bars and reports direction-adjusted underlying
points. It is a signal-quality research mode, not an options-PnL simulation.
Use it for long windows where IB no longer exposes expired option contracts.
`--skip-missing-option-data` instead retains real-option mode but excludes and
reports an entire day if any required option fill is unavailable.
Every new report carries an experiment label and configuration fingerprint in
both its filename and metadata, so variants cannot overwrite one another.
`backtest-compare` recomputes expectancy from each trade ledger rather than
trusting older persisted summary fields. Its confidence interval is
Bonferroni-adjusted across every supplied variant, preventing a lucky result
from being promoted merely because many configurations were tried. Even a
positive familywise interval remains exploratory until the chosen policy
passes a subsequently frozen holdout.
Symbol-specific strategy overrides, actual entry-to-stop caps, opening-window
confirmation, ATR stops, regime filters, and level-quality filters are typed
settings documented in `.env.example`; all remain disabled by default.

`historical-signal-tournament` is a separate causal stock-signal research
engine over cached TWS one-minute bars. It compares a small preregistered set of
intraday rules on a development period, enters only after a completed signal,
holds for a fixed 60 bars, deducts stock execution costs, and opens the
chronological/cross-symbol validation split only for a mechanically eligible
development winner. Confidence intervals cluster observations by trading day
across symbols. `historical-signal-followup` implements the one declared
development-derived opening-drive fade; it cannot silently substitute another
rule after validation. Neither tool simulates option PnL—only a stock signal
that passes every validation gate may advance to exact contract prices.

`historical-daily-fetch` requests long-run RTH daily stock `TRADES` bars from
TWS and caches the returned series under `data/historical/daily/`. The loader
refuses adjacent-close discontinuities above 80% rather than assuming a split
adjustment. `historical-daily-tournament` tests preregistered multi-day signals
using next-session entries, non-overlapping positions, return-normalized costs,
unseen-symbol validation, and a final temporal holdout that remains unopened
unless validation passes. A daily stock signal still cannot establish an
options edge without a separate exact-contract cost and decay test.

`historical-aapl-focus` applies the already-selected daily trend/RSI rule to a
fresh, isolated TWS AAPL temporal holdout. It opens MSFT replication only after
every AAPL gate passes, and exposes `options_stage_eligible` only for an AAPL
pass. The default cache is `data/tws-focused/`; populate it directly from TWS
with `historical-daily-fetch --symbols AAPL,MSFT --cache-dir data/tws-focused`.

`edge-audit` is the guard against promoting an attractive backtest into a
supposed edge. It requires an explicitly declared holdout (`--validation-start`
or `--validation-only`), at least 50 trades by default, positive expectancy, a
95% expectancy interval above zero, positive results in at least 60% of active
months, and adequate data coverage. Historical-options reports must also have
a disclosed cost model and profit factor of at least 1.2. Underlying-only
reports can support a stock signal but can never confirm an options edge. A
holdout must be declared before looking at its results; the flag records the
research claim but cannot make an already-inspected sample out-of-sample.

Because IB may stop qualifying a weekly option soon after expiration,
`forward-validate` is the durable options-research path. Run it after each
completed session (normally for the previous trading day). The first capture
locks the resolved symbol strategy, cost assumptions, deterministic no-LLM
mode, and a fingerprint of the actual Python sources—including uncommitted
changes. Each day also records the actual TWS option-chain snapshot, so replay
uses the expirations that were genuinely listed (including Monday/Wednesday
weeklies) instead of the synthetic Friday calendar required for old history.
Later captures refuse to join that cohort if any definition changes.
Each successful day is stored under `logs/forward/<name>/days/`, and
`cumulative.json` is rebuilt for direct use with `edge-audit --validation-only`.
Missing exact option data aborts the capture rather than silently excluding the
day. Use a new cohort name whenever strategy or execution code intentionally
changes.

For daily operation, prefer `forward-latest`. It connects with dedicated
read-only API client ID 117 by default, searches backward for the newest prior
weekday that actually has TWS stock bars, and captures it idempotently. This
handles weekends and holidays without guessing a session date and avoids the
live trader's normal client ID. Run it once after each market day (or the next
morning before the nearest weekly expires); use `--client-id` if 117 is already
occupied. It places no orders.

Completed-session capture forcibly refreshes the underlying day from TWS even
when a CSV is already cached. A cohort admits the day only when regular-session
bars begin by 09:31 ET and contain either a full session through 15:59 ET (at
least 370 bars) or a standard 13:00 ET early close (at least 200 bars). The bar
count, first/last timestamps, and session kind are preserved per day in the
cumulative report. Truncated intraday caches fail loudly instead of becoming
apparently complete backtests.

`execution-calibrate` matches journaled paper fills to the exact contract's
historical option bar immediately after each signal, then compares actual fills
with the configured spread/slippage model. It reports signal-to-fill latency,
delay beyond the modeled next-bar open, net round trips, and
actual-versus-modeled adversity. Modern journals apply exact IB commission
reports by order ID; configured fees fill any unmatched legacy records. Treat LLM and
deterministic calibration samples separately: LLM samples include model
decision latency and cannot justify replacing deterministic backtest costs.
Older journals without commission reports remain explicitly labeled estimates.

Plain `run` applies the same deterministic rule-follower used by
`backtest --no-llm` to entries and management, so paper/live observations match
the frozen forward cohorts by default. `run --deterministic` remains an accepted
explicit compatibility flag. LLM decisions require `run --llm`; treat that as a
separate experimental policy and never combine its fills with deterministic
validation samples.

AAPL's source-level default is the frozen forward candidate: touch-rejection
entries, at most $1.00 from entry to the stock stop, no entries after 14:00 ET,
and no puts in a detected `trend_up` regime. The override is AAPL-specific;
other symbols retain the global strategy unless `SYMBOL_STRATEGY_OVERRIDES`
explicitly replaces the built-in map.

`run --pattern-data` is another separate policy and is hard-blocked when
`TRADING_MODE=live`. It works with the normal text model or the `codex`
subscription backend. The prompt contains compact completed OHLCV rows and an
objective pivot list; the journal stores only its SHA-256, bar/pivot counts,
structured classification, and validation outcome rather than duplicating the
full prompt. Scans default to once per five
completed bars after at least 60 bars are available, so a long historical
backtest can make many billable, nondeterministic model calls. Use a short
development range first, then freeze settings and evaluate an untouched AAPL
holdout; never mix its evidence with deterministic cohorts.

The pattern catalog is evidence-informed, not a claim that historical shapes are an
edge. Lo, Mamaysky, and Wang found that some objectively defined patterns added
information in a long U.S. stock sample, while other work found no significantly
positive head-and-shoulders returns in a different market. That mixed record is
why this mode requires confirmation and must earn its own prospective result:
[NBER Working Paper 7613](https://www.nber.org/papers/w7613) and
[Lucke (2003)](https://doi.org/10.1080/00036840210150884).

Use `shadow` to collect execution evidence without authorizing even paper
orders. It runs that same deterministic graph on live TWS stock bars and option
quotes through dedicated client ID 116, simulates buys at the displayed ask and
sells at the displayed bid, applies configured commissions, and writes isolated
state/journals under `logs/shadow/`. Its broker exposes market data only: entry,
exit, and protective-stop operations are local simulations and never call an IB
order method. Delayed or invalid bid/ask quotes are rejected.

`shadow-report` converts those quote-side fills into the standard options
ledger accepted by `edge-audit`. Coverage counts only sessions where the
process observed nearly the entire regular session through 15:54 ET; merely
starting and stopping the process does not earn a complete day. Keep one
predeclared deterministic policy and collect at least 50 closed trades across
at least three active months before the ordinary audit gates can confirm an
edge. Shadow mode is intentionally long-running; stop it with Ctrl-C after the
session. No order was placed by building or testing this feature.

Run `forward-init` before the first session of a new prospective cohort. It is
a local, no-TWS operation that freezes the symbol, strategy configuration,
execution model, capture protocol, and executable-source fingerprint. A
pre-session manifest is conservatively eligible starting the next calendar
day, so a cohort initialized after observing any part of today's session cannot
retroactively ingest today. Repeating the same initialization is idempotent;
reusing its name after any fingerprint change is refused.

Forward capture also records a predeclared option panel for every base trade:
one listed strike ITM at the base expiration, one strike OTM, and the base
strike at the next listed expiration. Every alternative uses the exact base
entry/exit timestamps, quantities, adverse spread/slippage, and commissions.
`option-panel-compare` reports net PnL and average return on premium and clearly
marks incomplete variants. Dollar PnL is not capital-normalized because
quantity is deliberately held fixed; use return on premium when comparing
contract efficiency. Panel results are exploratory contract-selection evidence
and do not alter or replace the base cohort's edge audit. Alternatives are
paired to their exact base trades and need at least 50 complete pairs plus a
positive Bonferroni-adjusted 95% familywise confidence bound before the tool
labels their return-on-premium advantage positive. Even then, no contract
variant is promotable until the base options strategy passes `edge-audit`.

Cohorts fingerprint executable trading and backtest behavior, including
uncommitted changes, while reporting-only modules are excluded. The capture
pipeline is separately locked by an explicit protocol version recorded in the
manifest. This preserves legitimate evidence across presentation/statistics
improvements without allowing entry, exit, pricing, or contract-universe code
to change inside a cohort.

Optional multi-timeframe context (`MULTI_TIMEFRAME_CONTEXT=true`) keeps entries
and stops on the 1-minute chart while adding completed daily trend/ATR/reference
levels and 09:30-aligned 5-minute structure. Daily levels are context only. The
forming 5-minute candle is labeled as incomplete, and higher-timeframe evidence
only ranks already-valid candidates; it never creates or vetoes a setup. A
frozen AAPL July 2025-June 2026 underlying-only A/B produced 158 exactly
identical trade records in each arm, so rank-only context was behaviorally inert
and failed promotion. The feature remains off; do not enable it as an alleged
improvement without a new, independently declared hypothesis and holdout.

## Safety

- Paper by default. Going live requires changing **both** `TRADING_MODE=live` and
  `IB_PORT` to a live port (4001 for IB Gateway, 7496 for TWS) — one without the
  other refuses to start; paper mode refuses to connect to any live port.
- Kill switch: `touch KILL` in the repo root blocks all new entries immediately
  (existing positions are still managed and can exit).
- `run` refuses to start if the IB account already holds option positions in a
  configured symbol — the agent only manages positions it opened itself (it
  knows their plan and stop); flatten strays manually first.
- If the IB connection drops (e.g. the Gateway's nightly restart), the loop
  reconnects automatically on the next minute tick and journals the outage.
- A partially filled order that had to be cancelled activates the kill switch
  automatically: untracked contracts at IB mean no new entries until the
  operator reconciles and deletes the KILL file.
- Ctrl-C during `run` offers to flatten any open position.
- Everything is journaled to `logs/journal-YYYY-MM-DD.jsonl`: snapshots, candidates,
  every LLM decision + reasoning, risk vetoes, fills, quote preflights, order
  timelines, execution quality, diagnostics, and commission reports.

## Out of scope (v1)

Multi-symbol scanning (the watchlist is a fixed list, not a scanner), dashboards,
greeks/IV modeling, option-spread strategies, limit orders, holiday calendar.
Market orders only.

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
