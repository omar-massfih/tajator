"""Monthly momentum-basket rebalancer — the live, long-only *stock* strategy.

This is the tradeable form of the one edge that survived out-of-sample validation
(see `backtest/momentum.py` and the edge-search research): rank a large-cap
universe by 12-1 month momentum, hold an equal-weight basket of the top decile,
rebalance monthly. Holdout (2022-26) Sharpe ~1.1 / ~35%/yr on the cached universe
— honestly, survivorship-inflated to a true ~0.7-0.9, and long-biased (it draws
down in bear markets). It is a *stock* book, not options.

This module is deliberately split: `compute_target` and `build_plan` are pure
functions over bars / prices / positions (unit-tested), and the IB I/O lives in
`cmd_momentum_rebalance` (CLI). Orders are plain stock market orders; execution is
opt-in (`--execute`) and paper-by-default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .models import Bar

DEFAULT_LOOKBACK = 252  # ~12 months
DEFAULT_SKIP = 21       # skip the most recent month (12-1 momentum)
DEFAULT_DECILE = 0.10   # top decile
STALE_TRADING_DAYS = 5  # drop a name whose last bar lags the freshest this much


@dataclass(frozen=True)
class TargetBasket:
    as_of: date
    members: list[str]           # top-decile symbols, strongest first
    scores: dict[str, float]     # symbol -> 12-1 momentum (members only)
    ranked_universe: int         # how many names had a valid score
    lookback: int
    skip: int
    decile: float


@dataclass(frozen=True)
class Order:
    symbol: str
    action: str          # "BUY" | "SELL"
    qty: int
    ref_price: float     # for display / value estimate only (market order)

    @property
    def notional(self) -> float:
        return self.qty * self.ref_price


@dataclass(frozen=True)
class RebalancePlan:
    target: TargetBasket
    capital: float
    prices: dict[str, float]
    desired_shares: dict[str, int]   # symbol -> target share count
    current: dict[str, int]          # symbol -> shares held now
    orders: list[Order]

    @property
    def deployed(self) -> float:
        return sum(q * self.prices.get(s, 0.0) for s, q in self.desired_shares.items())

    @property
    def turnover(self) -> float:
        return sum(o.notional for o in self.orders)


def _momentum_score(bars: list[Bar], lookback: int, skip: int) -> float | None:
    """12-1 momentum from the latest bar: return over [-lookback, -skip]. Causal by
    construction (uses only closes up to the freshest bar)."""
    if len(bars) < lookback + 1:
        return None
    start = bars[-1 - lookback]
    end = bars[-1 - skip] if skip else bars[-1]
    if start.close <= 0:
        return None
    return end.close / start.close - 1.0


def compute_target(
    daily: dict[str, list[Bar]],
    *,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    decile: float = DEFAULT_DECILE,
) -> TargetBasket:
    """Rank the universe by 12-1 momentum and return the top-decile basket.

    Names whose freshest bar lags the newest in the panel by more than
    STALE_TRADING_DAYS are dropped — a stale series would score on old prices."""
    last_dates = [b[-1].ts.date() for b in daily.values() if b]
    if not last_dates:
        return TargetBasket(date.today(), [], {}, 0, lookback, skip, decile)
    newest = max(last_dates)
    cutoff = newest - timedelta(days=STALE_TRADING_DAYS * 2 + 4)  # calendar pad for weekends

    scores: dict[str, float] = {}
    for symbol, bars in daily.items():
        if not bars or bars[-1].ts.date() < cutoff:
            continue
        sc = _momentum_score(bars, lookback, skip)
        if sc is not None:
            scores[symbol] = sc

    ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
    k = max(1, int(len(ranked) * decile)) if ranked else 0
    members = ranked[:k]
    return TargetBasket(
        as_of=newest,
        members=members,
        scores={s: scores[s] for s in members},
        ranked_universe=len(ranked),
        lookback=lookback,
        skip=skip,
        decile=decile,
    )


def build_plan(
    target: TargetBasket,
    current: dict[str, int],
    prices: dict[str, float],
    capital: float,
) -> RebalancePlan:
    """Reconcile the target equal-weight basket against current holdings into a
    set of stock market orders. Held names not in the target are fully liquidated;
    held names in the target are topped up / trimmed to the target share count."""
    members = [s for s in target.members if prices.get(s, 0.0) > 0]
    n = len(members)
    per_name = capital / n if n else 0.0
    desired = {s: int(per_name // prices[s]) for s in members}

    orders: list[Order] = []
    # Sells first (frees buying power): trim or liquidate anything held above target.
    for symbol in sorted(current):
        held = current[symbol]
        want = desired.get(symbol, 0)
        if held > want:
            orders.append(Order(symbol, "SELL", held - want, prices.get(symbol, 0.0)))
    # Then buys: bring target names up to the desired share count.
    for symbol in members:
        held = current.get(symbol, 0)
        want = desired[symbol]
        if want > held:
            orders.append(Order(symbol, "BUY", want - held, prices[symbol]))

    return RebalancePlan(
        target=target,
        capital=capital,
        prices=prices,
        desired_shares=desired,
        current=dict(current),
        orders=orders,
    )


def format_plan(plan: RebalancePlan) -> str:
    """Human-readable rebalance preview."""
    t = plan.target
    lines = [
        f"Momentum basket rebalance  (as of {t.as_of}, "
        f"{t.lookback}-{t.skip} momentum, top {t.decile:.0%} of {t.ranked_universe} names)",
        f"Capital ${plan.capital:,.0f}  →  {len(t.members)} names, "
        f"~${plan.capital / max(1, len(t.members)):,.0f} each",
        "",
        f"{'TARGET':<8}{'mom%':>8}{'price':>10}{'shares':>8}{'value':>12}   {'held':>6}",
    ]
    for s in t.members:
        px = plan.prices.get(s, 0.0)
        want = plan.desired_shares.get(s, 0)
        lines.append(
            f"{s:<8}{t.scores[s] * 100:>7.1f}%{px:>10.2f}{want:>8}"
            f"{want * px:>12,.0f}   {plan.current.get(s, 0):>6}"
        )
    lines += ["", f"{'ORDERS':<8}{'action':>7}{'qty':>8}{'~price':>10}{'~notional':>12}"]
    if not plan.orders:
        lines.append("  (none — already aligned)")
    for o in plan.orders:
        lines.append(f"{o.symbol:<8}{o.action:>7}{o.qty:>8}{o.ref_price:>10.2f}{o.notional:>12,.0f}")
    lines += [
        "",
        f"Deploy ${plan.deployed:,.0f} of ${plan.capital:,.0f}  "
        f"(cash left ${plan.capital - plan.deployed:,.0f})   "
        f"turnover ${plan.turnover:,.0f} across {len(plan.orders)} orders",
    ]
    return "\n".join(lines)
