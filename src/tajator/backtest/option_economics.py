"""Translate a validated *underlying* signal into option economics.

A small directional edge in the stock can still be a loser once expressed as
options, because you pay bid/ask + theta. This prices the same expected move
across four expressions — stock, deep-ITM (~70Δ), ATM, and a debit spread — with
a Black-Scholes model, and reports each one's net return plus the underlying move
it needs just to break even. We have no expired-option data, so this is a *model*;
the definitive test is forward-collecting real fills once a signal is validated.
"""

from __future__ import annotations

from math import erf, exp, log, sqrt
from statistics import pstdev

from ..models import Bar

TRADING_DAYS = 252


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price(S: float, K: float, T: float, sigma: float, r: float = 0.04, call: bool = True) -> float:
    """European Black-Scholes price. T in years."""
    if T <= 0 or sigma <= 0:
        intrinsic = (S - K) if call else (K - S)
        return max(0.0, intrinsic)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if call:
        return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    return K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def realized_vol(daily_bars: list[Bar], window: int = 20) -> float:
    """Annualized realized volatility from the last `window` daily closes."""
    closes = [b.close for b in daily_bars][-(window + 1):]
    if len(closes) < 3:
        return 0.0
    rets = [log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(rets) < 2:
        return 0.0
    return pstdev(rets) * sqrt(TRADING_DAYS)


def _leg_roundtrip(S0, S1, K, T0, T1, sigma, r, call, half_spread_pct) -> tuple[float, float]:
    """Return (entry_cost, exit_value) for buying one option and selling it later,
    paying half the bid/ask on each side."""
    entry_mid = bs_price(S0, K, T0, sigma, r, call)
    exit_mid = bs_price(S1, K, T1, sigma, r, call)
    entry_cost = entry_mid * (1 + half_spread_pct)
    exit_value = exit_mid * (1 - half_spread_pct)
    return entry_cost, exit_value


def _net_pct(entry_cost: float, exit_value: float) -> float:
    return (exit_value / entry_cost - 1.0) if entry_cost > 0 else 0.0


def _breakeven_move(S0, K, T0, T1, sigma, r, call, half_spread_pct) -> float:
    """The signed underlying move fraction at which the option round-trip nets 0."""
    entry_cost, _ = _leg_roundtrip(S0, S0, K, T0, T1, sigma, r, call, half_spread_pct)
    lo, hi = (0.0, 0.5) if call else (-0.5, 0.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        _, exit_value = _leg_roundtrip(S0, S0 * (1 + mid), K, T0, T1, sigma, r, call, half_spread_pct)
        net = exit_value - entry_cost
        if call:
            lo, hi = (mid, hi) if net < 0 else (lo, mid)
        else:
            lo, hi = (lo, mid) if net < 0 else (mid, hi)
    return round((lo + hi) / 2, 4)


def compare_expressions(
    S0: float,
    expected_move_frac: float,
    horizon_days: int,
    sigma: float,
    *,
    dte_days: int | None = None,
    half_spread_pct: float = 0.03,
    r: float = 0.04,
) -> dict:
    """Compare stock / ITM / ATM / debit-spread for a directional bet with expected
    move `expected_move_frac` over `horizon_days`. Positive move → calls, negative → puts."""
    call = expected_move_frac >= 0
    dte_days = dte_days if dte_days is not None else max(horizon_days + 5, 7)
    T0 = dte_days / TRADING_DAYS
    T1 = max(dte_days - horizon_days, 0) / TRADING_DAYS
    S1 = S0 * (1 + expected_move_frac)

    # ITM ~0.97 moneyness (calls) / 1.03 (puts); ATM = S0; spread sells the ~1.02 wing.
    itm_k = S0 * (0.97 if call else 1.03)
    atm_k = S0
    otm_k = S0 * (1.03 if call else 0.97)

    def option_row(K):
        entry, exit_ = _leg_roundtrip(S0, S1, K, T0, T1, sigma, r, call, half_spread_pct)
        return {
            "net_pct": round(_net_pct(entry, exit_) * 100, 1),
            "breakeven_move_pct": round(_breakeven_move(S0, K, T0, T1, sigma, r, call, half_spread_pct) * 100, 2),
        }

    # Debit spread: long ATM, short OTM wing (both round-tripped adversely).
    la_e, la_x = _leg_roundtrip(S0, S1, atm_k, T0, T1, sigma, r, call, half_spread_pct)
    so_e, so_x = _leg_roundtrip(S0, S1, otm_k, T0, T1, sigma, r, call, half_spread_pct)
    spread_entry = la_e - so_x  # buy ATM at ask, sell OTM at bid
    spread_exit = la_x - so_e
    spread_net = _net_pct(spread_entry, spread_exit) * 100 if spread_entry > 0 else 0.0

    return {
        "underlying": S0,
        "expected_move_pct": round(expected_move_frac * 100, 3),
        "horizon_days": horizon_days,
        "dte_days": dte_days,
        "iv_annual_pct": round(sigma * 100, 1),
        "stock": {"net_pct": round(expected_move_frac * 100, 3)},
        "itm_option": option_row(itm_k),
        "atm_option": option_row(atm_k),
        "debit_spread": {"net_pct": round(spread_net, 1)},
    }
