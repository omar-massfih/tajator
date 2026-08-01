"""Unit tests for the momentum-basket rebalancer's pure logic."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from tajator.models import Bar
from tajator.rebalance import build_plan, compute_target

ET = ZoneInfo("America/New_York")


def _series(closes, *, start=datetime(2024, 1, 2, tzinfo=ET)):
    """Daily bars with the given closes on consecutive calendar days."""
    return [
        Bar(ts=start + timedelta(days=i), open=c, high=c, low=c, close=c)
        for i, c in enumerate(closes)
    ]


def test_compute_target_ranks_by_12_1_momentum():
    # 300 bars each; STRONG rises 2x over the window, WEAK falls, FLAT is flat.
    strong = _series([100 + i for i in range(300)])          # steady climber
    weak = _series([300 - i for i in range(300)])            # steady faller
    flat = _series([100.0] * 300)
    daily = {"STRONG": strong, "WEAK": weak, "FLAT": flat}

    target = compute_target(daily, lookback=252, skip=21, decile=0.34)  # top ~1 of 3
    assert target.members[0] == "STRONG"
    assert "WEAK" not in target.members
    assert target.scores["STRONG"] > 0


def test_compute_target_drops_stale_series():
    fresh = _series([100 + i for i in range(300)])
    # stale name: same shape but its last bar is a month older
    stale = _series([100 + i for i in range(300)],
                    start=datetime(2023, 11, 1, tzinfo=ET))
    target = compute_target({"FRESH": fresh, "STALE": stale}, decile=1.0)
    assert "FRESH" in target.members
    assert "STALE" not in target.members


def test_compute_target_skips_names_without_enough_history():
    long = _series([100 + i for i in range(300)])
    short = _series([100 + i for i in range(50)])  # < lookback+1
    target = compute_target({"LONG": long, "SHORT": short}, decile=1.0)
    assert target.members == ["LONG"]
    assert target.ranked_universe == 1


def _basket(members):
    from tajator.rebalance import TargetBasket
    from datetime import date
    return TargetBasket(date(2024, 6, 1), members, {m: 0.5 for m in members}, len(members),
                        252, 21, 0.1)


def test_build_plan_equal_weights_and_buys_from_flat():
    target = _basket(["AAA", "BBB"])
    plan = build_plan(target, current={}, prices={"AAA": 100.0, "BBB": 50.0}, capital=1000.0)
    # $500 per name: 5 shares AAA, 10 shares BBB
    assert plan.desired_shares == {"AAA": 5, "BBB": 10}
    actions = {(o.symbol, o.action, o.qty) for o in plan.orders}
    assert actions == {("AAA", "BUY", 5), ("BBB", "BUY", 10)}


def test_build_plan_liquidates_dropped_names_and_trims():
    target = _basket(["AAA"])
    # holding a dropped name (ZZZ) and too many AAA already
    plan = build_plan(
        target,
        current={"AAA": 20, "ZZZ": 7},
        prices={"AAA": 100.0, "ZZZ": 30.0},
        capital=1000.0,
    )
    # target AAA = 10 shares -> SELL 10; ZZZ fully liquidated -> SELL 7
    orders = {(o.symbol, o.action): o.qty for o in plan.orders}
    assert orders[("AAA", "SELL")] == 10
    assert orders[("ZZZ", "SELL")] == 7
    # sells come before any buys in the plan
    assert all(o.action == "SELL" for o in plan.orders)


def test_build_plan_noop_when_already_aligned():
    target = _basket(["AAA", "BBB"])
    plan = build_plan(
        target,
        current={"AAA": 5, "BBB": 10},
        prices={"AAA": 100.0, "BBB": 50.0},
        capital=1000.0,
    )
    assert plan.orders == []
