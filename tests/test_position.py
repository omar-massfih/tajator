from datetime import datetime
from zoneinfo import ZoneInfo

from tajator.models import OpenPosition, SelectedContract, Snapshot
from tajator.trade.position import build_plan, evaluate, split_pieces, update_extreme

ET = ZoneInfo("America/New_York")
TS = datetime(2026, 7, 6, 11, 0, tzinfo=ET)

CONTRACT = SelectedContract(symbol="SPY", expiry="20260710", strike=499.0, right="C")


def snap(price, ema9=None, ema50=None, vwap=None, hod=None, lod=None):
    return Snapshot(symbol="SPY", ts=TS, price=price, ema9=ema9, ema50=ema50, vwap=vwap, hod=hod, lod=lod)


def call_position(qty=4, pieces_sold=0, entry=499.2, stop=498.6):
    plan = build_plan("call", 499.0, stop, entry, 2.0, qty)
    return OpenPosition(
        contract=CONTRACT, plan=plan, qty_remaining=qty - pieces_sold,
        pieces_sold=pieces_sold, opened_at=TS,
    )


def put_position(qty=4, pieces_sold=0, entry=501.8, stop=502.4):
    plan = build_plan("put", 502.0, stop, entry, 2.0, qty)
    return OpenPosition(
        contract=SelectedContract(symbol="SPY", expiry="20260710", strike=502.0, right="P"),
        plan=plan, qty_remaining=qty - pieces_sold, pieces_sold=pieces_sold, opened_at=TS,
    )


def test_split_pieces():
    assert split_pieces(4) == [1, 1, 1, 1]
    assert split_pieces(5) == [2, 1, 1, 1]
    assert split_pieces(3) == [1, 1, 1]
    assert split_pieces(2) == [1, 1]
    assert split_pieces(1) == [1]
    assert split_pieces(10) == [3, 3, 2, 2]


def test_plan_targets():
    assert build_plan("call", 499.0, 498.6, 499.2, 2.0, 4).target_refs == [
        "ema9", "ema50_vwap", "hod_lod", "runner",
    ]
    assert build_plan("call", 499.0, 498.6, 499.2, 2.0, 2).target_refs == ["ema9", "runner"]
    assert build_plan("call", 499.0, 498.6, 499.2, 2.0, 1).target_refs == ["ema9"]


def test_stop_fires_first_for_call_and_put():
    action = evaluate(call_position(), snap(498.55, ema9=500.0))
    assert action.kind == "stop_exit"
    action = evaluate(put_position(), snap(502.45, ema9=501.0))
    assert action.kind == "stop_exit"


def test_stop_beats_scale_target():
    # Even if a target were somehow touched, stop wins (checked first).
    action = evaluate(call_position(), snap(498.5, ema9=498.0))
    assert action.kind == "stop_exit"


def test_scale_sequence_for_call():
    pos = call_position()
    # piece 1: 9 EMA
    assert evaluate(pos, snap(499.6, ema9=500.0)).kind == "hold"
    a = evaluate(pos, snap(500.1, ema9=500.0))
    assert a.kind == "scale_candidate" and a.target_ref == "ema9"
    # piece 2: nearer of ema50/vwap above
    pos.pieces_sold = 1
    a = evaluate(pos, snap(500.6, ema9=500.0, ema50=500.5, vwap=500.8))
    assert a.kind == "scale_candidate" and a.target_ref == "ema50_vwap"
    # piece 3: HOD
    pos.pieces_sold = 2
    assert evaluate(pos, snap(501.0, ema9=500.0, hod=501.5)).kind == "hold"
    a = evaluate(pos, snap(501.6, ema9=500.0, hod=501.5))
    assert a.kind == "scale_candidate" and a.target_ref == "hod_lod"


def test_missing_indicator_holds():
    assert evaluate(call_position(), snap(500.1)).kind == "hold"  # no ema9 yet


def test_runner_break_even_exit():
    pos = call_position(pieces_sold=3)
    assert evaluate(pos, snap(500.0, ema9=500.0)).kind == "hold"
    a = evaluate(pos, snap(499.15, ema9=500.0))
    assert a.kind == "runner_exit" and "break-even" in a.reason


def test_runner_vwap_loss_exit_only_after_being_beyond():
    pos = call_position(pieces_sold=3)
    # Price never got above VWAP: crossing below it is not an exit signal.
    update_extreme(pos, 500.0)
    assert evaluate(pos, snap(499.9, vwap=500.5)).kind == "hold"
    # Price had been above VWAP, now lost it: exit the runner.
    update_extreme(pos, 501.0)
    a = evaluate(pos, snap(500.4, vwap=500.5))
    assert a.kind == "runner_exit" and "VWAP" in a.reason


def test_put_runner_vwap_reclaim_exit():
    pos = put_position(pieces_sold=3)
    update_extreme(pos, 500.2)  # traded below VWAP in our favor
    a = evaluate(pos, snap(501.2, vwap=501.0))
    assert a.kind == "runner_exit"


def test_single_contract_exits_at_first_target_no_runner():
    pos = call_position(qty=1)
    a = evaluate(pos, snap(500.1, ema9=500.0))
    assert a.kind == "scale_candidate" and a.target_ref == "ema9"
    assert len(pos.plan.pieces) == 1  # selling this piece closes the position
