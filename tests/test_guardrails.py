from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tajator.config import Settings
from tajator.models import (
    Decision,
    Level,
    OpenPosition,
    PositionPlan,
    SelectedContract,
    SetupCandidate,
)
from tajator.risk.guardrails import check, cooldown_filter, entry_blockers

ET = ZoneInfo("America/New_York")
MIDDAY = datetime(2026, 7, 6, 11, 0, tzinfo=ET)  # Monday 11:00 ET

CANDIDATE = SetupCandidate(
    direction="call",
    level=Level(price=499.0, kind="support", label="prev_day_low"),
    distance=0.2,
    speed=-0.8,
)
GOOD_ENTRY = Decision(
    action="enter_call", level_price=499.0, stop_price=498.6, confidence="high", reasoning="test"
)


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path)


def run_check(settings, decision=GOOD_ENTRY, *, now=MIDDAY, position=None, trades=0,
              candidates=(CANDIDATE,)):
    return check(
        decision, now=now, position=position, trades_today=trades,
        candidates=list(candidates), settings=settings,
    )


def open_position():
    contract = SelectedContract(symbol="SPY", expiry="20260710", strike=499.0, right="C")
    plan = PositionPlan(
        direction="call", level_price=499.0, stop_price=498.6, entry_equity_price=499.2,
        entry_premium=2.0, total_qty=4, pieces=[2, 1, 1],
        target_refs=["ema50_vwap", "hod_lod", "runner"],
    )
    return OpenPosition(contract=contract, plan=plan, qty_remaining=4, opened_at=MIDDAY)


def test_valid_entry_approved(settings):
    verdict = run_check(settings)
    assert verdict.approved, verdict.violations


def test_wait_and_exits_always_pass(settings):
    for action in ("wait", "scale_out", "exit"):
        assert run_check(settings, Decision(action=action, reasoning="x")).approved


def test_kill_switch_vetoes(settings):
    settings.kill_switch_file.write_text("stop")
    verdict = run_check(settings)
    assert not verdict.approved and any("kill switch" in v for v in verdict.violations)


def test_delayed_data_vetoes_entries(settings):
    verdict = check(
        GOOD_ENTRY, now=MIDDAY, position=None, trades_today=0,
        candidates=[CANDIDATE], settings=settings, delayed_data=True,
    )
    assert not verdict.approved and any("DELAYED" in v for v in verdict.violations)


def test_delayed_data_veto_can_be_disabled(settings):
    settings.block_entries_on_delayed_data = False
    verdict = check(
        GOOD_ENTRY, now=MIDDAY, position=None, trades_today=0,
        candidates=[CANDIDATE], settings=settings, delayed_data=True,
    )
    assert verdict.approved


def test_weekend_vetoes(settings):
    saturday = MIDDAY + timedelta(days=5)
    assert not run_check(settings, now=saturday).approved


def test_late_entry_vetoes(settings):
    late = MIDDAY.replace(hour=15, minute=45)
    assert not run_check(settings, now=late).approved


def test_early_entry_vetoes_configurable_window(settings):
    settings.no_new_entries_before = MIDDAY.time().replace(hour=11, minute=30)
    assert not run_check(settings, now=MIDDAY).approved


def test_max_trades_vetoes(settings):
    assert not run_check(settings, trades=2).approved


def test_open_position_vetoes_new_entry(settings):
    assert not run_check(settings, position=open_position()).approved


def test_llm_cannot_invent_trades(settings):
    # No candidate at this level / direction.
    put = Decision(action="enter_put", level_price=505.0, stop_price=505.4, reasoning="x")
    verdict = run_check(settings, put)
    assert not verdict.approved and any("may not invent" in v for v in verdict.violations)


def test_stop_on_wrong_side_vetoes(settings):
    bad = Decision(action="enter_call", level_price=499.0, stop_price=499.4, reasoning="x")
    assert not run_check(settings, bad).approved


def test_stop_distance_outside_rule_vetoes(settings):
    too_far = Decision(action="enter_call", level_price=499.0, stop_price=497.5, reasoning="x")
    assert not run_check(settings, too_far).approved


def test_missing_stop_vetoes(settings):
    no_stop = Decision(action="enter_call", level_price=499.0, reasoning="x")
    assert not run_check(settings, no_stop).approved


def test_stop_band_is_configurable(settings, tmp_path):
    # 80 cents: outside the default 20–60 rule, inside a widened 10–100 band.
    far = Decision(action="enter_call", level_price=499.0, stop_price=498.2, reasoning="x")
    assert not run_check(settings, far).approved
    wide = Settings(
        _env_file=None, kill_switch_file=tmp_path / "KILL", log_dir=tmp_path,
        stop_min_cents=10, stop_max_cents=100,
    )
    assert run_check(wide, far).approved


def test_actual_entry_to_stop_risk_cap(settings):
    settings.max_entry_to_stop_cents = 100
    verdict = check(
        GOOD_ENTRY, now=MIDDAY, position=None, trades_today=0,
        candidates=[CANDIDATE], settings=settings, snapshot_price=499.8,
    )
    assert not verdict.approved
    assert any("actual entry-to-stop risk" in v for v in verdict.violations)


def test_entry_blockers_clear_midday(settings):
    assert entry_blockers(now=MIDDAY, position=None, trades_today=0, settings=settings) == []


def test_entry_blockers_honors_process_local_broker_halt_without_kill_file(settings):
    blockers = entry_blockers(
        now=MIDDAY,
        position=None,
        trades_today=0,
        settings=settings,
        internal_halt_reason="partial fill requires reconciliation",
    )

    assert blockers == [
        "broker halted new entries for this process: partial fill requires reconciliation"
    ]
    assert not settings.kill_switch_file.exists()


def test_entry_blockers_collects_cheap_vetoes(settings):
    settings.kill_switch_file.write_text("stop")
    blockers = entry_blockers(
        now=MIDDAY.replace(hour=16),  # after the entry window
        position=open_position(),
        trades_today=2,
        settings=settings,
    )
    assert len(blockers) == 4  # kill switch, time window, max trades, open position


def test_cooldown_filter_drops_the_stopped_out_level():
    other = SetupCandidate(
        direction="put",
        level=Level(price=502.0, kind="resistance", label="double_top"),
        distance=0.2,
        speed=0.8,
    )
    # 499.5 is within LEVEL_MATCH_TOL (0.2% ~ $1) of the 499.0 candidate —
    # a re-formed double at a slightly different print must still be cooled.
    kept, dropped = cooldown_filter([CANDIDATE, other], [499.5])
    assert kept == [other]
    assert dropped == [CANDIDATE]


def test_cooldown_filter_without_cooled_levels_keeps_everything():
    kept, dropped = cooldown_filter([CANDIDATE], [])
    assert kept == [CANDIDATE] and dropped == []
