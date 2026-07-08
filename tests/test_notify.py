import json
from datetime import datetime
from zoneinfo import ZoneInfo

from tajator.models import ExecutedAction, OpenPosition, PositionPlan, SelectedContract
from tajator.notify import NullNotifier, TelegramNotifier

ET = ZoneInfo("America/New_York")


def make_position() -> OpenPosition:
    contract = SelectedContract(symbol="SPY", expiry="20260710", strike=500.0, right="C")
    plan = PositionPlan(
        direction="call", level_price=500.0, stop_price=499.6, entry_equity_price=500.0,
        entry_premium=1.5, total_qty=2, pieces=[1, 1], target_refs=["ema9"],
    )
    return OpenPosition(contract=contract, plan=plan, qty_remaining=2, opened_at=datetime(2026, 7, 6, 11, 0, tzinfo=ET))


def make_action(kind="entry", reason="") -> ExecutedAction:
    return ExecutedAction(kind=kind, qty=2, premium=1.55, equity_price=500.1, ts=datetime.now(ET), reason=reason)


def test_null_notifier_is_inert():
    notifier = NullNotifier()
    notifier.notify_fill("SPY", make_action(), make_position())
    notifier.notify_status("anything")


def test_telegram_notifier_posts_fill_message(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        captured["timeout"] = timeout

    monkeypatch.setattr("tajator.notify.urllib.request.urlopen", fake_urlopen)

    notifier = TelegramNotifier("TOKEN", "12345")
    notifier.notify_fill("SPY", make_action(kind="entry", reason="broke above level"), make_position())

    assert captured["url"] == "https://api.telegram.org/botTOKEN/sendMessage"
    assert captured["body"]["chat_id"] == "12345"
    assert "SPY ENTRY 2x SPY 20260710 500C @ 1.55" in captured["body"]["text"]
    assert "broke above level" in captured["body"]["text"]


def test_telegram_notifier_posts_status_message(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "tajator.notify.urllib.request.urlopen",
        lambda req, timeout=None: captured.update(body=json.loads(req.data)),
    )

    TelegramNotifier("TOKEN", "12345").notify_status("tajator started | SPY | PAPER")

    assert captured["body"]["text"] == "tajator started | SPY | PAPER"


def test_telegram_notifier_swallows_send_failures(monkeypatch):
    def raise_error(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr("tajator.notify.urllib.request.urlopen", raise_error)

    notifier = TelegramNotifier("TOKEN", "12345")
    notifier.notify_fill("SPY", make_action(), make_position())  # must not raise
    notifier.notify_status("hello")  # must not raise
