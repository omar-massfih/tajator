"""Telegram trade notifications — best-effort, never blocks trading."""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Protocol

from .models import ExecutedAction, OpenPosition

log = logging.getLogger(__name__)


class Notifier(Protocol):
    def notify_fill(self, symbol: str, action: ExecutedAction, position: OpenPosition) -> None: ...

    def notify_status(self, text: str) -> None: ...


class NullNotifier:
    def notify_fill(self, symbol: str, action: ExecutedAction, position: OpenPosition) -> None:
        pass

    def notify_status(self, text: str) -> None:
        pass


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id

    def _send(self, text: str) -> None:
        body = json.dumps({"chat_id": self._chat_id, "text": text}).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:  # noqa: BLE001 — a failed notification must not break trading
            log.warning("telegram notify failed: %s", exc)

    def notify_fill(self, symbol: str, action: ExecutedAction, position: OpenPosition) -> None:
        text = (
            f"{symbol} {action.kind.upper()} {action.qty}x {position.contract.local_name} "
            f"@ {action.premium:.2f} (underlying {action.equity_price:.2f})"
        )
        if action.reason:
            text += f"\n{action.reason}"
        self._send(text)

    def notify_status(self, text: str) -> None:
        self._send(text)
