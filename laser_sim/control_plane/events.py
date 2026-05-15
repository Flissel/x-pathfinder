"""Thread-safe event bus for live EA streaming.

The GeneticEngine emits one EvolutionEvent per generation (and optionally
per-evaluation events later). Subscribers — typically SSE endpoints — pull
events from a queue with a timeout-aware get(). Multiple subscribers each
get their own queue so a slow client doesn't back-pressure the EA.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    CAMPAIGN_START = "campaign_start"
    GENERATION = "generation"
    HF_PROMOTION = "hf_promotion"
    CAMPAIGN_END = "campaign_end"
    ERROR = "error"


@dataclass(frozen=True)
class EvolutionEvent:
    type: EventType
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        data = {"type": self.type.value, "ts": self.timestamp, "payload": self.payload}
        return f"data: {json.dumps(data, default=float)}\n\n"


class EventBus:
    """Multi-subscriber broker. Subscribers get their own thread-safe queue."""

    def __init__(self, max_queue: int = 256) -> None:
        self._subs: list[queue.Queue[EvolutionEvent]] = []
        self._lock = threading.Lock()
        self._max_queue = max_queue
        self._history: list[EvolutionEvent] = []
        self._history_cap = 64

    def publish(self, event: EvolutionEvent) -> None:
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_cap:
                self._history = self._history[-self._history_cap :]
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                # drop on overflow — slow subscribers don't stall the EA
                pass

    def subscribe(self) -> tuple[queue.Queue[EvolutionEvent], list[EvolutionEvent]]:
        q: queue.Queue[EvolutionEvent] = queue.Queue(maxsize=self._max_queue)
        with self._lock:
            self._subs.append(q)
            backlog = list(self._history)
        return q, backlog

    def unsubscribe(self, q: queue.Queue[EvolutionEvent]) -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)
