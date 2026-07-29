"""Thread-safe event bus for live EA streaming.

The GeneticEngine emits one EvolutionEvent per generation (and optionally
per-evaluation events later). Subscribers — typically SSE endpoints — pull
events from a queue with a timeout-aware get(). Multiple subscribers each
get their own queue so a slow client doesn't back-pressure the EA.
"""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from enum import Enum
from typing import Any


class EventType(str, Enum):
    CAMPAIGN_START = "campaign_start"
    GENERATION = "generation"
    HF_PROMOTION = "hf_promotion"
    CAMPAIGN_END = "campaign_end"
    ERROR = "error"
    VOLUME_FRAME = "volume_frame"


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


def pack_volume_frame(
    t_max_K: np.ndarray,
    *,
    vmin_K: float,
    vmax_K: float,
    t_s: float,
    laser_x_mm: float,
    laser_y_mm: float,
    frame_index: int,
    n_frames: int,
    roi_mm: tuple[float, float, float, float] | None = None,
    depth_mm: float | None = None,
) -> dict:
    """Quantize a (Nx, Ny, Nz) T_max volume to uint8 + base64 for SSE wire.

    Decoder (browser-side):
      const bytes = atob(payload.data_b64);
      const arr = new Uint8Array(bytes.length);
      for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
      // arr indexed (Nx, Ny, Nz) in C-order
      // T_K(i,j,k) = vmin + (arr[i*Ny*Nz + j*Nz + k] / 255) * (vmax - vmin)
    """
    if t_max_K.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {t_max_K.shape}")
    span = max(vmax_K - vmin_K, 1e-6)
    q = np.clip((t_max_K - vmin_K) / span, 0.0, 1.0)
    u8 = (q * 255.0 + 0.5).astype(np.uint8)
    payload = {
        "frame_index": int(frame_index),
        "n_frames": int(n_frames),
        "t_s": float(t_s),
        "laser_x_mm": float(laser_x_mm),
        "laser_y_mm": float(laser_y_mm),
        "shape": list(t_max_K.shape),
        "dtype": "uint8",
        "vmin_K": float(vmin_K),
        "vmax_K": float(vmax_K),
        "data_b64": base64.b64encode(u8.tobytes()).decode("ascii"),
    }
    # optional fallback metadata for late subscribers who missed campaign_start
    if roi_mm is not None:
        payload["roi_mm"] = [float(v) for v in roi_mm]
    if depth_mm is not None:
        payload["depth_mm"] = float(depth_mm)
    return payload
