"""Promotion of validated stage rows into Supabase.

Writes go through the Kong REST API because the Proxmox Postgres port is
not reachable from this host. Only rows the validator confirmed are ever
promoted; everything else stays in stage with its refusal reason.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_SUPABASE_URL = "http://192.168.178.65:54321"


class MissingServiceKey(RuntimeError):
    """No Supabase service key configured, so nothing may be written."""


def _default_poster(url: str, headers: dict, payload: dict) -> int:
    import requests

    response = requests.post(url, headers=headers, json=payload, timeout=(5, 30))
    return response.status_code


class SupabaseWriter:
    def __init__(
        self,
        base_url: str = DEFAULT_SUPABASE_URL,
        service_key: str = "",
        poster: Callable[[str, dict, dict], int] = _default_poster,
    ):
        self._base_url = base_url.rstrip("/")
        self._service_key = service_key
        self._poster = poster

    def insert(self, table: str, row: dict) -> None:
        if not self._service_key:
            raise MissingServiceKey(
                "SUPABASE_SERVICE_KEY is not set; refusing to write"
            )
        status = self._poster(
            f"{self._base_url}/rest/v1/{table}",
            {
                "apikey": self._service_key,
                "Authorization": f"Bearer {self._service_key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            },
            row,
        )
        if not (200 <= int(status) < 300):
            raise RuntimeError(f"supabase insert failed with status {status}")


class PromotionGate:
    """One-way gate: only validated=True rows leave stage."""

    TABLE = "marketing_accounts"

    def __init__(self, db, writer):
        self._db = db
        self._writer = writer

    def promote(self, rows: list[dict]) -> dict:
        promoted = skipped = 0
        for row in rows:
            if row.get("validated") is not True or row.get("promoted_at"):
                skipped += 1
                continue
            self._writer.insert(self.TABLE, row)
            self._db.mark_promoted(row["handle"])
            promoted += 1
        return {"promoted": promoted, "skipped": skipped}
