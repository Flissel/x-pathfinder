import pytest

from x_pathfinder.promotion import MissingServiceKey, PromotionGate, SupabaseWriter


class _StubDb:
    def __init__(self):
        self.marked = []

    def mark_promoted(self, handle):
        self.marked.append(handle)


class _StubWriter:
    def __init__(self):
        self.rows = []

    def insert(self, table, row):
        self.rows.append((table, row))


def test_only_validated_rows_are_promoted():
    db, writer = _StubDb(), _StubWriter()
    gate = PromotionGate(db, writer)
    summary = gate.promote(
        [
            {"handle": "good", "validated": True, "fitness_score": 90.0},
            {"handle": "bad", "validated": False, "fitness_score": 80.0},
            {"handle": "unknown", "validated": None, "fitness_score": 70.0},
        ]
    )

    assert summary == {"promoted": 1, "skipped": 2}
    assert [row["handle"] for _, row in writer.rows] == ["good"]
    assert db.marked == ["good"]


def test_already_promoted_row_is_not_promoted_twice():
    db, writer = _StubDb(), _StubWriter()
    gate = PromotionGate(db, writer)
    summary = gate.promote(
        [{"handle": "good", "validated": True, "promoted_at": "2026-09-02T10:00:00"}]
    )

    assert summary == {"promoted": 0, "skipped": 1}
    assert writer.rows == []


@pytest.mark.parametrize("falsy_promoted_at", ["", 0, False])
def test_falsy_but_not_none_promoted_at_still_skips(falsy_promoted_at):
    db, writer = _StubDb(), _StubWriter()
    gate = PromotionGate(db, writer)
    summary = gate.promote(
        [{"handle": "good", "validated": True, "promoted_at": falsy_promoted_at}]
    )

    assert summary == {"promoted": 0, "skipped": 1}
    assert writer.rows == []
    assert db.marked == []


def test_writer_without_service_key_raises_before_any_network_call():
    calls = []

    def _poster(url, headers, payload):
        calls.append(url)
        return 201

    with pytest.raises(MissingServiceKey):
        SupabaseWriter("http://192.168.178.65:54321", "", poster=_poster).insert(
            "marketing_accounts", {"handle": "x"}
        )
    assert calls == []


def test_writer_posts_to_the_rest_endpoint_with_auth_headers():
    seen = {}

    def _poster(url, headers, payload):
        seen["url"] = url
        seen["headers"] = headers
        seen["payload"] = payload
        return 201

    SupabaseWriter("http://192.168.178.65:54321", "svc-key", poster=_poster).insert(
        "marketing_accounts", {"handle": "x"}
    )

    assert seen["url"] == "http://192.168.178.65:54321/rest/v1/marketing_accounts"
    assert seen["headers"]["apikey"] == "svc-key"
    assert seen["headers"]["Authorization"] == "Bearer svc-key"
    assert seen["payload"] == {"handle": "x"}
