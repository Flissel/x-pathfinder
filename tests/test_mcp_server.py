import pytest

from x_pathfinder.mcp_server import TOOLS, handle_call

EXPECTED_TOOLS = {
    "xpf_discover", "xpf_score", "xpf_validate",
    "xpf_promote", "xpf_status", "xpf_export",
}


def test_all_six_tools_are_registered():
    assert set(TOOLS) == EXPECTED_TOOLS


def test_unknown_tool_returns_error_not_exception():
    result = handle_call("xpf_nonexistent", {})
    assert result["ok"] is False
    assert "unknown tool" in result["error"]


def test_handler_exception_is_returned_as_error_payload():
    def _boom(**kwargs):
        raise RuntimeError("stage db down")

    TOOLS["xpf_status"] = _boom
    try:
        result = handle_call("xpf_status", {})
        assert result["ok"] is False
        assert "stage db down" in result["error"]
    finally:
        from x_pathfinder.mcp_server import _tool_status
        TOOLS["xpf_status"] = _tool_status


def test_promote_uses_get_validated_unpromoted_not_get_unvalidated():
    """Wiring check: _tool_promote must source rows from
    get_validated_unpromoted(), never get_unvalidated().

    get_unvalidated() selects WHERE validated IS NULL (rows the validator
    hasn't touched yet); PromotionGate.promote() only promotes rows where
    validated is exactly True. Wired to get_unvalidated(), every row would
    be skipped forever while xpf_promote kept reporting success. This test
    would fail if someone "simplified" _tool_promote back to that shape.
    """
    from x_pathfinder import mcp_server

    calls = []

    class _FakeDb:
        def get_unvalidated(self, limit=100):
            calls.append(("get_unvalidated", limit))
            return [{"handle": "wrong-method", "validated": None}]

        def get_validated_unpromoted(self, limit=100):
            calls.append(("get_validated_unpromoted", limit))
            return []

        def close(self):
            pass

    original_database = mcp_server._database
    mcp_server._database = lambda: _FakeDb()
    try:
        result = handle_call("xpf_promote", {"limit": 10})
        assert result["ok"] is True
        called_methods = [name for name, _ in calls]
        assert "get_validated_unpromoted" in called_methods
        assert "get_unvalidated" not in called_methods
    finally:
        mcp_server._database = original_database
