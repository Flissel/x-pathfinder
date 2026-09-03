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


def test_discover_stages_its_results_and_uses_the_provider(monkeypatch):
    """Two wiring facts in one run, neither of which held before.

    1. AccountDiscoverer receives a CompositeScorer, so the GA scores
       through the provider seam instead of the dead Twitter fields.
    2. The discovered accounts are written to the stage DB. Without that,
       xpf_score could only ever see rows the email daemon left behind and
       the spec's discovery -> stage step was unwired.
    """
    from x_pathfinder import account_discoverer as ad_module
    from x_pathfinder import mcp_server
    from x_pathfinder.composite_scorer import CompositeScorer
    from x_pathfinder.models import XAccount

    discovered = [XAccount(handle="alpha", niche="ai"),
                  XAccount(handle="beta", niche="ai")]
    constructed = []
    saved = []
    closed = []

    class _FakeDiscoverer:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        async def run(self, generations=15, on_progress=None):
            return discovered

    class _FakeDb:
        def save_scored_accounts(self, accounts):
            saved.extend(accounts)
            return len(accounts)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(ad_module, "AccountDiscoverer", _FakeDiscoverer)
    monkeypatch.setattr(mcp_server, "_database", lambda: _FakeDb())

    result = handle_call("xpf_discover", {"niche": "ai", "generations": 1, "top": 1})

    assert result["ok"] is True
    assert result["result"] == {
        "discovered": 2, "staged": 2, "handles": ["alpha"],
    }
    assert [a.handle for a in saved] == ["alpha", "beta"]
    assert closed == [True], "the database handle must be closed"
    assert isinstance(constructed[0]["provider"], CompositeScorer)


def test_validate_skips_rows_with_no_evidence_instead_of_refuting_them(monkeypatch):
    """No evidence is not a refutation.

    get_unvalidated() selects validated IS NULL, so a FALSE written for a
    row that had nothing to check would exclude it from the validator
    forever — even after xpf_score gave it real evidence. Calling
    xpf_validate before xpf_score would then permanently kill every
    candidate the email daemon inserted.
    """
    from x_pathfinder import mcp_server
    from x_pathfinder import validator as validator_module

    verdicts = []

    class _FakeDb:
        def get_unvalidated(self, limit=100):
            return [
                {"handle": "hasevidence", "evidence_urls": ["https://a.example"]},
                {"handle": "emptylist", "evidence_urls": []},
                {"handle": "nullcolumn", "evidence_urls": None},
            ]

        def record_verdict(self, handle, validated, reason):
            verdicts.append((handle, validated, reason))

        def close(self):
            pass

    real_validator = validator_module.EvidenceValidator
    monkeypatch.setattr(mcp_server, "_database", lambda: _FakeDb())
    monkeypatch.setattr(
        validator_module,
        "EvidenceValidator",
        lambda *a, **k: real_validator(
            fetcher=lambda url: (200, "hasevidence is mentioned here")
        ),
    )

    result = handle_call("xpf_validate", {})

    assert result["ok"] is True
    assert result["result"] == {
        "checked": 1, "confirmed": 1, "skipped_no_evidence": 2,
    }
    # The decisive assertion: no verdict was written for the two rows that
    # had nothing to verify.
    assert [handle for handle, _, _ in verdicts] == ["hasevidence"]


def test_score_carries_the_stage_niche_through_to_the_research_scorer(monkeypatch):
    """_tool_score built XAccount(handle=...) with no niche, and
    ResearchScorer reads the niche off candidates[0] with a "general"
    fallback -- so every xpf_score researched against a generic niche.
    This drives the real CompositeScorer/ResearchScorer with stubbed IO and
    asserts on the prompt that actually goes out.
    """
    from x_pathfinder import fitness_providers as fp_module
    from x_pathfinder import mcp_server
    from x_pathfinder import research_scorer as rs_module

    prompts = []

    class _FakeDb:
        saved = None

        def get_unvalidated(self, limit=100):
            return [{"handle": "acme", "niche": "security"}]

        def save_scored_accounts(self, accounts):
            type(self).saved = list(accounts)
            return len(accounts)

        def close(self):
            pass

    def _transport(message):
        prompts.append(message)
        return ('{"results": [{"handle": "acme", "score": 90, '
                '"evidence_urls": ["https://news.example/acme"], '
                '"reason": "series A"}]}')

    real_deterministic = fp_module.DeterministicScorer
    real_research = rs_module.ResearchScorer
    monkeypatch.setattr(mcp_server, "_database", lambda: _FakeDb())
    monkeypatch.setattr(
        fp_module, "DeterministicScorer",
        lambda *a, **k: real_deterministic(resolver=lambda url: 200),
    )
    monkeypatch.setattr(
        rs_module, "ResearchScorer",
        lambda *a, **k: real_research(transport=_transport),
    )

    result = handle_call("xpf_score", {})

    assert result["ok"] is True
    assert prompts, "the research scorer was never reached"
    assert 'niche "security"' in prompts[0]
    assert 'niche "general"' not in prompts[0]

    saved = _FakeDb.saved
    assert saved[0].niche == "security"
    # Signals from both scorers survive to the stage write (FIX 6).
    assert saved[0].signals == {
        "handle_wellformed": True,
        "profile_resolves": True,
        "reason": "series A",
    }


@pytest.mark.parametrize(
    "url",
    [
        "https://x.com/acme",
        "https://X.com/ACME",
        "http://twitter.com/acme",
        "https://www.twitter.com/acme/",
        "https://mobile.twitter.com/acme",
        "https://x.com/acme/status/1234567890",
    ],
)
def test_self_citation_urls_are_rejected(url):
    from x_pathfinder.mcp_server import _is_self_citation

    assert _is_self_citation(url, "acme") is True


@pytest.mark.parametrize(
    "url",
    [
        "https://news.example/acme-raises-series-a",
        "https://x.com/someoneelse",
        "https://x.com/someoneelse/status/1",
        "https://acme.example/about",
        "https://xcom.example/acme",
    ],
)
def test_third_party_urls_are_not_treated_as_self_citation(url):
    from x_pathfinder.mcp_server import _is_self_citation

    assert _is_self_citation(url, "acme") is False


def test_a_candidate_citing_only_itself_is_not_validated(monkeypatch):
    """The research agent picks its own evidence URLs, so without this it
    can cite the candidate's own profile and self-certify. With nothing
    independent left to check, the row is skipped (unexamined), not
    refuted -- the same rule as a row with no evidence at all."""
    from x_pathfinder import mcp_server
    from x_pathfinder import validator as validator_module

    verdicts = []
    fetched = []

    class _FakeDb:
        def get_unvalidated(self, limit=100):
            return [{
                "handle": "acme",
                "evidence_urls": ["https://x.com/acme",
                                  "https://twitter.com/acme/status/9"],
            }]

        def record_verdict(self, handle, validated, reason):
            verdicts.append((handle, validated, reason))

        def close(self):
            pass

    def _fetch(url):
        fetched.append(url)
        return 200, "acme acme acme"   # would validate if it were ever reached

    real_validator = validator_module.EvidenceValidator
    monkeypatch.setattr(mcp_server, "_database", lambda: _FakeDb())
    monkeypatch.setattr(
        validator_module, "EvidenceValidator",
        lambda *a, **k: real_validator(fetcher=_fetch),
    )

    result = handle_call("xpf_validate", {})

    assert result["result"] == {
        "checked": 0, "confirmed": 0, "skipped_no_evidence": 1,
    }
    assert verdicts == [], "self-citation must not produce a verdict"
    assert fetched == [], "self-cited URLs must never even be fetched"


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
