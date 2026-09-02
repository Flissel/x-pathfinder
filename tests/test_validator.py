from x_pathfinder.validator import EvidenceValidator, EmailValidator, Verdict


def _fetcher(pages):
    def _fetch(url: str):
        if url not in pages:
            return 404, ""
        return 200, pages[url]

    return _fetch


def test_reachable_url_containing_the_claim_validates():
    validator = EvidenceValidator(
        fetcher=_fetcher({"https://a.example": "Acme Corp raised a Series A round"})
    )
    verdict = validator.validate("acme", ["https://a.example"], ["acme", "raised"])

    assert isinstance(verdict, Verdict)
    assert verdict.validated is True


def test_404_evidence_url_is_refuted():
    """The core anti-self-report test: a dead source proves nothing."""
    validator = EvidenceValidator(fetcher=_fetcher({}))
    verdict = validator.validate("acme", ["https://gone.example"], ["acme"])

    assert verdict.validated is False
    assert "unreachable" in verdict.reason


def test_reachable_url_without_the_claim_is_refuted():
    validator = EvidenceValidator(
        fetcher=_fetcher({"https://a.example": "An unrelated page about cats"})
    )
    verdict = validator.validate("acme", ["https://a.example"], ["acme", "raised"])

    assert verdict.validated is False
    assert "claim not found" in verdict.reason


def test_no_evidence_urls_is_refuted_not_passed():
    validator = EvidenceValidator(fetcher=_fetcher({}))
    verdict = validator.validate("acme", [], ["acme"])

    assert verdict.validated is False
    assert "no evidence" in verdict.reason


def test_fetcher_exception_fails_closed():
    def _boom(url: str):
        raise OSError("dns failure")

    verdict = EvidenceValidator(fetcher=_boom).validate("acme", ["https://a.example"], ["acme"])
    assert verdict.validated is False


def test_one_good_source_among_several_is_enough():
    validator = EvidenceValidator(
        fetcher=_fetcher({"https://good.example": "Acme raised money"})
    )
    verdict = validator.validate(
        "acme", ["https://dead.example", "https://good.example"], ["acme", "raised"]
    )
    assert verdict.validated is True


def test_email_with_mx_and_smtp_accept_validates():
    verdict = EmailValidator(lambda d: True, lambda e: True).validate("a@b.example")
    assert verdict.validated is True


def test_email_without_mx_is_refuted():
    verdict = EmailValidator(lambda d: False, lambda e: True).validate("a@b.example")
    assert verdict.validated is False
    assert "no MX record" in verdict.reason


def test_unknown_smtp_result_fails_closed():
    """Unknown is not a pass. This mirrors the unscored-vs-zero rule."""
    verdict = EmailValidator(lambda d: True, lambda e: None).validate("a@b.example")
    assert verdict.validated is False
    assert "failing closed" in verdict.reason


def test_malformed_email_is_refuted():
    verdict = EmailValidator(lambda d: True, lambda e: True).validate("not-an-email")
    assert verdict.validated is False
