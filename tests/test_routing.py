"""Rule-based review routing, tested on hand-built claim decisions (no model)."""

from radtrace.routing import is_positive_finding, route_review_priority
from radtrace.types import AuditResult, ClaimAudit, ErrorType, Label, RiskTier
from radtrace.verifier import RadTraceVerifier

NORMAL_SOURCE = (
    "The heart size and mediastinal contours are within normal limits. The lungs are clear. "
    "There is no pleural effusion or pneumothorax."
)


def claim(text: str, label: Label, *error_types: ErrorType) -> ClaimAudit:
    return ClaimAudit(
        claim_id="C1",
        claim=text,
        label=label,
        confidence=0.9,
        probabilities={label.value: 0.9} if label != Label.REVIEW else {},
        risk_tier=RiskTier.LOW,
        error_types=list(error_types),
    )


def audit(findings: str, impression: str, claims: list[ClaimAudit]) -> AuditResult:
    return AuditResult(
        findings=findings,
        impression=impression,
        claims=claims,
        summary=RadTraceVerifier._summarize(claims),
    )


def route(source: str, summary: str, commission: list[ClaimAudit], omission: list[ClaimAudit]):
    return route_review_priority(audit(source, summary, commission), audit(summary, source, omission))


def test_positive_finding_detection():
    assert is_positive_finding("Small left pneumothorax.")
    assert is_positive_finding("Possible right lower lobe pneumonia.")
    assert not is_positive_finding("No pleural effusion or pneumothorax.")
    assert not is_positive_finding("The lungs are clear.")


def test_concise_faithful_summary_is_routine():
    summary = "No acute cardiopulmonary process."
    decision = route(
        NORMAL_SOURCE,
        summary,
        [claim(summary, Label.ENTAILED)],
        [
            claim("The heart size and mediastinal contours are within normal limits.", Label.ENTAILED),
            claim("The lungs are clear.", Label.UNSUPPORTED),
            # Model noise on a normal statement must not raise the priority.
            claim("There is no pleural effusion or pneumothorax.", Label.CONTRADICTED),
        ],
    )
    assert decision.priority == "ROUTINE"
    assert decision.reasons == ()


def test_contradiction_is_high():
    decision = route(
        "Small left pneumothorax.",
        "Small right pneumothorax.",
        [claim("Small right pneumothorax.", Label.CONTRADICTED, ErrorType.LATERALITY)],
        [],
    )
    assert decision.priority == "HIGH"
    assert "laterality" in decision.reasons[0]


def test_structured_conflict_on_unsupported_claim_is_high():
    decision = route(
        "Moderate left pleural effusion.",
        "Large left pleural effusion.",
        [claim("Large left pleural effusion.", Label.UNSUPPORTED, ErrorType.SEVERITY, ErrorType.UNSUPPORTED)],
        [],
    )
    assert decision.priority == "HIGH"


def test_invented_finding_is_high_and_unsupported_recommendation_is_review():
    invented = route(
        NORMAL_SOURCE,
        "A 2 cm right upper lobe mass is present.",
        [claim("A 2 cm right upper lobe mass is present.", Label.UNSUPPORTED)],
        [],
    )
    assert invented.priority == "HIGH"
    assert "mass" in invented.reasons[0]

    recommendation = route(
        NORMAL_SOURCE,
        "Recommend follow-up in six weeks.",
        [claim("Recommend follow-up in six weeks.", Label.UNSUPPORTED)],
        [],
    )
    assert recommendation.priority == "REVIEW"


def test_abstention_is_review():
    decision = route(NORMAL_SOURCE, "Heart is normal.", [claim("Heart is normal.", Label.REVIEW)], [])
    assert decision.priority == "REVIEW"


def test_negative_claim_matching_source_absence_is_not_high():
    decision = route(
        "No pneumothorax or pleural effusion.",
        "No pleural effusion.",
        [claim("No pleural effusion.", Label.CONTRADICTED)],
        [],
    )
    assert decision.priority == "REVIEW"
    assert "same absence" in decision.reasons[0]


def test_omitted_critical_finding_is_high_and_other_omission_is_review():
    critical = route(
        "Small left pneumothorax. Mild cardiomegaly.",
        "Mild cardiomegaly.",
        [claim("Mild cardiomegaly.", Label.ENTAILED)],
        [claim("Small left pneumothorax.", Label.UNSUPPORTED), claim("Mild cardiomegaly.", Label.ENTAILED)],
    )
    assert critical.priority == "HIGH"
    assert "omits a critical" in critical.reasons[0]

    ordinary = route(
        "Moderate left pleural effusion. Mild cardiomegaly.",
        "Mild cardiomegaly.",
        [claim("Mild cardiomegaly.", Label.ENTAILED)],
        [claim("Moderate left pleural effusion.", Label.UNSUPPORTED)],
    )
    assert ordinary.priority == "REVIEW"


def test_finding_still_reported_in_summary_is_not_an_omission():
    decision = route(
        "There is moderate left pleural effusion.",
        "Left pleural effusion.",
        [claim("Left pleural effusion.", Label.ENTAILED)],
        [claim("There is moderate left pleural effusion.", Label.UNSUPPORTED)],
    )
    assert decision.priority == "ROUTINE"


def test_reasons_put_high_first_and_are_unique():
    decision = route(
        "Small left pneumothorax. Moderate left pleural effusion.",
        "Small right pneumothorax.",
        [claim("Small right pneumothorax.", Label.CONTRADICTED)],
        [
            claim("Moderate left pleural effusion.", Label.UNSUPPORTED),
            claim("Moderate left pleural effusion.", Label.UNSUPPORTED),
        ],
    )
    assert decision.priority == "HIGH"
    assert decision.reasons[0].startswith("Summary conflicts")
    assert len(decision.reasons) == len(set(decision.reasons)) == 2
