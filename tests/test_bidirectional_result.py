"""Saved batch results must round-trip (regression for the Dataset Lab crash)."""

from radtrace.bidirectional import (
    BidirectionalAuditResult,
    DirectionRisk,
    ReportRiskFeatures,
    combine_count_aware_risk,
)
from radtrace.types import AuditResult, ClaimAudit, Label, RiskTier
from radtrace.verifier import RadTraceVerifier


def _audit(findings: str, impression: str) -> AuditResult:
    claims = [
        ClaimAudit(
            claim_id="R00001-C001",
            claim=impression,
            label=Label.ENTAILED,
            confidence=0.97,
            probabilities={"ENTAILED": 0.97, "CONTRADICTED": 0.01, "UNSUPPORTED": 0.02},
            risk_tier=RiskTier.LOW,
        )
    ]
    return AuditResult(findings, impression, claims, RadTraceVerifier._summarize(claims))


def _result() -> BidirectionalAuditResult:
    features = ReportRiskFeatures(
        commission_risk=0.1,
        omission_risk=0.2,
        commission_maximum=0.1,
        omission_maximum=0.2,
        commission_alert_fraction=0.0,
        omission_alert_fraction=0.5,
        length_ratio_deviation=0.4,
        commission_error_mass=0.03,
        omission_error_mass=0.9,
    )
    return BidirectionalAuditResult(
        reference_report="Small left pneumothorax.",
        candidate_summary="Small left pneumothorax.",
        commission_audit=_audit("Small left pneumothorax.", "Small left pneumothorax."),
        omission_audit=_audit("Small left pneumothorax.", "Small left pneumothorax."),
        commission=DirectionRisk("commission", 0.1, 0.1, 0.1, 0.0, 1, 0, 0),
        omission=DirectionRisk("omission", 0.2, 0.2, 0.2, 0.5, 2, 1, 0),
        risk_features=features,
        raw_risk=0.15,
        count_aware_risk=0.33,
        aligned_risk=0.05,
        alignment_applied=True,
        review_priority="ROUTINE",
        priority_reasons=[],
    )


def test_round_trip_preserves_every_field():
    original = _result()
    restored = BidirectionalAuditResult.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert restored.count_aware_risk == 0.33


def test_missing_count_aware_risk_is_recomputed_from_features():
    payload = _result().to_dict()
    del payload["count_aware_risk"]
    restored = BidirectionalAuditResult.from_dict(payload)
    assert restored.count_aware_risk == combine_count_aware_risk(restored.risk_features)
    assert restored.routing_method == "clinical-rules-v1"
