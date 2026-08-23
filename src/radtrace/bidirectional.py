"""Bidirectional commission/omission auditing and report-risk calibration.

The claim verifier answers whether a candidate statement is supported by a
source report.  A summary can also be unsafe because it omits an important
source statement, so this module runs the verifier in both directions:

* reference -> candidate: additions, contradictions, and unsupported claims;
* candidate -> reference: source claims that the candidate omitted or changed.

The two directions remain visible in every result.  A compact report-level
risk vector may optionally be mapped to a probability by a separately fitted
radiologist-alignment calibrator.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

from .types import AuditResult, ClaimAudit, Label, RiskTier
from .verifier import RadTraceVerifier

_TIER_WEIGHT = {
    RiskTier.LOW: 0.55,
    RiskTier.MODERATE: 0.72,
    RiskTier.HIGH: 0.90,
    RiskTier.CRITICAL: 1.00,
}


def claim_error_probability(claim: ClaimAudit) -> float:
    """Return the model probability that a claim is not source-supported."""

    value = float(claim.probabilities.get(Label.CONTRADICTED.value, 0.0)) + float(
        claim.probabilities.get(Label.UNSUPPORTED.value, 0.0)
    )
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class DirectionRisk:
    direction: str
    score: float
    maximum_claim_risk: float
    mean_top_claim_risk: float
    alert_fraction: float
    claim_count: int
    alerted_claims: int
    high_priority_claims: int
    # Count-aware evidence is intentionally kept separate from ``score``.
    # ``score`` protects a single dangerous claim, while these quantities
    # preserve cumulative burden for review-queue ranking.
    error_mass: float = 0.0
    mean_error_probability: float = 0.0
    noisy_or_error_probability: float = 0.0
    high_priority_error_mass: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_direction(audit: AuditResult, direction: str) -> DirectionRisk:
    """Aggregate transparent claim probabilities without hiding claim detail."""

    if not audit.claims:
        return DirectionRisk(direction, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)
    unweighted = [claim_error_probability(claim) for claim in audit.claims]
    weighted = [
        claim_error_probability(claim) * _TIER_WEIGHT.get(claim.risk_tier, 0.72)
        for claim in audit.claims
    ]
    ordered = sorted(weighted, reverse=True)
    top = ordered[: min(3, len(ordered))]
    mean_top = sum(top) / len(top)
    maximum = ordered[0]
    # The maximum protects single safety-critical claims; the top-claim mean
    # prevents a long report from dominating merely because it has more text.
    score = max(0.0, min(1.0, 0.68 * maximum + 0.32 * mean_top))
    alerts = sum(claim.label != Label.ENTAILED for claim in audit.claims)
    high_priority = sum(
        claim.label != Label.ENTAILED
        and claim.risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL)
        for claim in audit.claims
    )
    error_mass = sum(unweighted)
    noisy_or = 1.0 - math.prod(1.0 - min(0.999999, value) for value in unweighted)
    high_priority_mass = sum(
        probability
        for probability, claim in zip(unweighted, audit.claims, strict=True)
        if claim.risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL)
    )
    return DirectionRisk(
        direction=direction,
        score=score,
        maximum_claim_risk=maximum,
        mean_top_claim_risk=mean_top,
        alert_fraction=alerts / len(audit.claims),
        claim_count=len(audit.claims),
        alerted_claims=alerts,
        high_priority_claims=high_priority,
        error_mass=error_mass,
        mean_error_probability=error_mass / len(unweighted),
        noisy_or_error_probability=noisy_or,
        high_priority_error_mass=high_priority_mass,
    )


@dataclass(slots=True)
class ReportRiskFeatures:
    commission_risk: float
    omission_risk: float
    commission_maximum: float
    omission_maximum: float
    commission_alert_fraction: float
    omission_alert_fraction: float
    length_ratio_deviation: float
    commission_error_mass: float = 0.0
    omission_error_mass: float = 0.0
    commission_mean_error_probability: float = 0.0
    omission_mean_error_probability: float = 0.0
    commission_noisy_or: float = 0.0
    omission_noisy_or: float = 0.0
    commission_high_priority_mass: float = 0.0
    omission_high_priority_mass: float = 0.0

    ORDER: ClassVar[tuple[str, ...]] = (
        "commission_risk",
        "omission_risk",
        "commission_maximum",
        "omission_maximum",
        "commission_alert_fraction",
        "omission_alert_fraction",
        "length_ratio_deviation",
        "commission_error_mass",
        "omission_error_mass",
        "commission_mean_error_probability",
        "omission_mean_error_probability",
        "commission_noisy_or",
        "omission_noisy_or",
        "commission_high_priority_mass",
        "omission_high_priority_mass",
    )

    def vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in self.ORDER]

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.ORDER}


@dataclass(slots=True)
class BidirectionalAuditResult:
    reference_report: str
    candidate_summary: str
    commission_audit: AuditResult
    omission_audit: AuditResult
    commission: DirectionRisk
    omission: DirectionRisk
    risk_features: ReportRiskFeatures
    raw_risk: float
    count_aware_risk: float
    aligned_risk: float
    alignment_applied: bool
    review_priority: str
    scope: str = "textual consistency review"

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_report": self.reference_report,
            "candidate_summary": self.candidate_summary,
            "commission_audit": self.commission_audit.to_dict(),
            "omission_audit": self.omission_audit.to_dict(),
            "commission": self.commission.to_dict(),
            "omission": self.omission.to_dict(),
            "risk_features": self.risk_features.to_dict(),
            "raw_risk": self.raw_risk,
            "count_aware_risk": self.count_aware_risk,
            "aligned_risk": self.aligned_risk,
            "alignment_applied": self.alignment_applied,
            "review_priority": self.review_priority,
            "scope": self.scope,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def combine_direction_risk(commission_risk: float, omission_risk: float) -> float:
    """Combine both audit directions into a bounded, monotonic raw score."""

    return max(
        0.0,
        min(
            1.0,
            0.55 * float(commission_risk)
            + 0.45 * float(omission_risk)
            + 0.15 * float(commission_risk) * float(omission_risk),
        ),
    )


def combine_count_aware_risk(features: ReportRiskFeatures) -> float:
    """Combine single-claim safety and cumulative factual-error burden.

    The transformation is fixed and monotonic in every component.  It was
    introduced to avoid the old max/top-three bottleneck: a report with several
    moderately suspicious omissions should rank above one isolated low-risk
    alert.  It is an ordinal review score, not a calibrated clinical
    probability.
    """

    total_mass = max(0.0, features.commission_error_mass) + max(
        0.0, features.omission_error_mass
    )
    absolute_mass = 1.0 - math.exp(-0.45 * total_mass)
    mean_mass = min(
        1.0,
        (
            max(0.0, features.commission_mean_error_probability)
            + max(0.0, features.omission_mean_error_probability)
        )
        / 2.0,
    )
    noisy_or = max(features.commission_noisy_or, features.omission_noisy_or)
    return max(0.0, min(1.0, 0.55 * absolute_mass + 0.30 * mean_mass + 0.15 * noisy_or))


class ReportRiskCalibrator:
    """Map held-out benchmark risk to a reproducible transfer score."""

    def __init__(
        self,
        *,
        random_state: int = 42,
        method: str = "standardized_l2_logistic",
    ) -> None:
        self.random_state = int(random_state)
        self.method = str(method)
        self.estimator: Any | None = None
        self.metadata: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.estimator is not None

    def fit(
        self,
        features: Sequence[ReportRiskFeatures],
        labels: Sequence[int | bool],
    ) -> ReportRiskCalibrator:
        if len(features) != len(labels) or len(features) < 10:
            raise ValueError("Calibrator requires at least ten aligned feature/label rows.")
        if len({int(value) for value in labels}) != 2:
            raise ValueError("Calibrator labels must contain both classes.")
        targets = [int(value) for value in labels]
        if self.method == "monotonic_raw_isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.estimator = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                increasing=True,
                out_of_bounds="clip",
            )
            self.estimator.fit([self._raw_score(row) for row in features], targets)
        elif self.method == "nonnegative_two_feature_logistic":
            import numpy as np
            from scipy.optimize import minimize

            matrix = np.asarray(
                [[row.commission_risk, row.omission_risk] for row in features],
                dtype=float,
            )
            target_array = np.asarray(targets, dtype=float)
            positive_weight = len(targets) / (2.0 * max(1, sum(targets)))
            negative_weight = len(targets) / (2.0 * max(1, len(targets) - sum(targets)))
            sample_weight = np.where(target_array == 1.0, positive_weight, negative_weight)

            def objective(parameters: Any) -> float:
                logits = parameters[0] + matrix @ parameters[1:]
                losses = np.logaddexp(0.0, logits) - target_array * logits
                regularization = 0.025 * float(np.square(parameters[1:]).sum())
                return float(np.average(losses, weights=sample_weight) + regularization)

            fitted = minimize(
                objective,
                x0=np.asarray([0.0, 1.0, 1.0], dtype=float),
                method="L-BFGS-B",
                bounds=[(None, None), (0.0, None), (0.0, None)],
            )
            if not fitted.success:
                raise RuntimeError(f"Non-negative risk fit failed: {fitted.message}")
            self.estimator = {
                "intercept": float(fitted.x[0]),
                "commission_weight": float(fitted.x[1]),
                "omission_weight": float(fitted.x[2]),
            }
        elif self.method == "standardized_l2_logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            self.estimator = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            C=0.5,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )
            self.estimator.fit([row.vector() for row in features], targets)
        else:
            raise ValueError(f"Unsupported risk calibration method: {self.method}")
        self.metadata = {
            "feature_order": list(ReportRiskFeatures.ORDER),
            "fit_rows": len(features),
            "positive_rows": sum(targets),
            "random_state": self.random_state,
            "method": self.method,
        }
        if self.method == "nonnegative_two_feature_logistic":
            self.metadata["coefficients"] = dict(self.estimator)
        return self

    @staticmethod
    def _raw_score(feature: ReportRiskFeatures) -> float:
        return combine_direction_risk(feature.commission_risk, feature.omission_risk)

    def predict(self, features: Sequence[ReportRiskFeatures]) -> list[float]:
        if self.estimator is None:
            raise RuntimeError("Radiologist risk calibrator is not fitted.")
        if self.method == "monotonic_raw_isotonic":
            probabilities = self.estimator.predict([self._raw_score(row) for row in features])
        elif self.method == "nonnegative_two_feature_logistic":
            probabilities = []
            for row in features:
                logit = (
                    float(self.estimator["intercept"])
                    + float(self.estimator["commission_weight"]) * row.commission_risk
                    + float(self.estimator["omission_weight"]) * row.omission_risk
                )
                probabilities.append(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit)))))
        else:
            probabilities = self.estimator.predict_proba([row.vector() for row in features])[:, 1]
        return [max(0.0, min(1.0, float(value))) for value in probabilities]

    def predict_one(self, features: ReportRiskFeatures) -> float:
        return self.predict([features])[0]

    def save(self, path: str | Path) -> Path:
        if self.estimator is None:
            raise RuntimeError("Cannot save an unfitted calibrator.")
        import joblib

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "metadata": self.metadata,
                "random_state": self.random_state,
                "method": self.method,
            },
            destination,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ReportRiskCalibrator:
        import joblib

        payload = joblib.load(Path(path))
        method = str(
            payload.get("method")
            or payload.get("metadata", {}).get("method")
            or "standardized_l2_logistic"
        )
        instance = cls(random_state=int(payload.get("random_state", 42)), method=method)
        instance.estimator = payload["estimator"]
        instance.metadata = dict(payload.get("metadata", {}))
        return instance


class BidirectionalAuditor:
    """Run a real claim verifier in both safety-relevant directions."""

    def __init__(
        self,
        verifier: RadTraceVerifier,
        *,
        calibrator: ReportRiskCalibrator | None = None,
        review_threshold: float | None = None,
        high_priority_threshold: float | None = None,
    ) -> None:
        self.verifier = verifier
        self.calibrator = calibrator
        metadata = calibrator.metadata if calibrator is not None else {}
        self.review_threshold = float(
            review_threshold
            if review_threshold is not None
            else metadata.get("review_threshold", 0.50)
        )
        self.high_priority_threshold = float(
            high_priority_threshold
            if high_priority_threshold is not None
            else metadata.get("high_priority_threshold", 0.75)
        )

    def audit(self, reference_report: str, candidate_summary: str) -> BidirectionalAuditResult:
        return self.audit_many([(reference_report, candidate_summary)])[0]

    def audit_many(
        self,
        report_pairs: Sequence[tuple[str, str]],
        *,
        nli_batch_size: int = 32,
    ) -> list[BidirectionalAuditResult]:
        normalized = [
            ((reference or "").strip(), (candidate or "").strip())
            for reference, candidate in report_pairs
        ]
        forward = self.verifier.audit_many(normalized, nli_batch_size=nli_batch_size)
        reverse = self.verifier.audit_many(
            [(candidate, reference) for reference, candidate in normalized],
            nli_batch_size=nli_batch_size,
        )
        features: list[ReportRiskFeatures] = []
        summaries: list[tuple[DirectionRisk, DirectionRisk]] = []
        for (reference, candidate), commission_audit, omission_audit in zip(
            normalized,
            forward,
            reverse,
            strict=True,
        ):
            commission = summarize_direction(commission_audit, "commission")
            omission = summarize_direction(omission_audit, "omission")
            reference_words = max(1, len(reference.split()))
            candidate_words = max(1, len(candidate.split()))
            length_deviation = min(3.0, abs(math.log(candidate_words / reference_words))) / 3.0
            features.append(
                ReportRiskFeatures(
                    commission_risk=commission.score,
                    omission_risk=omission.score,
                    commission_maximum=commission.maximum_claim_risk,
                    omission_maximum=omission.maximum_claim_risk,
                    commission_alert_fraction=commission.alert_fraction,
                    omission_alert_fraction=omission.alert_fraction,
                    length_ratio_deviation=length_deviation,
                    commission_error_mass=commission.error_mass,
                    omission_error_mass=omission.error_mass,
                    commission_mean_error_probability=commission.mean_error_probability,
                    omission_mean_error_probability=omission.mean_error_probability,
                    commission_noisy_or=commission.noisy_or_error_probability,
                    omission_noisy_or=omission.noisy_or_error_probability,
                    commission_high_priority_mass=commission.high_priority_error_mass,
                    omission_high_priority_mass=omission.high_priority_error_mass,
                )
            )
            summaries.append((commission, omission))

        calibrated = (
            self.calibrator.predict(features)
            if self.calibrator is not None and self.calibrator.is_fitted
            else [None] * len(features)
        )
        output: list[BidirectionalAuditResult] = []
        for index, ((reference, candidate), commission_audit, omission_audit, feature) in enumerate(
            zip(normalized, forward, reverse, features, strict=True)
        ):
            commission, omission = summaries[index]
            raw = combine_direction_risk(commission.score, omission.score)
            count_aware = combine_count_aware_risk(feature)
            aligned = float(calibrated[index]) if calibrated[index] is not None else raw
            if aligned >= self.high_priority_threshold:
                priority = "HIGH"
            elif aligned >= self.review_threshold:
                priority = "REVIEW"
            else:
                priority = "ROUTINE"
            output.append(
                BidirectionalAuditResult(
                    reference_report=reference,
                    candidate_summary=candidate,
                    commission_audit=commission_audit,
                    omission_audit=omission_audit,
                    commission=commission,
                    omission=omission,
                    risk_features=feature,
                    raw_risk=raw,
                    count_aware_risk=count_aware,
                    aligned_risk=aligned,
                    alignment_applied=calibrated[index] is not None,
                    review_priority=priority,
                )
            )
        return output
