"""Serializable domain objects used by the MIRROR-Rad pipeline.

The dataclasses in this module deliberately contain only JSON-friendly values.
That makes the core verifier usable from the command line, Streamlit, a REST
API, or a notebook without coupling the UI to model internals.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StrEnum(str, Enum):
    """A small Python 3.10-compatible string enum."""

    def __str__(self) -> str:  # pragma: no cover - trivial convenience
        return self.value


class Label(StrEnum):
    ENTAILED = "ENTAILED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"
    REVIEW = "REVIEW"

    @classmethod
    def coerce(cls, value: Label | str) -> Label:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper()
        aliases = {
            "ENTAILMENT": cls.ENTAILED,
            "ENTAIL": cls.ENTAILED,
            "SUPPORTED": cls.ENTAILED,
            "CONTRADICTION": cls.CONTRADICTED,
            "CONTRADICT": cls.CONTRADICTED,
            "NEUTRAL": cls.UNSUPPORTED,
            "UNKNOWN": cls.UNSUPPORTED,
            "ABSTAIN": cls.REVIEW,
        }
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


class RiskTier(StrEnum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def coerce(cls, value: RiskTier | str) -> RiskTier:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper()
        aliases = {"MEDIUM": cls.MODERATE, "MED": cls.MODERATE}
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)

    @property
    def rank(self) -> int:
        return {self.LOW: 0, self.MODERATE: 1, self.HIGH: 2, self.CRITICAL: 3}[self]


class ErrorType(StrEnum):
    NONE = "NONE"
    NEGATION = "NEGATION_OR_POLARITY"
    LATERALITY = "LATERALITY_OR_ANATOMY"
    SEVERITY = "SEVERITY_OR_SIZE"
    TEMPORAL = "TEMPORAL_CHANGE"
    MEASUREMENT = "MEASUREMENT_OR_DEVICE"
    ENTITY = "CLINICAL_ENTITY"
    UNSUPPORTED = "UNSUPPORTED_INSERTION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass(slots=True)
class TextSpan:
    start: int
    end: int
    text: str
    error_type: ErrorType = ErrorType.NONE
    role: str = "claim"

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TextSpan:
        return cls(
            start=int(value["start"]),
            end=int(value["end"]),
            text=str(value.get("text", "")),
            error_type=ErrorType(value.get("error_type", ErrorType.NONE.value)),
            role=str(value.get("role", "claim")),
        )


@dataclass(slots=True)
class EvidenceMatch:
    sentence_index: int
    text: str
    score: float
    rank: int = 1
    nli_probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvidenceMatch:
        return cls(
            sentence_index=int(value["sentence_index"]),
            text=str(value["text"]),
            score=float(value.get("score", 0.0)),
            rank=int(value.get("rank", 1)),
            nli_probabilities={str(k): float(v) for k, v in value.get("nli_probabilities", {}).items()},
        )


@dataclass(slots=True)
class ClaimAudit:
    claim_id: str
    claim: str
    label: Label
    confidence: float
    probabilities: dict[str, float]
    risk_tier: RiskTier
    error_types: list[ErrorType] = field(default_factory=list)
    evidence: list[EvidenceMatch] = field(default_factory=list)
    spans: list[TextSpan] = field(default_factory=list)
    rule_flags: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    @property
    def requires_review(self) -> bool:
        return self.label == Label.REVIEW

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ClaimAudit:
        return cls(
            claim_id=str(value["claim_id"]),
            claim=str(value["claim"]),
            label=Label.coerce(value["label"]),
            confidence=float(value.get("confidence", 0.0)),
            probabilities={str(k): float(v) for k, v in value.get("probabilities", {}).items()},
            risk_tier=RiskTier.coerce(value.get("risk_tier", RiskTier.LOW.value)),
            error_types=[ErrorType(item) for item in value.get("error_types", [])],
            evidence=[EvidenceMatch.from_dict(item) for item in value.get("evidence", [])],
            spans=[TextSpan.from_dict(item) for item in value.get("spans", [])],
            rule_flags=dict(value.get("rule_flags", {})),
            explanation=str(value.get("explanation", "")),
        )


@dataclass(slots=True)
class AuditSummary:
    total_claims: int
    label_counts: dict[str, int]
    factual_precision: float
    hallucination_rate: float
    review_rate: float
    highest_risk: RiskTier
    risk_weighted_score: float

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AuditSummary:
        return cls(
            total_claims=int(value.get("total_claims", 0)),
            label_counts={str(k): int(v) for k, v in value.get("label_counts", {}).items()},
            factual_precision=float(value.get("factual_precision", 0.0)),
            hallucination_rate=float(value.get("hallucination_rate", 0.0)),
            review_rate=float(value.get("review_rate", 0.0)),
            highest_risk=RiskTier.coerce(value.get("highest_risk", RiskTier.LOW.value)),
            risk_weighted_score=float(value.get("risk_weighted_score", 0.0)),
        )


@dataclass(slots=True)
class AuditResult:
    findings: str
    impression: str
    claims: list[ClaimAudit]
    summary: AuditSummary
    model_info: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary (including enum values)."""

        return _json_ready(asdict(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AuditResult:
        return cls(
            findings=str(value.get("findings", "")),
            impression=str(value.get("impression", "")),
            claims=[ClaimAudit.from_dict(item) for item in value.get("claims", [])],
            summary=AuditSummary.from_dict(value.get("summary", {})),
            model_info=dict(value.get("model_info", {})),
            schema_version=str(value.get("schema_version", "1.0")),
        )

    @classmethod
    def from_json(cls, value: str) -> AuditResult:
        return cls.from_dict(json.loads(value))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


THREE_WAY_LABELS: tuple[Label, ...] = (
    Label.ENTAILED,
    Label.CONTRADICTED,
    Label.UNSUPPORTED,
)
