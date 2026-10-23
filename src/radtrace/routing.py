"""Transparent review routing over claim-level decisions.

The review priority shown to users is decided by explicit rules over the claim
decisions in both audit directions instead of a learned report-level router.

Why: the earlier ReXErr-fitted router was trained on report pairs of almost
identical length.  A real Impression is much shorter than its Findings, which
put the router's length feature far outside its training range, so concise
faithful summaries were routed HIGH while some invented findings were routed
ROUTINE.  Rules keep every decision explainable and independent of length.

Priority rules
--------------
HIGH
    * a summary claim conflicts with the source (CONTRADICTED, or a non-entailed
      claim with a structured conflict such as negation, laterality, severity,
      temporal change, measurement or device detail);
    * a summary claim reports a finding or device that the source does not
      support (an invented finding);
    * a summary claim conflicts with a positive source finding it mentions, or
      the summary omits a positive critical finding (for example pneumothorax
      or a line or tube).
REVIEW
    * any other unsupported or low-confidence summary claim (for example a
      recommendation that the source does not contain);
    * a positive, non-critical source finding that the summary does not
      preserve;
    * a model conflict flag on a negative summary claim whose absence the
      source also states.
ROUTINE
    * everything else.  Normal or negative source statements ("lungs are clear",
      "no pneumothorax") that a concise impression does not restate do not raise
      the priority; they stay visible in the omission view.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .clinical_rules import CRITICAL_ENTITIES, extract_entities
from .types import AuditResult, ClaimAudit, ErrorType, Label

ROUTING_METHOD = "clinical-rules-v1"
PRIORITY_ORDER = {"ROUTINE": 0, "REVIEW": 1, "HIGH": 2}

_CONFLICT_TYPES = frozenset(
    {
        ErrorType.NEGATION,
        ErrorType.LATERALITY,
        ErrorType.SEVERITY,
        ErrorType.TEMPORAL,
        ErrorType.MEASUREMENT,
    }
)
_MAX_REASON_CLAIM = 90


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    priority: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TextEntities:
    mentioned: frozenset[str]
    present: frozenset[str]
    negated: frozenset[str]

    @classmethod
    def of(cls, text: str) -> _TextEntities:
        mentions = extract_entities(text or "")
        return cls(
            mentioned=frozenset(item.entity for item in mentions),
            present=frozenset(item.entity for item in mentions if not item.negated),
            negated=frozenset(item.entity for item in mentions if item.negated),
        )


def positive_entities(text: str) -> set[str]:
    """Return entities the text asserts as present (uncertain mentions count)."""

    return set(_TextEntities.of(text).present)


def is_positive_finding(text: str) -> bool:
    """True when a statement reports at least one present (non-negated) finding."""

    return bool(_TextEntities.of(text).present)


def _quote(claim: str) -> str:
    value = " ".join(claim.split())
    if len(value) > _MAX_REASON_CLAIM:
        value = value[: _MAX_REASON_CLAIM - 1].rstrip() + "…"
    return f"“{value}”"


def _conflict_names(claim: ClaimAudit) -> list[str]:
    return sorted(
        item.value.replace("_", " ").lower() for item in claim.error_types if item in _CONFLICT_TYPES
    )


def _commission_decision(claim: ClaimAudit, source: _TextEntities) -> tuple[str, str] | None:
    if claim.label == Label.ENTAILED:
        return None
    entities = _TextEntities.of(claim.claim)
    conflicts = _conflict_names(claim)
    if claim.label == Label.CONTRADICTED and not conflicts and not entities.present:
        # A negative claim ("No pleural effusion.") whose absence the source
        # also states is consistent by rule even if the model flags a conflict.
        if entities.negated and entities.negated <= source.negated:
            return "REVIEW", (
                "Model flags a conflict, but the source states the same absence: "
                f"{_quote(claim.claim)}"
            )
    if claim.label == Label.CONTRADICTED or conflicts:
        detail = f" ({', '.join(conflicts)})" if conflicts else ""
        return "HIGH", f"Summary conflicts with the source{detail}: {_quote(claim.claim)}"
    if claim.label == Label.UNSUPPORTED and entities.present:
        names = ", ".join(sorted(entities.present))
        return "HIGH", f"Summary reports a finding the source does not support ({names}): {_quote(claim.claim)}"
    if claim.label == Label.UNSUPPORTED:
        return "REVIEW", f"Summary claim is not supported by the source: {_quote(claim.claim)}"
    return "REVIEW", f"Model is unsure whether the source supports: {_quote(claim.claim)}"


def _omission_decision(claim: ClaimAudit, summary: _TextEntities) -> tuple[str, str] | None:
    if claim.label == Label.ENTAILED:
        return None
    present = _TextEntities.of(claim.claim).present
    if not present:
        # A normal or negative source statement a concise impression may omit.
        return None
    if claim.label == Label.CONTRADICTED and present & summary.mentioned:
        return "HIGH", f"Summary conflicts with a source finding: {_quote(claim.claim)}"
    if present <= summary.present:
        # The summary still reports every finding in this statement, so it is
        # not an omission; attribute changes are judged in the summary→source
        # direction.
        return None
    critical = sorted(present & CRITICAL_ENTITIES)
    if critical:
        return "HIGH", (
            f"Summary omits a critical source finding ({', '.join(critical)}): {_quote(claim.claim)}"
        )
    return "REVIEW", f"Summary omits a source finding: {_quote(claim.claim)}"


def _collect(
    claims: Iterable[ClaimAudit],
    decide: Callable[[ClaimAudit, _TextEntities], tuple[str, str] | None],
    context: _TextEntities,
) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for claim in claims:
        decision = decide(claim, context)
        if decision is not None:
            output.append(decision)
    return output


def route_review_priority(commission: AuditResult, omission: AuditResult) -> RoutingDecision:
    """Decide the report review priority and the human-readable reasons for it.

    ``commission`` audits summary claims against the source report and
    ``omission`` audits source statements against the summary, exactly as
    produced by :class:`radtrace.bidirectional.BidirectionalAuditor`.
    """

    source = _TextEntities.of(commission.findings)
    summary = _TextEntities.of(commission.impression)
    decisions = _collect(commission.claims, _commission_decision, source) + _collect(
        omission.claims, _omission_decision, summary
    )
    if not decisions:
        return RoutingDecision("ROUTINE", ())
    priority = max((item[0] for item in decisions), key=PRIORITY_ORDER.__getitem__)
    ordered = sorted(decisions, key=lambda item: -PRIORITY_ORDER[item[0]])
    reasons: list[str] = []
    for _, reason in ordered:
        if reason not in reasons:
            reasons.append(reason)
    return RoutingDecision(priority, tuple(reasons))
