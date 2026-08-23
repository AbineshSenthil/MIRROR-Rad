"""Transparent clinical consistency features for MIRROR-Rad.

These rules are not a diagnostic ontology.  They are deliberately conservative
text-comparison signals that complement NLI and make error categories visible
to a reviewer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .retrieval import tokenize
from .types import ErrorType, RiskTier, TextSpan

ENTITY_PATTERNS: dict[str, tuple[str, ...]] = {
    "pneumothorax": ("pneumothorax", "pneumothoraces", "ptx"),
    "pleural effusion": (
        "pleural effusions",
        "pleural effusion",
        "effusions",
        "effusion",
        "pleural fluid",
    ),
    "consolidation": ("consolidation", "airspace disease", "airspace opacity"),
    "opacity": ("opacity", "opacification"),
    "atelectasis": ("atelectasis", "atelectatic"),
    "edema": (
        "congestive heart failure",
        "heart failure",
        "pulmonary edema",
        "vascular congestion",
        "edema",
        "chf",
    ),
    "cardiomegaly": ("cardiomegaly", "cardiac enlargement", "enlarged heart"),
    "infiltrate": ("infiltrate", "infiltration"),
    "nodule": ("pulmonary nodules", "pulmonary nodule", "nodules", "nodule", "nodular density"),
    "mass": ("masses", "mass", "lesions", "lesion"),
    "fracture": ("fractures", "fracture", "osseous injuries", "osseous injury"),
    "free air": ("free air", "pneumoperitoneum"),
    "infection": ("infection", "infectious process", "pneumonia"),
    "hyperinflation": ("hyperinflation", "hyperinflated"),
    "fibrosis": ("fibrosis", "fibrotic", "scarring", "scar"),
    "granuloma": ("granuloma", "granulomatous"),
    "hiatal hernia": ("hiatal hernia",),
    "endotracheal tube": ("endotracheal tube", "et tube", "ett"),
    "enteric tube": ("enteric tube", "feeding tube", "nasogastric tube", "ng tube"),
    "central venous catheter": ("central venous catheter", "central line", "picc"),
    "chest tube": ("chest tube", "thoracostomy tube"),
    "pacemaker": ("pacemaker", "pacing device"),
}

CONTRASTING_ENTITY_PAIRS = {
    frozenset(("atelectasis", "consolidation")),
    frozenset(("atelectasis", "infection")),
    frozenset(("fibrosis", "consolidation")),
    frozenset(("edema", "infection")),
    frozenset(("pneumothorax", "pleural effusion")),
    frozenset(("nodule", "consolidation")),
    frozenset(("nodule", "infiltrate")),
    frozenset(("mass", "consolidation")),
    frozenset(("mass", "edema")),
    frozenset(("cardiomegaly", "pneumothorax")),
    frozenset(("fracture", "pleural effusion")),
}

CRITICAL_ENTITIES = {"pneumothorax", "free air", "endotracheal tube", "central venous catheter", "chest tube"}

NEGATION_TERMS = {"no", "not", "without", "absent", "neither", "negative", "free"}
UNCERTAINTY_TERMS = {"possible", "possibly", "probable", "may", "might", "suggest", "suggests", "cannot", "questionable"}
LATERALITY = {"left", "right", "bilateral"}
ANATOMY = {"upper", "lower", "middle", "apical", "basilar", "base", "bases", "hilar", "perihilar"}
SEVERITY_LEVEL = {
    "trace": 0,
    "minimal": 0,
    "tiny": 0,
    "small": 1,
    "mild": 1,
    "moderate": 2,
    "large": 3,
    "severe": 3,
    "marked": 3,
    "extensive": 3,
    "massive": 4,
}
TEMPORAL_CLASS = {
    "new": "new",
    "developed": "new",
    "interval": "change",
    "increased": "worse",
    "increasing": "worse",
    "worsened": "worse",
    "worsening": "worse",
    "progressed": "worse",
    "progression": "worse",
    "progressing": "worse",
    "increase": "worse",
    "more": "worse",
    "decreased": "better",
    "decreasing": "better",
    "improved": "better",
    "improving": "better",
    "improvement": "better",
    "resolving": "better",
    "decrease": "better",
    "less": "better",
    "resolved": "resolved",
    "recurred": "new",
    "stable": "stable",
    "unchanged": "stable",
    "persistent": "stable",
    "chronic": "stable",
    "old": "stable",
    "remote": "stable",
    "age-indeterminate": "stable",
}

_MEASUREMENT_RE = re.compile(r"(?<!\w)(\d+(?:\.\d+)?)\s*(mm|cm|m)(?!\w)", re.IGNORECASE)


@dataclass(slots=True)
class EntityMention:
    entity: str
    start: int
    end: int
    text: str
    negated: bool = False
    uncertain: bool = False
    laterality: frozenset[str] = frozenset()
    anatomy: frozenset[str] = frozenset()


@dataclass(slots=True)
class RuleAssessment:
    features: dict[str, float]
    violations: list[ErrorType] = field(default_factory=list)
    spans: list[TextSpan] = field(default_factory=list)
    risk_tier: RiskTier = RiskTier.LOW
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def has_contradiction(self) -> bool:
        return any(
            item in self.violations
            for item in (
                ErrorType.NEGATION,
                ErrorType.LATERALITY,
                ErrorType.SEVERITY,
                ErrorType.TEMPORAL,
                ErrorType.MEASUREMENT,
            )
        )


def extract_entities(text: str) -> list[EntityMention]:
    """Extract a compact set of radiology entities and their local polarity."""

    value = text or ""
    lower = value.lower()
    mentions: list[EntityMention] = []
    occupied: list[tuple[int, int]] = []
    # Long aliases first prevents "effusion" duplicating "pleural effusion".
    candidates: list[tuple[str, str]] = []
    for entity, aliases in ENTITY_PATTERNS.items():
        candidates.extend((entity, alias) for alias in aliases)
    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    for entity, alias in candidates:
        pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
        for match in pattern.finditer(value):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            prefix = lower[max(0, match.start() - 55) : match.start()]
            prefix_tokens = tokenize(prefix)[-8:]
            negated = _is_negated(prefix, prefix_tokens)
            uncertain = any(token in UNCERTAINTY_TERMS for token in prefix_tokens)
            mentions.append(
                EntityMention(
                    entity,
                    match.start(),
                    match.end(),
                    match.group(0),
                    negated,
                    uncertain,
                    _nearest_markers(lower, match.start(), match.end(), LATERALITY),
                    _nearest_markers(lower, match.start(), match.end(), ANATOMY),
                )
            )
            occupied.append((match.start(), match.end()))
    return sorted(mentions, key=lambda item: item.start)


def _is_negated(prefix: str, prefix_tokens: list[str]) -> bool:
    if any(token in NEGATION_TERMS for token in prefix_tokens[-6:]):
        return True
    return bool(re.search(r"(?:absence|lack)\s+of\s+(?:\w+\s+){0,4}$", prefix))


def _nearest_markers(
    text: str,
    entity_start: int,
    entity_end: int,
    vocabulary: set[str],
    *,
    radius: int = 48,
) -> frozenset[str]:
    candidates: list[tuple[int, str]] = []
    start = max(0, entity_start - radius)
    end = min(len(text), entity_end + radius)
    for marker in vocabulary:
        for match in re.finditer(rf"\b{re.escape(marker)}\b", text[start:end], re.IGNORECASE):
            absolute_start = start + match.start()
            absolute_end = start + match.end()
            distance = min(abs(entity_start - absolute_end), abs(absolute_start - entity_end))
            candidates.append((distance, marker))
    if not candidates:
        return frozenset()
    nearest_distance = min(item[0] for item in candidates)
    return frozenset(marker for distance, marker in candidates if distance <= nearest_distance + 2)


def _entity_state(mentions: Iterable[EntityMention]) -> dict[str, EntityMention]:
    # Retain the nearest/last mention, the common report convention when a
    # finding is subsequently qualified.
    return {mention.entity: mention for mention in mentions}


def _marker_set(text: str, vocabulary: set[str]) -> set[str]:
    return set(tokenize(text)) & vocabulary


def _measurement_values(text: str) -> list[tuple[float, str, int, int, str]]:
    values: list[tuple[float, str, int, int, str]] = []
    for match in _MEASUREMENT_RE.finditer(text or ""):
        amount = float(match.group(1))
        unit = match.group(2).lower()
        millimetres = amount * {"mm": 1.0, "cm": 10.0, "m": 1000.0}[unit]
        values.append((millimetres, unit, match.start(), match.end(), match.group(0)))
    return values


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _modifier_alignment(left: str, right: str, modifiers: set[str]) -> float:
    """Compare phrases after masking the modifier that may have been changed."""

    def masked_tokens(value: str) -> list[str]:
        return ["<modifier>" if token in modifiers else token for token in tokenize(value)]

    return SequenceMatcher(None, masked_tokens(left), masked_tokens(right), autojunk=False).ratio()


def _generic_acute_positive(text: str) -> bool:
    """Detect positive generic claims created by removing a safety negation."""

    value = text or ""
    patterns = (
        (
            r"\b(?:acute|active)\s+(?:(?:cardio)?pulmonary\s+)?"
            r"(?:abnormalit(?:y|ies)|disease|findings?|process)\b"
        ),
        r"\bevidence\s+of\s+active\s+disease\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, value, re.IGNORECASE):
            prefix_tokens = tokenize(value[max(0, match.start() - 45) : match.start()])[-7:]
            if not any(token in NEGATION_TERMS for token in prefix_tokens):
                return True
    return False


def _reassuring_evidence(text: str, mentions: list[EntityMention]) -> bool:
    """Return true when a sentence explicitly describes no active disease."""

    active_entities = {
        "pneumothorax",
        "pleural effusion",
        "consolidation",
        "opacity",
        "atelectasis",
        "edema",
        "infiltrate",
        "nodule",
        "mass",
        "free air",
        "infection",
    }
    positive_active = any(
        mention.entity in active_entities and not mention.negated and not mention.uncertain
        for mention in mentions
    )
    reassuring_language = bool(
        re.search(
            r"\b(?:no|without)\b|\b(?:lungs?|chest)\s+(?:are|is|appear)?\s*clear\b|"
            r"\b(?:normal|unremarkable|within normal limits)\b|\bno acute\b",
            text or "",
            re.IGNORECASE,
        )
    )
    return reassuring_language and not positive_active


def assess_clinical_consistency(claim: str, evidence: str) -> RuleAssessment:
    """Compare a claim with one evidence sentence using inspectable rules."""

    claim_mentions = extract_entities(claim)
    evidence_mentions = extract_entities(evidence)
    claim_states = _entity_state(claim_mentions)
    evidence_states = _entity_state(evidence_mentions)
    claim_entities = set(claim_states)
    evidence_entities = set(evidence_states)
    shared_entities = claim_entities & evidence_entities
    unsupported_entities = claim_entities - evidence_entities
    lexical_similarity = _jaccard(claim, evidence)
    entity_overlap = (
        len(shared_entities) / len(claim_entities) if claim_entities else lexical_similarity
    )
    unsupported_ratio = len(unsupported_entities) / max(1, len(claim_entities))
    substitution_pairs = sorted(
        (claim_entity, evidence_entity)
        for claim_entity in unsupported_entities
        for evidence_entity in (evidence_entities - claim_entities)
        if frozenset((claim_entity, evidence_entity)) in CONTRASTING_ENTITY_PAIRS
    )
    entity_substitution_mismatch = bool(substitution_pairs and lexical_similarity >= 0.18)

    negation_entities = sorted(
        entity for entity in shared_entities
        if claim_states[entity].negated != evidence_states[entity].negated
    )
    global_polarity_mismatch = _generic_acute_positive(claim) and _reassuring_evidence(
        evidence, evidence_mentions
    )
    uncertainty_entities = sorted(
        entity for entity in shared_entities
        if claim_states[entity].uncertain != evidence_states[entity].uncertain
    )

    claim_laterality = _marker_set(claim, LATERALITY)
    evidence_laterality = _marker_set(evidence, LATERALITY)
    laterality_entities = sorted(
        entity
        for entity in shared_entities
        if claim_states[entity].laterality
        and evidence_states[entity].laterality
        and claim_states[entity].laterality.isdisjoint(evidence_states[entity].laterality)
    )
    claim_laterality_profile = Counter(token for token in tokenize(claim) if token in LATERALITY)
    evidence_laterality_profile = Counter(token for token in tokenize(evidence) if token in LATERALITY)
    laterality_alignment = _modifier_alignment(claim, evidence, LATERALITY)
    profile_laterality_mismatch = bool(
        claim_laterality_profile
        and evidence_laterality_profile
        and claim_laterality_profile != evidence_laterality_profile
        and (
            (claim_laterality.isdisjoint(evidence_laterality) and laterality_alignment >= 0.42)
            or laterality_alignment >= 0.72
        )
    )
    laterality_mismatch = bool(laterality_entities) or profile_laterality_mismatch
    anatomy_entities = sorted(
        entity
        for entity in shared_entities
        if claim_states[entity].anatomy
        and evidence_states[entity].anatomy
        and claim_states[entity].anatomy.isdisjoint(evidence_states[entity].anatomy)
    )
    anatomy_mismatch = bool(anatomy_entities)

    claim_severity_values = [SEVERITY_LEVEL[token] for token in tokenize(claim) if token in SEVERITY_LEVEL]
    evidence_severity_values = [
        SEVERITY_LEVEL[token] for token in tokenize(evidence) if token in SEVERITY_LEVEL
    ]
    claim_severity = set(claim_severity_values)
    evidence_severity = set(evidence_severity_values)
    severity_distance = 0
    if claim_severity_values and evidence_severity_values:
        if len(claim_severity_values) == len(evidence_severity_values):
            severity_distance = max(
                abs(left - right)
                for left, right in zip(claim_severity_values, evidence_severity_values, strict=True)
            )
        else:
            claim_to_evidence = max(
                min(abs(left - right) for right in evidence_severity) for left in claim_severity
            )
            evidence_to_claim = max(
                min(abs(right - left) for left in claim_severity) for right in evidence_severity
            )
            severity_distance = max(claim_to_evidence, evidence_to_claim)
    severity_mismatch = bool(severity_distance >= 2)

    claim_temporal = {TEMPORAL_CLASS[token] for token in tokenize(claim) if token in TEMPORAL_CLASS}
    evidence_temporal = {TEMPORAL_CLASS[token] for token in tokenize(evidence) if token in TEMPORAL_CLASS}
    incompatible_temporal = {
        ("worse", "better"), ("better", "worse"), ("new", "stable"),
        ("new", "resolved"), ("resolved", "new"), ("resolved", "worse"),
        ("worse", "stable"), ("better", "stable"), ("stable", "worse"),
        ("stable", "better"), ("stable", "new"), ("resolved", "stable"),
    }
    temporal_mismatch = bool(
        any((left, right) in incompatible_temporal for left in claim_temporal for right in evidence_temporal)
    )

    claim_measurements = _measurement_values(claim)
    evidence_measurements = _measurement_values(evidence)
    measurement_relative_error = 0.0
    measurement_mismatch = False
    if claim_measurements and evidence_measurements:
        measurement_relative_error = min(
            abs(left[0] - right[0]) / max(left[0], right[0], 1.0)
            for left in claim_measurements for right in evidence_measurements
        )
        measurement_mismatch = measurement_relative_error > 0.10

    violations: list[ErrorType] = []
    if negation_entities or global_polarity_mismatch:
        violations.append(ErrorType.NEGATION)
    if laterality_mismatch or anatomy_mismatch:
        violations.append(ErrorType.LATERALITY)
    if severity_mismatch:
        violations.append(ErrorType.SEVERITY)
    if temporal_mismatch:
        violations.append(ErrorType.TEMPORAL)
    if measurement_mismatch:
        violations.append(ErrorType.MEASUREMENT)
    if unsupported_entities:
        violations.append(ErrorType.ENTITY)

    spans: list[TextSpan] = []
    marked_entities = set(negation_entities) | unsupported_entities
    if laterality_mismatch or anatomy_mismatch or severity_mismatch or temporal_mismatch or measurement_mismatch:
        marked_entities |= shared_entities
    for mention in claim_mentions:
        if mention.entity in marked_entities:
            error_type = ErrorType.ENTITY
            if mention.entity in negation_entities:
                error_type = ErrorType.NEGATION
            elif laterality_mismatch or anatomy_mismatch:
                error_type = ErrorType.LATERALITY
            elif measurement_mismatch:
                error_type = ErrorType.MEASUREMENT
            elif temporal_mismatch:
                error_type = ErrorType.TEMPORAL
            elif severity_mismatch:
                error_type = ErrorType.SEVERITY
            spans.append(TextSpan(mention.start, mention.end, mention.text, error_type))
    if global_polarity_mismatch:
        acute_match = re.search(
            r"\b(?:acute|active)\b.*?\b(?:abnormalit(?:y|ies)|disease|findings?|process)\b",
            claim,
            re.IGNORECASE,
        )
        if acute_match:
            spans.append(
                TextSpan(
                    acute_match.start(),
                    acute_match.end(),
                    acute_match.group(0),
                    ErrorType.NEGATION,
                )
            )
    for marker, error_type in (
        (claim_laterality if laterality_mismatch else set(), ErrorType.LATERALITY),
        ({token for token in tokenize(claim) if token in SEVERITY_LEVEL} if severity_mismatch else set(), ErrorType.SEVERITY),
        ({token for token in tokenize(claim) if token in TEMPORAL_CLASS} if temporal_mismatch else set(), ErrorType.TEMPORAL),
    ):
        for token in marker:
            for match in re.finditer(rf"\b{re.escape(token)}\b", claim, re.IGNORECASE):
                spans.append(TextSpan(match.start(), match.end(), match.group(0), error_type))
    if measurement_mismatch:
        for _, _, start, end, value in claim_measurements:
            spans.append(TextSpan(start, end, value, ErrorType.MEASUREMENT))
    spans = _deduplicate_spans(spans)

    critical_entity_involved = bool((shared_entities | unsupported_entities) & CRITICAL_ENTITIES)
    if violations and critical_entity_involved and any(
        item in violations for item in (ErrorType.NEGATION, ErrorType.LATERALITY, ErrorType.MEASUREMENT)
    ):
        risk = RiskTier.CRITICAL
    elif global_polarity_mismatch or any(
        item in violations for item in (ErrorType.NEGATION, ErrorType.LATERALITY, ErrorType.MEASUREMENT)
    ):
        risk = RiskTier.HIGH
    elif violations:
        risk = RiskTier.MODERATE
    else:
        risk = RiskTier.LOW

    features = {
        "lexical_jaccard": lexical_similarity,
        "entity_overlap": float(entity_overlap),
        "unsupported_entity_ratio": float(unsupported_ratio),
        "entity_substitution_mismatch": float(entity_substitution_mismatch),
        "negation_mismatch": float(bool(negation_entities) or global_polarity_mismatch),
        "global_polarity_mismatch": float(global_polarity_mismatch),
        "uncertainty_mismatch": float(bool(uncertainty_entities)),
        "laterality_mismatch": float(laterality_mismatch),
        "laterality_profile_mismatch": float(profile_laterality_mismatch),
        "laterality_alignment": float(laterality_alignment),
        "anatomy_mismatch": float(anatomy_mismatch),
        "severity_mismatch": float(severity_mismatch),
        "severity_distance": float(severity_distance),
        "temporal_mismatch": float(temporal_mismatch),
        "measurement_mismatch": float(measurement_mismatch),
        "measurement_relative_error": float(min(1.0, measurement_relative_error)),
        "claim_entity_count": float(len(claim_entities)),
        "shared_entity_count": float(len(shared_entities)),
    }
    details: dict[str, Any] = {
        "claim_entities": sorted(claim_entities),
        "evidence_entities": sorted(evidence_entities),
        "shared_entities": sorted(shared_entities),
        "unsupported_entities": sorted(unsupported_entities),
        "entity_substitution_pairs": [list(item) for item in substitution_pairs],
        "negation_mismatch_entities": negation_entities,
        "global_polarity_mismatch": global_polarity_mismatch,
        "uncertainty_mismatch_entities": uncertainty_entities,
        "claim_laterality": sorted(claim_laterality),
        "evidence_laterality": sorted(evidence_laterality),
        "laterality_mismatch_entities": laterality_entities,
        "laterality_profile_mismatch": profile_laterality_mismatch,
        "laterality_alignment": laterality_alignment,
        "anatomy_mismatch_entities": anatomy_entities,
        "claim_temporal": sorted(claim_temporal),
        "evidence_temporal": sorted(evidence_temporal),
    }
    return RuleAssessment(features, violations, spans, risk, details)


def combine_assessments(assessments: list[RuleAssessment]) -> RuleAssessment:
    """Select the most supportive evidence while retaining serious conflicts."""

    if not assessments:
        return RuleAssessment(
            features={
                "lexical_jaccard": 0.0,
                "entity_overlap": 0.0,
                "unsupported_entity_ratio": 1.0,
            },
            violations=[ErrorType.UNSUPPORTED],
            risk_tier=RiskTier.MODERATE,
            details={"reason": "No evidence sentences were available."},
        )
    # The sentence with the best entity/lexical support governs the rule result.
    best = max(
        assessments,
        key=lambda item: (
            item.features.get("entity_overlap", 0.0) - item.features.get("unsupported_entity_ratio", 0.0),
            item.features.get("lexical_jaccard", 0.0),
        ),
    )
    return best


def _deduplicate_spans(spans: list[TextSpan]) -> list[TextSpan]:
    output: list[TextSpan] = []
    seen: set[tuple[int, int, ErrorType]] = set()
    for span in sorted(spans, key=lambda value: (value.start, value.end, value.error_type.value)):
        key = (span.start, span.end, span.error_type)
        if key not in seen:
            seen.add(key)
            output.append(span)
    return output


def severity_to_probability(value: float) -> float:
    """Stable sigmoid helper used by the heuristic fusion model."""

    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))
