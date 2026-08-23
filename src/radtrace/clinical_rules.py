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
    # Device aliases deliberately collapse spelling variants and true synonyms
    # into one canonical identity.  Device subtypes remain separate so a
    # semantically changed device can be surfaced instead of appearing as a
    # generic unsupported concept.
    "endotracheal tube": (
        "endotracheal tubes",
        "endotracheal tube",
        "endotracheal catheter",
        "et tube",
        "ett",
    ),
    "tracheostomy tube": ("tracheostomy tube", "tracheostomy", "trach tube"),
    "enteric tube": ("enteric tube", "enteral tube"),
    "nasogastric tube": (
        "nasogastric tubes",
        "nasogastric tube",
        "nasogastric catheter",
        "ng tube",
        "ngt",
    ),
    "orogastric tube": ("orogastric tube", "og tube", "ogt"),
    "feeding tube": ("feeding tube", "dobbhoff tube", "dobhoff tube", "dobbhoff", "dobhoff"),
    "central venous catheter": (
        "central venous catheters",
        "central venous catheter",
        "central venous line",
        "central line",
        "cvc",
    ),
    "picc": (
        "peripherally inserted central catheter",
        "picc catheter",
        "picc line",
        "picc",
    ),
    "implanted venous port": (
        "implanted venous port",
        "implantable port",
        "port-a-cath",
        "portacath",
        "chest port",
    ),
    "pulmonary artery catheter": (
        "pulmonary artery catheter",
        "swan-ganz catheter",
        "swan ganz catheter",
        "swan-ganz",
        "swan ganz",
    ),
    "dialysis catheter": ("hemodialysis catheter", "haemodialysis catheter", "dialysis catheter"),
    "arterial line": ("arterial catheter", "arterial line", "a-line"),
    "chest tube": ("thoracostomy tubes", "thoracostomy tube", "chest tubes", "chest tube", "pleural drain"),
    "pigtail pleural catheter": ("pigtail pleural catheter", "pigtail catheter", "pigtail drain"),
    "pacemaker": ("cardiac pacemaker", "pacemaker", "pacing device"),
    "defibrillator": (
        "automatic implantable cardioverter defibrillator",
        "implantable cardioverter defibrillator",
        "cardioverter defibrillator",
        "defibrillator",
        "aicd",
        "icd",
    ),
    "intra-aortic balloon pump": (
        "intra-aortic balloon pump",
        "intraaortic balloon pump",
        "iabp",
    ),
    "vascular stent": ("vascular stent", "venous stent", "arterial stent"),
    "urinary catheter": ("urinary catheter", "foley catheter", "foley"),
    "ventriculoperitoneal shunt": ("ventriculoperitoneal shunt", "vp shunt"),
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

DEVICE_ENTITIES = {
    "endotracheal tube",
    "tracheostomy tube",
    "enteric tube",
    "nasogastric tube",
    "orogastric tube",
    "feeding tube",
    "central venous catheter",
    "picc",
    "implanted venous port",
    "pulmonary artery catheter",
    "dialysis catheter",
    "arterial line",
    "chest tube",
    "pigtail pleural catheter",
    "pacemaker",
    "defibrillator",
    "intra-aortic balloon pump",
    "vascular stent",
    "urinary catheter",
    "ventriculoperitoneal shunt",
}

# A generic device name may legitimately summarize a more specific subtype.
# These pairs are support-compatible, but two distinct specific subtypes are
# still treated as a possible identity substitution.
GENERIC_DEVICE_COMPATIBILITY = {
    frozenset(("enteric tube", "nasogastric tube")),
    frozenset(("enteric tube", "orogastric tube")),
    frozenset(("enteric tube", "feeding tube")),
    frozenset(("central venous catheter", "picc")),
    frozenset(("central venous catheter", "implanted venous port")),
    frozenset(("central venous catheter", "dialysis catheter")),
    frozenset(("chest tube", "pigtail pleural catheter")),
}

CRITICAL_ENTITIES = {
    "pneumothorax",
    "free air",
    "endotracheal tube",
    "tracheostomy tube",
    "central venous catheter",
    "picc",
    "pulmonary artery catheter",
    "dialysis catheter",
    "chest tube",
    "pigtail pleural catheter",
    "pacemaker",
    "defibrillator",
    "intra-aortic balloon pump",
}

NEGATION_TERMS = {"no", "not", "without", "absent", "neither", "negative", "free"}
UNCERTAINTY_TERMS = {"possible", "possibly", "probable", "may", "might", "suggest", "suggests", "cannot", "questionable"}
LATERALITY = {"left", "right", "bilateral"}

ANATOMY_PATTERNS: dict[str, tuple[str, ...]] = {
    "upper": ("upper",),
    "middle": ("middle", "mid"),
    "lower": ("lower",),
    "apical": ("apices", "apex", "apical"),
    "basilar": ("bibasilar", "bases", "base", "basal", "basilar"),
    "hilar": ("hila", "hilum", "hilar"),
    "perihilar": ("perihilar", "peri-hilar"),
    "retrocardiac": ("retrocardiac", "retro-cardiac"),
    "lingular": ("lingula", "lingular"),
    "costophrenic": ("costophrenic", "costophrenic angle"),
    "subpulmonic": ("subpulmonic", "sub-pulmonic"),
    "central": ("central",),
    "peripheral": ("peripheral",),
    "medial": ("medial",),
    "lateral": ("lateral",),
    "anterior": ("anterior",),
    "posterior": ("posterior",),
    "superior": ("superior",),
    "inferior": ("inferior",),
    "proximal": ("proximal",),
    "distal": ("distal",),
    "cervical": ("cervical",),
    "thoracic": ("thoracic",),
    "lumbar": ("lumbar",),
    "abdominal": ("abdominal", "abdomen"),
    "pelvic": ("pelvic", "pelvis"),
}
ANATOMY = {alias for aliases in ANATOMY_PATTERNS.values() for alias in aliases if " " not in alias}
CONTRASTING_ANATOMY_PAIRS = {
    frozenset(("upper", "middle")),
    frozenset(("upper", "lower")),
    frozenset(("middle", "lower")),
    frozenset(("apical", "basilar")),
    frozenset(("central", "peripheral")),
    frozenset(("hilar", "peripheral")),
    frozenset(("perihilar", "peripheral")),
    frozenset(("medial", "lateral")),
    frozenset(("anterior", "posterior")),
    frozenset(("superior", "inferior")),
    frozenset(("proximal", "distal")),
    frozenset(("cervical", "thoracic")),
    frozenset(("cervical", "lumbar")),
    frozenset(("thoracic", "lumbar")),
    frozenset(("thoracic", "abdominal")),
    frozenset(("thoracic", "pelvic")),
    frozenset(("abdominal", "pelvic")),
}

DEVICE_POSITION_PATTERNS: dict[str, tuple[str, ...]] = {
    "upper_svc": ("upper superior vena cava", "upper svc"),
    "mid_svc": ("middle superior vena cava", "mid superior vena cava", "middle svc", "mid svc"),
    "lower_svc": ("lower superior vena cava", "distal superior vena cava", "lower svc", "distal svc"),
    "svc": ("superior vena cava", "svc"),
    "cavoatrial_junction": ("cavoatrial junction", "atrio-caval junction", "atriocaval junction"),
    "right_atrium": ("right atrium", "right atrial"),
    "right_ventricle": ("right ventricle", "right ventricular"),
    "ivc": ("inferior vena cava", "ivc"),
    "brachiocephalic_vein": ("brachiocephalic vein", "innominate vein"),
    "subclavian_vein": ("subclavian vein",),
    "internal_jugular_vein": ("internal jugular vein", "jugular vein"),
    "azygos_vein": ("azygos vein", "azygous vein"),
    "trachea": ("within the trachea", "in the trachea", "trachea"),
    "above_carina": ("above the carina", "above carina", "proximal to the carina"),
    "at_carina": ("at the carina", "at carina"),
    "below_carina": ("below the carina", "below carina", "distal to the carina"),
    "right_mainstem": ("right mainstem bronchus", "right main bronchus"),
    "left_mainstem": ("left mainstem bronchus", "left main bronchus"),
    "proximal_esophagus": ("proximal esophagus", "upper esophagus"),
    "mid_esophagus": ("mid esophagus", "middle esophagus"),
    "distal_esophagus": ("distal esophagus", "lower esophagus"),
    "stomach": (
        "gastric fundus",
        "gastric body",
        "gastric antrum",
        "gastric lumen",
        "within the stomach",
        "in the stomach",
        "stomach",
    ),
    "duodenum": ("postpyloric", "post-pyloric", "duodenal", "duodenum"),
    "thoracic_apex": ("thoracic apex", "lung apex", "apical pleural space", "apex", "apical"),
    "thoracic_base": ("thoracic base", "lung base", "basilar pleural space", "base", "basilar"),
    "too_high": ("too high", "high position", "high positioned"),
    "too_low": ("too low", "low position", "low positioned"),
    "shallow": ("shallow",),
    "deep": ("too deep", "deep position", "deeply positioned"),
}
DEVICE_POSITION_GROUP = {
    "upper_svc": "venous",
    "mid_svc": "venous",
    "lower_svc": "venous",
    "svc": "venous",
    "cavoatrial_junction": "venous",
    "right_atrium": "venous",
    "right_ventricle": "venous",
    "ivc": "venous",
    "brachiocephalic_vein": "venous",
    "subclavian_vein": "venous",
    "internal_jugular_vein": "venous",
    "azygos_vein": "venous",
    "trachea": "airway",
    "above_carina": "airway",
    "at_carina": "airway",
    "below_carina": "airway",
    "right_mainstem": "airway",
    "left_mainstem": "airway",
    "proximal_esophagus": "enteric",
    "mid_esophagus": "enteric",
    "distal_esophagus": "enteric",
    "stomach": "enteric",
    "duodenum": "enteric",
    "thoracic_apex": "pleural",
    "thoracic_base": "pleural",
    "too_high": "depth",
    "too_low": "depth",
    "shallow": "depth",
    "deep": "depth",
}
SEVERITY_LEVEL = {
    "trace": 0,
    "minimal": 0,
    "tiny": 0,
    "small": 1,
    "mild": 1,
    "mildly": 1,
    "slight": 1,
    "slightly": 1,
    "moderate": 2,
    "moderately": 2,
    "large": 3,
    "severe": 3,
    "severely": 3,
    "marked": 3,
    "markedly": 3,
    "substantial": 3,
    "extensive": 3,
    "massive": 4,
}
TEMPORAL_CLASS = {
    "new": "new",
    "newly": "new",
    "now": "new",
    "developed": "new",
    "developing": "new",
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
    "resolution": "resolved",
    "recurred": "new",
    "recurrent": "new",
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
    severity: frozenset[int] = frozenset()
    temporal: frozenset[str] = frozenset()
    device_position: frozenset[str] = frozenset()


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
            severity_markers = _nearest_markers(
                lower,
                match.start(),
                match.end(),
                set(SEVERITY_LEVEL),
            )
            temporal_markers = _nearest_markers(
                lower,
                match.start(),
                match.end(),
                set(TEMPORAL_CLASS),
            )
            mentions.append(
                EntityMention(
                    entity=entity,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                    negated=negated,
                    uncertain=uncertain,
                    laterality=_nearest_markers(
                        lower,
                        match.start(),
                        match.end(),
                        LATERALITY,
                    ),
                    anatomy=_nearest_concepts(
                        lower,
                        match.start(),
                        match.end(),
                        ANATOMY_PATTERNS,
                    ),
                    severity=frozenset(SEVERITY_LEVEL[item] for item in severity_markers),
                    temporal=frozenset(TEMPORAL_CLASS[item] for item in temporal_markers),
                    device_position=(
                        _nearest_concepts(
                            lower,
                            match.start(),
                            match.end(),
                            DEVICE_POSITION_PATTERNS,
                            radius=86,
                        )
                        if entity in DEVICE_ENTITIES
                        else frozenset()
                    ),
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


def _nearest_concepts(
    text: str,
    entity_start: int,
    entity_end: int,
    patterns: dict[str, tuple[str, ...]],
    *,
    radius: int = 52,
) -> frozenset[str]:
    """Return canonical concepts nearest to one entity mention.

    Multi-word aliases are matched before shorter aliases so ``lower SVC`` is
    represented as one specific position rather than both ``lower_svc`` and a
    generic ``svc`` marker.
    """

    start = max(0, entity_start - radius)
    end = min(len(text), entity_end + radius)
    window = text[start:end]
    candidates: list[tuple[int, int, int, str]] = []
    for concept, aliases in patterns.items():
        for alias in sorted(aliases, key=len, reverse=True):
            for match in re.finditer(rf"\b{re.escape(alias)}\b", window, re.IGNORECASE):
                absolute_start = start + match.start()
                absolute_end = start + match.end()
                distance = min(
                    abs(entity_start - absolute_end),
                    abs(absolute_start - entity_end),
                )
                candidates.append((distance, absolute_start, absolute_end, concept))
    if not candidates:
        return frozenset()
    candidates.sort(key=lambda item: (item[0], -(item[2] - item[1])))
    nearest_distance = candidates[0][0]
    selected: list[tuple[int, int, str]] = []
    for distance, candidate_start, candidate_end, concept in candidates:
        if distance > nearest_distance + 2:
            continue
        if any(candidate_start < kept_end and candidate_end > kept_start for kept_start, kept_end, _ in selected):
            continue
        selected.append((candidate_start, candidate_end, concept))
    return frozenset(concept for _, _, concept in selected)


def _concept_spans(
    text: str,
    concepts: set[str] | frozenset[str],
    patterns: dict[str, tuple[str, ...]],
    error_type: ErrorType,
) -> list[TextSpan]:
    spans: list[TextSpan] = []
    occupied: list[tuple[int, int]] = []
    aliases = sorted(
        (
            (concept, alias)
            for concept in concepts
            for alias in patterns.get(concept, ())
        ),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for _, alias in aliases:
        for match in re.finditer(rf"\b{re.escape(alias)}\b", text or "", re.IGNORECASE):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            spans.append(TextSpan(match.start(), match.end(), match.group(0), error_type))
            occupied.append((match.start(), match.end()))
    return spans


def _device_identity_compatible(left: str, right: str) -> bool:
    return left == right or frozenset((left, right)) in GENERIC_DEVICE_COMPATIBILITY


def _device_position_mismatch(
    left: frozenset[str],
    right: frozenset[str],
) -> tuple[bool, list[tuple[str, str]]]:
    """Compare device positions only within the same anatomical axis.

    A generic ``SVC`` description is compatible with a more specific SVC
    segment, avoiding a false alarm when a summary is less granular.
    """

    mismatches: list[tuple[str, str]] = []
    for group in set(DEVICE_POSITION_GROUP.values()):
        left_group = {item for item in left if DEVICE_POSITION_GROUP.get(item) == group}
        right_group = {item for item in right if DEVICE_POSITION_GROUP.get(item) == group}
        if not left_group or not right_group or left_group & right_group:
            continue
        if group == "venous" and ("svc" in left_group or "svc" in right_group):
            other = (right_group if "svc" in left_group else left_group)
            if other <= {"upper_svc", "mid_svc", "lower_svc", "cavoatrial_junction"}:
                continue
        if group == "airway" and ("trachea" in left_group or "trachea" in right_group):
            other = right_group if "trachea" in left_group else left_group
            if other <= {"above_carina", "at_carina"}:
                continue
        mismatches.extend(sorted((left_item, right_item) for left_item in left_group for right_item in right_group))
    return bool(mismatches), mismatches


def _anatomy_sets_mismatch(left: frozenset[str], right: frozenset[str]) -> bool:
    return any(
        frozenset((left_item, right_item)) in CONTRASTING_ANATOMY_PAIRS
        for left_item in left
        for right_item in right
    )


def _ordinal_distance(left: frozenset[int], right: frozenset[int]) -> int:
    if not left or not right:
        return 0
    left_to_right = max(min(abs(a - b) for b in right) for a in left)
    right_to_left = max(min(abs(b - a) for a in left) for b in right)
    return max(left_to_right, right_to_left)


def _pattern_alignment(
    left: str,
    right: str,
    patterns: dict[str, tuple[str, ...]],
) -> float:
    """Compare two phrases after replacing all patterned aliases."""

    aliases = sorted(
        {alias for values in patterns.values() for alias in values},
        key=len,
        reverse=True,
    )

    def masked_tokens(value: str) -> list[str]:
        masked = value or ""
        for alias in aliases:
            masked = re.sub(rf"\b{re.escape(alias)}\b", " <concept> ", masked, flags=re.IGNORECASE)
        return tokenize(masked)

    return SequenceMatcher(None, masked_tokens(left), masked_tokens(right), autojunk=False).ratio()


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
    claim_devices = claim_entities & DEVICE_ENTITIES
    evidence_devices = evidence_entities & DEVICE_ENTITIES
    compatible_device_pairs = sorted(
        (claim_device, evidence_device)
        for claim_device in (claim_devices - shared_entities)
        for evidence_device in (evidence_devices - shared_entities)
        if _device_identity_compatible(claim_device, evidence_device)
    )
    compatible_claim_devices = {item[0] for item in compatible_device_pairs}
    compatible_evidence_devices = {item[1] for item in compatible_device_pairs}
    matched_claim_entities = shared_entities | compatible_claim_devices
    unsupported_entities = claim_entities - matched_claim_entities
    unmatched_evidence_entities = (
        evidence_entities - shared_entities - compatible_evidence_devices
    )
    lexical_similarity = _jaccard(claim, evidence)
    entity_overlap = (
        len(matched_claim_entities) / len(claim_entities)
        if claim_entities
        else lexical_similarity
    )
    unsupported_ratio = len(unsupported_entities) / max(1, len(claim_entities))
    substitution_pairs = sorted(
        (claim_entity, evidence_entity)
        for claim_entity in unsupported_entities
        for evidence_entity in unmatched_evidence_entities
        if frozenset((claim_entity, evidence_entity)) in CONTRASTING_ENTITY_PAIRS
    )
    entity_substitution_mismatch = bool(substitution_pairs and lexical_similarity >= 0.18)

    device_identity_pairs = sorted(
        (claim_device, evidence_device)
        for claim_device in (unsupported_entities & DEVICE_ENTITIES)
        for evidence_device in (unmatched_evidence_entities & DEVICE_ENTITIES)
        if not _device_identity_compatible(claim_device, evidence_device)
    )
    device_identity_alignment = _pattern_alignment(claim, evidence, ENTITY_PATTERNS)
    device_identity_mismatch = bool(
        device_identity_pairs and device_identity_alignment >= 0.58
    )

    # Exact entity matches plus compatible generic/specific devices form the
    # pairs on which local modifiers can safely be compared.
    aligned_entity_pairs = sorted(
        [(entity, entity) for entity in shared_entities] + compatible_device_pairs
    )

    negation_pairs = sorted(
        (claim_entity, evidence_entity)
        for claim_entity, evidence_entity in aligned_entity_pairs
        if claim_states[claim_entity].negated != evidence_states[evidence_entity].negated
    )
    negation_entities = sorted({item[0] for item in negation_pairs})
    global_polarity_mismatch = _generic_acute_positive(claim) and _reassuring_evidence(
        evidence, evidence_mentions
    )
    uncertainty_pairs = sorted(
        (claim_entity, evidence_entity)
        for claim_entity, evidence_entity in aligned_entity_pairs
        if claim_states[claim_entity].uncertain != evidence_states[evidence_entity].uncertain
    )
    uncertainty_entities = sorted({item[0] for item in uncertainty_pairs})

    claim_laterality = _marker_set(claim, LATERALITY)
    evidence_laterality = _marker_set(evidence, LATERALITY)
    laterality_pairs = sorted(
        (claim_entity, evidence_entity)
        for claim_entity, evidence_entity in aligned_entity_pairs
        if claim_states[claim_entity].laterality
        and evidence_states[evidence_entity].laterality
        and claim_states[claim_entity].laterality.isdisjoint(
            evidence_states[evidence_entity].laterality
        )
    )
    laterality_entities = sorted({item[0] for item in laterality_pairs})
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
    anatomy_pairs = sorted(
        (claim_entity, evidence_entity)
        for claim_entity, evidence_entity in aligned_entity_pairs
        if not (
            claim_entity in DEVICE_ENTITIES and evidence_entity in DEVICE_ENTITIES
        )
        if claim_states[claim_entity].anatomy
        and evidence_states[evidence_entity].anatomy
        and _anatomy_sets_mismatch(
            claim_states[claim_entity].anatomy,
            evidence_states[evidence_entity].anatomy,
        )
    )
    anatomy_entities = sorted({item[0] for item in anatomy_pairs})
    anatomy_mismatch = bool(anatomy_entities)

    device_position_details: list[dict[str, Any]] = []
    device_position_claim_concepts: set[str] = set()
    for claim_entity, evidence_entity in aligned_entity_pairs:
        if claim_entity not in DEVICE_ENTITIES or evidence_entity not in DEVICE_ENTITIES:
            continue
        is_mismatch, position_pairs = _device_position_mismatch(
            claim_states[claim_entity].device_position,
            evidence_states[evidence_entity].device_position,
        )
        if is_mismatch:
            device_position_claim_concepts.update(left for left, _ in position_pairs)
            device_position_details.append(
                {
                    "claim_device": claim_entity,
                    "evidence_device": evidence_entity,
                    "claim_positions": sorted(claim_states[claim_entity].device_position),
                    "evidence_positions": sorted(evidence_states[evidence_entity].device_position),
                    "position_pairs": [list(item) for item in position_pairs],
                }
            )
    device_position_mismatch = bool(device_position_details)

    claim_severity_values = [SEVERITY_LEVEL[token] for token in tokenize(claim) if token in SEVERITY_LEVEL]
    evidence_severity_values = [
        SEVERITY_LEVEL[token] for token in tokenize(evidence) if token in SEVERITY_LEVEL
    ]
    severity_pair_details: list[dict[str, Any]] = []
    local_severity_distances: list[int] = []
    for claim_entity, evidence_entity in aligned_entity_pairs:
        claim_levels = claim_states[claim_entity].severity
        evidence_levels = evidence_states[evidence_entity].severity
        if not claim_levels or not evidence_levels:
            continue
        distance = _ordinal_distance(claim_levels, evidence_levels)
        local_severity_distances.append(distance)
        if distance >= 1:
            severity_pair_details.append(
                {
                    "claim_entity": claim_entity,
                    "evidence_entity": evidence_entity,
                    "claim_levels": sorted(claim_levels),
                    "evidence_levels": sorted(evidence_levels),
                    "distance": distance,
                }
            )
    local_severity_comparable = bool(local_severity_distances)
    global_severity_distance = _ordinal_distance(
        frozenset(claim_severity_values),
        frozenset(evidence_severity_values),
    )
    severity_alignment = _modifier_alignment(claim, evidence, set(SEVERITY_LEVEL))
    profile_severity_mismatch = bool(
        not local_severity_comparable
        and global_severity_distance >= 1
        and severity_alignment >= 0.68
    )
    local_severity_distance = max(local_severity_distances, default=0)
    severity_distance = max(local_severity_distance, global_severity_distance if profile_severity_mismatch else 0)
    severity_mismatch = bool(severity_pair_details) or profile_severity_mismatch

    claim_temporal = {TEMPORAL_CLASS[token] for token in tokenize(claim) if token in TEMPORAL_CLASS}
    evidence_temporal = {TEMPORAL_CLASS[token] for token in tokenize(evidence) if token in TEMPORAL_CLASS}
    incompatible_temporal = {
        ("worse", "better"), ("better", "worse"), ("new", "stable"),
        ("new", "resolved"), ("resolved", "new"), ("resolved", "worse"),
        ("worse", "stable"), ("better", "stable"), ("stable", "worse"),
        ("stable", "better"), ("stable", "new"), ("resolved", "stable"),
    }
    temporal_pair_details: list[dict[str, Any]] = []
    local_temporal_comparable = False
    for claim_entity, evidence_entity in aligned_entity_pairs:
        claim_classes = claim_states[claim_entity].temporal
        evidence_classes = evidence_states[evidence_entity].temporal
        if not claim_classes or not evidence_classes:
            continue
        local_temporal_comparable = True
        conflicts = sorted(
            (left, right)
            for left in claim_classes
            for right in evidence_classes
            if (left, right) in incompatible_temporal
        )
        if conflicts:
            temporal_pair_details.append(
                {
                    "claim_entity": claim_entity,
                    "evidence_entity": evidence_entity,
                    "claim_classes": sorted(claim_classes),
                    "evidence_classes": sorted(evidence_classes),
                    "class_pairs": [list(item) for item in conflicts],
                }
            )
    global_temporal_conflict = bool(
        any((left, right) in incompatible_temporal for left in claim_temporal for right in evidence_temporal)
    )
    temporal_alignment = _modifier_alignment(claim, evidence, set(TEMPORAL_CLASS))
    profile_temporal_mismatch = bool(
        not local_temporal_comparable
        and global_temporal_conflict
        and temporal_alignment >= 0.64
    )
    temporal_mismatch = bool(temporal_pair_details) or profile_temporal_mismatch

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
    if measurement_mismatch or device_identity_mismatch or device_position_mismatch:
        violations.append(ErrorType.MEASUREMENT)
    if unsupported_entities:
        violations.append(ErrorType.ENTITY)

    spans: list[TextSpan] = []
    device_identity_claim_entities = {item[0] for item in device_identity_pairs}
    device_position_claim_entities = {
        str(item["claim_device"]) for item in device_position_details
    }
    laterality_claim_entities = set(laterality_entities)
    anatomy_claim_entities = set(anatomy_entities)
    severity_claim_entities = {
        str(item["claim_entity"]) for item in severity_pair_details
    }
    temporal_claim_entities = {
        str(item["claim_entity"]) for item in temporal_pair_details
    }
    marked_entities = (
        set(negation_entities)
        | unsupported_entities
        | laterality_claim_entities
        | anatomy_claim_entities
        | severity_claim_entities
        | temporal_claim_entities
        | device_identity_claim_entities
        | device_position_claim_entities
    )
    for mention in claim_mentions:
        if mention.entity in marked_entities:
            error_type = ErrorType.ENTITY
            if mention.entity in negation_entities:
                error_type = ErrorType.NEGATION
            elif mention.entity in device_identity_claim_entities or mention.entity in device_position_claim_entities:
                error_type = ErrorType.MEASUREMENT
            elif mention.entity in laterality_claim_entities or mention.entity in anatomy_claim_entities:
                error_type = ErrorType.LATERALITY
            elif mention.entity in temporal_claim_entities:
                error_type = ErrorType.TEMPORAL
            elif mention.entity in severity_claim_entities:
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
    if anatomy_mismatch:
        claim_anatomy_concepts = {
            concept
            for entity in anatomy_entities
            for concept in claim_states[entity].anatomy
        }
        spans.extend(
            _concept_spans(
                claim,
                claim_anatomy_concepts,
                ANATOMY_PATTERNS,
                ErrorType.LATERALITY,
            )
        )
    if device_position_mismatch:
        spans.extend(
            _concept_spans(
                claim,
                device_position_claim_concepts,
                DEVICE_POSITION_PATTERNS,
                ErrorType.MEASUREMENT,
            )
        )
    if measurement_mismatch:
        for _, _, start, end, value in claim_measurements:
            spans.append(TextSpan(start, end, value, ErrorType.MEASUREMENT))
    spans = _deduplicate_spans(spans)

    critical_entity_involved = bool(
        (matched_claim_entities | unsupported_entities) & CRITICAL_ENTITIES
    )
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
        "device_identity_mismatch": float(device_identity_mismatch),
        "device_identity_alignment": float(device_identity_alignment),
        "device_identity_mismatch_count": float(len(device_identity_pairs)),
        "device_position_mismatch": float(device_position_mismatch),
        "device_position_mismatch_count": float(len(device_position_details)),
        "negation_mismatch": float(bool(negation_entities) or global_polarity_mismatch),
        "global_polarity_mismatch": float(global_polarity_mismatch),
        "uncertainty_mismatch": float(bool(uncertainty_entities)),
        "laterality_mismatch": float(laterality_mismatch),
        "laterality_profile_mismatch": float(profile_laterality_mismatch),
        "laterality_alignment": float(laterality_alignment),
        "anatomy_mismatch": float(anatomy_mismatch),
        "anatomy_mismatch_count": float(len(anatomy_pairs)),
        "severity_mismatch": float(severity_mismatch),
        "severity_distance": float(severity_distance),
        "local_severity_mismatch": float(bool(severity_pair_details)),
        "profile_severity_mismatch": float(profile_severity_mismatch),
        "severity_alignment": float(severity_alignment),
        "temporal_mismatch": float(temporal_mismatch),
        "local_temporal_mismatch": float(bool(temporal_pair_details)),
        "profile_temporal_mismatch": float(profile_temporal_mismatch),
        "temporal_alignment": float(temporal_alignment),
        "measurement_mismatch": float(measurement_mismatch),
        "measurement_relative_error": float(min(1.0, measurement_relative_error)),
        "claim_entity_count": float(len(claim_entities)),
        "shared_entity_count": float(len(shared_entities)),
        "matched_entity_count": float(len(matched_claim_entities)),
    }
    details: dict[str, Any] = {
        "claim_entities": sorted(claim_entities),
        "evidence_entities": sorted(evidence_entities),
        "shared_entities": sorted(shared_entities),
        "unsupported_entities": sorted(unsupported_entities),
        "entity_substitution_pairs": [list(item) for item in substitution_pairs],
        "compatible_device_pairs": [list(item) for item in compatible_device_pairs],
        "device_identity_mismatch_pairs": [list(item) for item in device_identity_pairs],
        "device_identity_alignment": device_identity_alignment,
        "device_position_mismatches": device_position_details,
        "negation_mismatch_entities": negation_entities,
        "negation_mismatch_pairs": [list(item) for item in negation_pairs],
        "global_polarity_mismatch": global_polarity_mismatch,
        "uncertainty_mismatch_entities": uncertainty_entities,
        "claim_laterality": sorted(claim_laterality),
        "evidence_laterality": sorted(evidence_laterality),
        "laterality_mismatch_entities": laterality_entities,
        "laterality_mismatch_pairs": [list(item) for item in laterality_pairs],
        "laterality_profile_mismatch": profile_laterality_mismatch,
        "laterality_alignment": laterality_alignment,
        "anatomy_mismatch_entities": anatomy_entities,
        "anatomy_mismatch_pairs": [list(item) for item in anatomy_pairs],
        "severity_mismatches": severity_pair_details,
        "profile_severity_mismatch": profile_severity_mismatch,
        "severity_alignment": severity_alignment,
        "claim_temporal": sorted(claim_temporal),
        "evidence_temporal": sorted(evidence_temporal),
        "temporal_mismatches": temporal_pair_details,
        "profile_temporal_mismatch": profile_temporal_mismatch,
        "temporal_alignment": temporal_alignment,
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
