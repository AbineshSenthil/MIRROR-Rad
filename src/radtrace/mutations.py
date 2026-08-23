"""Controlled counterfactual mutations for radiology factuality experiments.

Mutations are intentionally transparent and rule-based.  Every generated row
records its operation, changed character span, expected NLI label, and risk
tier.  The caller must split reports *before* invoking this module.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass

from .text import normalize_text

CONTRADICTED = "CONTRADICTED"
UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class Mutation:
    mutation_id: str
    report_id: str
    mutation_type: str
    original_claim: str
    mutated_claim: str
    label: str
    risk_tier: str
    risk_weight: float
    description: str
    changed_original_text: str
    changed_mutated_text: str
    changed_original_start: int
    changed_original_end: int
    changed_mutated_start: int
    changed_mutated_end: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Candidate:
    text: str
    mutation_type: str
    label: str
    risk_tier: str
    risk_weight: float
    description: str


def _match_case(replacement: str, matched: str) -> str:
    if matched.isupper():
        return replacement.upper()
    if matched[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_once(text: str, pattern: str, replacement: str) -> str | None:
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    match = compiled.search(text)
    if not match:
        return None
    value = _match_case(replacement, match.group(0))
    return text[: match.start()] + value + text[match.end() :]


def _negation(claim: str, _: str, __: random.Random) -> _Candidate | None:
    removals = (
        (r"\bno evidence of\b", "evidence of"),
        (r"\bwithout evidence of\b", "with evidence of"),
        (r"\bnegative for\b", "positive for"),
        (r"\bnot visualized\b", "visualized"),
        (r"\bnot seen\b", "seen"),
        (r"\bnot present\b", "present"),
        (r"\bthere (?:is|are) no\b", "there is"),
        (r"^\s*no\b", ""),
        (r"\bwithout\b", "with"),
        (r"\babsent\b", "present"),
    )
    for pattern, replacement in removals:
        mutated = _replace_once(claim, pattern, replacement)
        if mutated:
            mutated = normalize_text(mutated)
            if claim[:1].isupper() and mutated[:1].islower():
                mutated = mutated[:1].upper() + mutated[1:]
            return _Candidate(
                mutated,
                "negation",
                CONTRADICTED,
                "high",
                3.0,
                "Reversed a clinical negation or absence cue.",
            )

    insertions = (
        (r"\b(is|are) present\b", r"\1 not present"),
        (r"\b(is|are) visualized\b", r"\1 not visualized"),
        (r"\b(is|are) seen\b", r"\1 not seen"),
        (r"\b(is|are) identified\b", r"\1 not identified"),
        (r"\bdemonstrates\b", "demonstrates no"),
        (r"\bshows\b", "shows no"),
    )
    for pattern, replacement in insertions:
        if re.search(pattern, claim, flags=re.IGNORECASE):
            mutated = re.sub(pattern, replacement, claim, count=1, flags=re.IGNORECASE)
            return _Candidate(
                normalize_text(mutated),
                "negation",
                CONTRADICTED,
                "high",
                3.0,
                "Inserted a clinical negation cue.",
            )
    return None


def _laterality(claim: str, _: str, __: random.Random) -> _Candidate | None:
    match = re.search(r"\b(left|right)\b", claim, flags=re.IGNORECASE)
    if not match:
        return None
    replacement = "right" if match.group(0).casefold() == "left" else "left"
    replacement = _match_case(replacement, match.group(0))
    mutated = claim[: match.start()] + replacement + claim[match.end() :]
    return _Candidate(
        normalize_text(mutated),
        "laterality",
        CONTRADICTED,
        "high",
        3.0,
        f"Swapped laterality from {match.group(0)} to {replacement}.",
    )


_SEVERITY_PAIRS = (
    ("trace", "large"),
    ("minimal", "severe"),
    ("mild", "severe"),
    ("moderate", "severe"),
    ("severe", "mild"),
    ("small", "large"),
    ("large", "small"),
    ("marked", "mild"),
    ("extensive", "minimal"),
)


def _severity(claim: str, _: str, __: random.Random) -> _Candidate | None:
    for original, replacement in _SEVERITY_PAIRS:
        pattern = rf"\b{re.escape(original)}\b"
        if mutated := _replace_once(claim, pattern, replacement):
            return _Candidate(
                normalize_text(mutated),
                "severity",
                CONTRADICTED,
                "medium",
                2.0,
                f"Changed severity/extent from {original} to {replacement}.",
            )
    return None


_TEMPORAL_PAIRS = (
    ("unchanged", "worsened"),
    ("stable", "worsened"),
    ("improved", "worsened"),
    ("improving", "worsening"),
    ("decreased", "increased"),
    ("decreasing", "increasing"),
    ("resolved", "recurred"),
    ("resolving", "progressing"),
    ("worsened", "improved"),
    ("worsening", "improving"),
    ("increased", "decreased"),
    ("increasing", "decreasing"),
    ("new", "chronic"),
    ("chronic", "new"),
)


def _temporal(claim: str, _: str, __: random.Random) -> _Candidate | None:
    for original, replacement in _TEMPORAL_PAIRS:
        pattern = rf"\b{re.escape(original)}\b"
        if mutated := _replace_once(claim, pattern, replacement):
            return _Candidate(
                normalize_text(mutated),
                "temporal_change",
                CONTRADICTED,
                "high",
                3.0,
                f"Reversed temporal status from {original} to {replacement}.",
            )
    return None


_MEASUREMENT_RE = re.compile(
    r"(?<![\w.])(?P<number>\d+(?:\.\d+)?)\s?(?P<unit>mm|cm|m|ml|cc|%)\b",
    flags=re.IGNORECASE,
)


def _format_changed_number(original: str) -> str:
    value = float(original)
    changed = 1.0 if value == 0 else (value * 2.0 if value < 10 else value + 10.0)
    decimals = len(original.partition(".")[2]) if "." in original else 0
    if decimals:
        return f"{changed:.{decimals}f}"
    return str(max(1, round(changed)))


def _measurement(claim: str, source: str, __: random.Random) -> _Candidate | None:
    for match in _MEASUREMENT_RE.finditer(claim):
        changed_number = _format_changed_number(match.group("number"))
        replacement = f"{changed_number} {match.group('unit')}"
        # Reject a replacement that is already explicitly stated in the source.
        if re.search(rf"\b{re.escape(replacement)}\b", source, flags=re.IGNORECASE):
            continue
        mutated = claim[: match.start()] + replacement + claim[match.end() :]
        return _Candidate(
            normalize_text(mutated),
            "measurement",
            CONTRADICTED,
            "high",
            3.0,
            f"Changed measurement from {match.group(0)} to {replacement}.",
        )
    return None


_ENTITY_PAIRS = (
    ("pleural effusion", "pneumothorax"),
    ("pneumothorax", "pleural effusion"),
    ("pulmonary edema", "pneumonia"),
    ("pneumonia", "pulmonary edema"),
    ("atelectasis", "consolidation"),
    ("consolidation", "atelectasis"),
    ("cardiomegaly", "pneumothorax"),
    ("fracture", "pleural effusion"),
    ("nodule", "infiltrate"),
    ("infiltrate", "nodule"),
    ("mass", "edema"),
)


def _entity_substitution(claim: str, source: str, rng: random.Random) -> _Candidate | None:
    candidates: list[tuple[str, str, re.Match[str]]] = []
    for original, replacement in _ENTITY_PAIRS:
        match = re.search(rf"\b{re.escape(original)}\b", claim, flags=re.IGNORECASE)
        if match and not re.search(
            rf"\b{re.escape(replacement)}\b", source, flags=re.IGNORECASE
        ):
            candidates.append((original, replacement, match))
    if not candidates:
        return None
    original, replacement, match = candidates[rng.randrange(len(candidates))]
    cased = _match_case(replacement, match.group(0))
    mutated = claim[: match.start()] + cased + claim[match.end() :]
    return _Candidate(
        normalize_text(mutated),
        "entity_substitution",
        CONTRADICTED,
        "high",
        3.0,
        f"Substituted clinical entity {original} with {replacement}.",
    )


_UNSUPPORTED_CLAIMS = (
    ("A new right apical pneumothorax is present.", ("pneumothorax",)),
    ("There is a new left pleural effusion.", ("pleural effusion",)),
    ("A displaced rib fracture is identified.", ("rib fracture", "fracture")),
    ("A new focal airspace consolidation is present.", ("consolidation",)),
    ("The central venous catheter tip is malpositioned.", ("central venous catheter", "catheter")),
    ("A suspicious pulmonary nodule is identified.", ("pulmonary nodule", "nodule")),
    ("There is severe pulmonary edema.", ("pulmonary edema",)),
)


def _unsupported(claim: str, source: str, rng: random.Random) -> _Candidate | None:
    combined = f"{source} {claim}".casefold()
    eligible = [
        candidate
        for candidate in _UNSUPPORTED_CLAIMS
        if all(term.casefold() not in combined for term in candidate[1])
    ]
    if not eligible:
        return None
    sentence, _ = eligible[rng.randrange(len(eligible))]
    return _Candidate(
        sentence,
        "unsupported_insertion",
        UNSUPPORTED,
        "high",
        3.0,
        "Inserted a clinically specific claim with no matching source entity.",
    )


_MUTATORS: dict[str, Callable[[str, str, random.Random], _Candidate | None]] = {
    "negation": _negation,
    "laterality": _laterality,
    "severity": _severity,
    "temporal_change": _temporal,
    "measurement": _measurement,
    "entity_substitution": _entity_substitution,
    "unsupported_insertion": _unsupported,
}


def mutation_types() -> tuple[str, ...]:
    return tuple(_MUTATORS)


def _changed_span(original: str, mutated: str) -> tuple[str, str, int, int, int, int]:
    prefix = 0
    maximum_prefix = min(len(original), len(mutated))
    while prefix < maximum_prefix and original[prefix] == mutated[prefix]:
        prefix += 1
    suffix = 0
    maximum_suffix = min(len(original) - prefix, len(mutated) - prefix)
    while suffix < maximum_suffix and original[-(suffix + 1)] == mutated[-(suffix + 1)]:
        suffix += 1
    original_end = len(original) - suffix
    mutated_end = len(mutated) - suffix
    return (
        original[prefix:original_end],
        mutated[prefix:mutated_end],
        prefix,
        original_end,
        prefix,
        mutated_end,
    )


def _stable_seed(seed: int, report_id: str, claim: str, operation: str) -> int:
    material = f"{seed}\0{report_id}\0{claim}\0{operation}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def generate_controlled_mutations(
    source_text: str,
    claim: str,
    report_id: str,
    *,
    seed: int = 42,
    include_types: Sequence[str] | None = None,
    max_mutations: int | None = None,
) -> list[Mutation]:
    """Generate deterministic, auditable counterfactuals for one claim.

    This function does not know the split.  For leakage safety it should only be
    called *after* report-level splitting (the preparation CLI does this).
    """

    source = normalize_text(source_text)
    original = normalize_text(claim)
    if not source or not original:
        return []
    operations = list(include_types or mutation_types())
    unknown = sorted(set(operations) - set(_MUTATORS))
    if unknown:
        raise ValueError(f"Unknown mutation types: {unknown}")

    generated: list[Mutation] = []
    seen: set[str] = {original.casefold()}
    for operation in operations:
        rng = random.Random(_stable_seed(seed, report_id, original, operation))
        candidate = _MUTATORS[operation](original, source, rng)
        if candidate is None:
            continue
        mutated = normalize_text(candidate.text)
        if not mutated or mutated.casefold() in seen:
            continue
        seen.add(mutated.casefold())
        changed = _changed_span(original, mutated)
        digest = hashlib.sha256(
            f"{report_id}\0{operation}\0{original}\0{mutated}".encode()
        ).hexdigest()[:16]
        generated.append(
            Mutation(
                mutation_id=f"mut-{digest}",
                report_id=str(report_id),
                mutation_type=candidate.mutation_type,
                original_claim=original,
                mutated_claim=mutated,
                label=candidate.label,
                risk_tier=candidate.risk_tier,
                risk_weight=candidate.risk_weight,
                description=candidate.description,
                changed_original_text=changed[0],
                changed_mutated_text=changed[1],
                changed_original_start=changed[2],
                changed_original_end=changed[3],
                changed_mutated_start=changed[4],
                changed_mutated_end=changed[5],
            )
        )
    if max_mutations is not None:
        if max_mutations < 0:
            raise ValueError("max_mutations must be non-negative or None")
        generated = generated[:max_mutations]
    return generated


def mutation_counts(mutations: Iterable[Mutation]) -> dict[str, int]:
    counts = {name: 0 for name in mutation_types()}
    for mutation in mutations:
        counts[mutation.mutation_type] = counts.get(mutation.mutation_type, 0) + 1
    return counts
