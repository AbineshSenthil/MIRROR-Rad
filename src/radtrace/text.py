"""Deterministic, dependency-free text utilities for radiology reports.

The functions in this module deliberately avoid destructive preprocessing.  In
particular, negation cues, decimal measurements, units, and uncertainty terms
are retained because they are clinically meaningful for factuality checking.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_SPACE_RE = re.compile(r"[^\S\r\n]+")
_BLANK_LINE_RE = re.compile(r"\s*\n\s*")
_REDACTION_RE = re.compile(r"(?<![A-Za-z])X{3,}(?![A-Za-z])", re.IGNORECASE)
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
# Abbreviation periods that do not end a sentence.  Each pattern requires a
# non-letter before the abbreviation so word endings such as "aneurysms." or
# "symptoms." are never mistaken for "ms.".  "No." is only an abbreviation when
# a number follows ("No. 2"); otherwise it is the clinical negation "no".
_ABBREVIATION_PATTERNS = (
    re.compile(r"(?<![A-Za-z])(?:dr|mr|mrs|ms|vs|approx)\.", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])(?:e\.g|i\.e)\.", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])no\.(?=\s*\d)", re.IGNORECASE),
)
# Enumeration markers such as "1." or "2)" at the start of the text or after a
# sentence boundary, and "2)" markers inside a run-on list.  Decimal values are
# already protected when these run, so "1.2 cm" is never treated as a marker.
_LEADING_LIST_MARKER_RE = re.compile(r"(^|\n|[.!?;:]\s+)\(?\d{1,2}[.)]\s+(?=\S)")
_INLINE_LIST_MARKER_RE = re.compile(r"(?<=\s)\(?\d{1,2}\)\s+(?=[A-Za-z])")
_MARKER_ONLY_RE = re.compile(r"^\(?\d{1,2}[.)]?$")
_SECTION_HEADER_RE = re.compile(r"^(?:findings?|impressions?)\s*:\s*", re.IGNORECASE)


def normalize_text(text: object | None) -> str:
    """Return stable display/model text without discarding clinical meaning.

    Normalization uses Unicode NFKC, canonical punctuation, an explicit
    ``[REDACTED]`` marker for the Open-i ``XXXX`` placeholder, and whitespace
    compaction.  It intentionally does *not* lowercase, stem, remove stopwords,
    or remove punctuation.
    """

    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\u00a0", " ")
    value = value.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
            }
        )
    )
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _REDACTION_RE.sub("[REDACTED]", value)
    value = _SPACE_RE.sub(" ", value)
    value = _BLANK_LINE_RE.sub("\n", value)
    return value.strip()


def _protect_periods(text: str) -> tuple[str, str]:
    """Protect decimals and common abbreviations before sentence splitting."""

    marker = "\ue000"
    protected = _DECIMAL_RE.sub(marker, text)
    for pattern in _ABBREVIATION_PATTERNS:
        protected = pattern.sub(lambda match: match.group(0).replace(".", marker), protected)
    # Initials such as "A. Smith" are not sentence boundaries.
    protected = re.sub(
        r"\b([A-Z])\.(?=\s+[A-Z][a-z])", rf"\1{marker}", protected
    )
    return protected, marker


def sentence_split(text: object | None) -> list[str]:
    """Split report text into sentences while preserving measurements.

    New lines are treated as explicit boundaries.  Full stops, question marks,
    and exclamation marks end a sentence when followed by whitespace.  Decimal
    values such as ``1.2 cm`` and common abbreviations are protected.
    """

    value = normalize_text(text)
    if not value:
        return []
    protected, marker = _protect_periods(value)
    # Numbered impressions ("1. No acute process. 2. Stable cardiomegaly.") put
    # each item on its own line so the markers never become claims.
    protected = _LEADING_LIST_MARKER_RE.sub(lambda match: match.group(1) + "\n", protected)
    protected = _INLINE_LIST_MARKER_RE.sub("\n", protected)
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+)", protected)
    sentences: list[str] = []
    for part in parts:
        restored = _SECTION_HEADER_RE.sub("", part.replace(marker, ".").strip(" \t"))
        if not restored or _MARKER_ONLY_RE.match(restored):
            continue
        if re.search(r"[A-Za-z0-9]", restored):
            sentences.append(restored)
    return sentences


def claim_split(text: object | None) -> list[str]:
    """Split an impression into compact claims.

    Sentence boundaries are always used.  Semicolons are additionally split,
    but commas and coordinating conjunctions are retained so that negation
    scope is not damaged (for example, ``no edema or pleural effusion``).
    """

    claims: list[str] = []
    for sentence in sentence_split(text):
        for part in re.split(r"\s*;\s*", sentence):
            claim = part.strip(" \t-\u2022")
            if claim and re.search(r"[A-Za-z0-9]", claim):
                claims.append(claim)
    return claims


def word_tokens(text: object | None) -> list[str]:
    """Tokenize without deleting negation or splitting decimal measurements."""

    value = normalize_text(text)
    return re.findall(
        r"\[REDACTED\]|\d+(?:\.\d+)?(?:\s?(?:mm|cm|m|ml|cc|%))?|"
        r"[A-Za-z]+(?:[-'][A-Za-z]+)*|[^\w\s]",
        value,
        flags=re.IGNORECASE,
    )


def join_nonempty(parts: Iterable[object | None], separator: str = " ") -> str:
    """Normalize and concatenate non-empty strings."""

    return separator.join(value for part in parts if (value := normalize_text(part)))
