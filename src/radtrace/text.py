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
_ABBREVIATIONS = (
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "vs.",
    "e.g.",
    "i.e.",
    "no.",
    "approx.",
)


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
    for abbreviation in _ABBREVIATIONS:
        protected = re.sub(
            re.escape(abbreviation),
            lambda match: match.group(0).replace(".", marker),
            protected,
            flags=re.IGNORECASE,
        )
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
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+)", protected)
    sentences: list[str] = []
    for part in parts:
        restored = part.replace(marker, ".").strip(" \t")
        if restored and re.search(r"[A-Za-z0-9]", restored):
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
