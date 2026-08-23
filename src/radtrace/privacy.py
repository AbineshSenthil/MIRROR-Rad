"""Conservative local preflight checks for obvious patient identifiers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PrivacyFlag:
    category: str
    start: int
    end: int
    guidance: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "medical_record_number",
        re.compile(
            r"\b(?:mrn|medical\s+record(?:\s+number)?|patient\s+id)\s*[:#-]?\s*[a-z0-9][a-z0-9-]{4,}\b",
            re.IGNORECASE,
        ),
        "Remove the medical record or patient identifier.",
    ),
    (
        "date_of_birth",
        re.compile(
            r"\b(?:dob|date\s+of\s+birth|born)\s*[:#-]?\s*(?:\d{1,2}[/-]){2}\d{2,4}\b",
            re.IGNORECASE,
        ),
        "Remove the date of birth.",
    ),
    (
        "email_address",
        re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE),
        "Remove the email address.",
    ),
    (
        "telephone_number",
        re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)"),
        "Remove the telephone number.",
    ),
    (
        "government_identifier",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "Remove the government identifier.",
    ),
    (
        "explicit_patient_name",
        re.compile(r"\b(?:patient\s+name|name\s+of\s+patient)\s*[:#-]\s*[^\n,;]{2,60}", re.IGNORECASE),
        "Remove the explicit patient name field.",
    ),
)


def scan_privacy_risk(text: str) -> list[PrivacyFlag]:
    """Return identifier categories and offsets without copying matched text."""

    value = str(text or "")
    output: list[PrivacyFlag] = []
    occupied: set[tuple[int, int, str]] = set()
    for category, pattern, guidance in _PATTERNS:
        for match in pattern.finditer(value):
            identity = (match.start(), match.end(), category)
            if identity in occupied:
                continue
            occupied.add(identity)
            output.append(
                PrivacyFlag(
                    category=category,
                    start=match.start(),
                    end=match.end(),
                    guidance=guidance,
                )
            )
    return sorted(output, key=lambda item: (item.start, item.end, item.category))
