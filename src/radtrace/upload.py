"""Safe, deterministic parsing for de-identified report-pair uploads.

The Streamlit boundary passes file bytes into this module.  Keeping parsing
outside the UI makes the accepted schema testable and prevents uploaded data
from being written to disk.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_CASES = max(1, int(os.getenv("MIRROR_RAD_MAX_UPLOAD_CASES", "100")))
SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl"}

_FIELD_ALIASES = {
    "case_id": ("case_id", "id", "report_id", "study_id"),
    "findings": ("findings", "source_findings", "report_findings"),
    "impression": (
        "impression",
        "candidate_impression",
        "generated_impression",
        "summary",
    ),
}
_OPTIONAL_METADATA = {
    "generator": ("generator", "generator_id", "model_name"),
    "site": ("site", "institution", "hospital"),
    "split": ("split", "dataset_split"),
    "subgroup": ("subgroup", "cohort", "group"),
}


class UploadValidationError(ValueError):
    """Raised when an uploaded file is unsafe or does not match the schema."""


def _normalized_key(value: Any) -> str:
    return str(value or "").lstrip("\ufeff").strip().casefold().replace(" ", "_")


def _text_value(value: Any, *, field: str, row_number: int) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        raise UploadValidationError(
            f"Row {row_number}: '{field}' must contain plain text, not a nested value."
        )
    return str(value).replace("\x00", "").strip()


def _extract_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        container = next(
            (
                payload[key]
                for key in ("records", "cases", "reports", "data")
                if key in payload and isinstance(payload[key], list)
            ),
            None,
        )
        rows = container if container is not None else [payload]
    else:
        raise UploadValidationError("JSON must contain an object or a list of report-pair objects.")
    if any(not isinstance(row, Mapping) for row in rows):
        raise UploadValidationError("Every uploaded record must be an object with named fields.")
    return list(rows)


def _parse_json(text: str) -> list[Mapping[str, Any]]:
    try:
        return _extract_rows(json.loads(text))
    except json.JSONDecodeError as exc:
        raise UploadValidationError(
            f"Invalid JSON near line {exc.lineno}, column {exc.colno}."
        ) from exc


def _parse_jsonl(text: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UploadValidationError(f"Invalid JSON on JSONL line {line_number}.") from exc
        if not isinstance(payload, Mapping):
            raise UploadValidationError(f"JSONL line {line_number} must be an object.")
        rows.append(payload)
    return rows


def _parse_csv(text: str) -> list[Mapping[str, Any]]:
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if not reader.fieldnames:
            raise UploadValidationError("The CSV file has no header row.")
        return list(reader)
    except csv.Error as exc:
        raise UploadValidationError(f"Invalid CSV: {exc}.") from exc


def _canonicalize(rows: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    if not rows:
        raise UploadValidationError("The uploaded file contains no report pairs.")
    if len(rows) > MAX_UPLOAD_CASES:
        raise UploadValidationError(
            f"Upload at most {MAX_UPLOAD_CASES} cases at a time; this file contains {len(rows)}."
        )

    normalized: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for row_index, raw_row in enumerate(rows, start=1):
        keyed = {_normalized_key(key): value for key, value in raw_row.items()}

        values: dict[str, str] = {}
        for canonical, aliases in _FIELD_ALIASES.items():
            raw_value = next((keyed[alias] for alias in aliases if alias in keyed), None)
            values[canonical] = _text_value(raw_value, field=canonical, row_number=row_index)

        missing = [field for field in ("findings", "impression") if not values[field]]
        if missing:
            joined = " and ".join(missing)
            raise UploadValidationError(f"Row {row_index}: missing {joined} text.")

        case_id = values["case_id"] or f"case-{row_index:03d}"
        if case_id in seen_ids:
            raise UploadValidationError(
                f"Duplicate case_id '{case_id}'. Use a unique identifier for every report pair."
            )
        seen_ids.add(case_id)
        canonical_row = {
            "case_id": case_id,
            "findings": values["findings"],
            "impression": values["impression"],
        }
        for field, aliases in _OPTIONAL_METADATA.items():
            present = next((alias for alias in aliases if alias in keyed), None)
            if present is None:
                continue
            metadata = _text_value(keyed[present], field=field, row_number=row_index)
            if len(metadata) > 256:
                raise UploadValidationError(
                    f"Row {row_index}: optional metadata field '{field}' exceeds 256 characters."
                )
            canonical_row[field] = metadata
        normalized.append(canonical_row)
    return normalized


def parse_uploaded_cases(filename: str, data: bytes) -> list[dict[str, str]]:
    """Parse an in-memory CSV, JSON, or JSONL file into canonical report pairs."""

    extension = Path(filename).suffix.casefold()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError("Use a .csv, .json, or .jsonl file.")
    if not data:
        raise UploadValidationError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadValidationError("The uploaded file exceeds the 10 MB application limit.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UploadValidationError("Save the file as UTF-8 text before uploading it.") from exc

    if extension == ".csv":
        rows = _parse_csv(text)
    elif extension == ".jsonl":
        rows = _parse_jsonl(text)
    else:
        rows = _parse_json(text)
    return _canonicalize(rows)


def upload_template_csv() -> str:
    """Return a minimal, de-identified upload template."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=("case_id", "findings", "impression", "generator", "site", "split", "subgroup"),
    )
    writer.writeheader()
    writer.writerow(
        {
            "case_id": "case-001",
            "findings": "No focal airspace opacity, pleural effusion, or pneumothorax.",
            "impression": "No acute cardiopulmonary abnormality.",
            "generator": "",
            "site": "",
            "split": "",
            "subgroup": "",
        }
    )
    return output.getvalue()


def batch_summary_csv(rows: list[dict[str, Any]]) -> str:
    """Serialize flattened batch summaries without exposing nested model structures."""

    if not rows:
        return ""
    fieldnames = (
        "case_id",
        "status",
        "overall_label",
        "claims",
        "flagged",
        "supported",
        "contradicted",
        "unsupported",
        "review",
        "supported_rate",
        "mean_confidence",
        "latency_ms",
        "error",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(
        {
            key: _safe_spreadsheet_cell(value) if isinstance(value, str) else value
            for key, value in row.items()
        }
        for row in rows
    )
    return output.getvalue()


def _safe_spreadsheet_cell(value: str) -> str:
    """Prevent exported user text from being interpreted as a spreadsheet formula."""

    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value
