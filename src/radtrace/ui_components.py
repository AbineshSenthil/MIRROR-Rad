"""Framework-independent, escaped presentation helpers for MIRROR-Rad."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from html import escape
from typing import Any

LABEL_META: dict[str, dict[str, str]] = {
    "ENTAILED": {
        "display": "Supported",
        "icon": "✓",
        "class": "supported",
        "description": "The claim is supported by the supplied reference text.",
    },
    "CONTRADICTED": {
        "display": "Contradicted",
        "icon": "!",
        "class": "contradicted",
        "description": "The claim conflicts with the supplied reference text.",
    },
    "UNSUPPORTED": {
        "display": "Unsupported",
        "icon": "?",
        "class": "unsupported",
        "description": "No adequate support was found in the supplied reference text.",
    },
    "REVIEW": {
        "display": "Review",
        "icon": "~",
        "class": "review",
        "description": "The system abstained because the claim score was not reliable enough.",
    },
}

RISK_META: dict[str, dict[str, str]] = {
    "CRITICAL": {"display": "Critical priority", "class": "critical"},
    "HIGH": {"display": "High priority", "class": "high"},
    "MEDIUM": {"display": "Medium priority", "class": "medium"},
    "LOW": {"display": "Low priority", "class": "low"},
    "UNSPECIFIED": {"display": "Priority not assigned", "class": "unspecified"},
}


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    for method_name in ("model_dump", "dict", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                converted = method()
                if isinstance(converted, Mapping):
                    return dict(converted)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
    try:
        return dict(vars(value))
    except (TypeError, ValueError):
        return {}


def to_serializable(value: Any) -> Any:
    """Recursively convert model values to standards-compliant JSON values."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return to_serializable(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    for method_name in ("model_dump", "dict", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return to_serializable(method())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
    try:
        return to_serializable(vars(value))
    except (TypeError, ValueError):
        return str(value)


def _first(data: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def _clean_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return re.sub(r"\s+", " ", str(value)).strip() or fallback


def _bounded_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if 1.0 < number <= 100.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


def normalize_label(value: Any, confidence: float = 1.0, review_threshold: float = 0.0) -> str:
    raw = _clean_text(getattr(value, "value", value), "REVIEW").upper()
    raw = re.sub(r"[^A-Z]+", "_", raw).strip("_")
    aliases = {
        "ENTAILMENT": "ENTAILED",
        "ENTAIL": "ENTAILED",
        "SUPPORTED": "ENTAILED",
        "SUPPORT": "ENTAILED",
        "TRUE": "ENTAILED",
        "CONTRADICTION": "CONTRADICTED",
        "CONTRADICT": "CONTRADICTED",
        "CONFLICT": "CONTRADICTED",
        "FALSE": "CONTRADICTED",
        "NEUTRAL": "UNSUPPORTED",
        "NOT_ENOUGH_INFO": "UNSUPPORTED",
        "NOT_SUPPORTED": "UNSUPPORTED",
        "UNKNOWN": "REVIEW",
        "ABSTAIN": "REVIEW",
        "UNCERTAIN": "REVIEW",
    }
    label = aliases.get(raw, raw)
    if label not in LABEL_META:
        label = "REVIEW"
    if review_threshold > 0 and confidence < review_threshold:
        return "REVIEW"
    return label


def normalize_risk(value: Any) -> str:
    raw = _clean_text(getattr(value, "value", value), "UNSPECIFIED").upper()
    raw = re.sub(r"[^A-Z]+", "_", raw).strip("_")
    aliases = {
        "SEVERE": "HIGH",
        "MODERATE": "MEDIUM",
        "MID": "MEDIUM",
        "MINOR": "LOW",
        "NONE": "LOW",
        "NA": "UNSPECIFIED",
        "N_A": "UNSPECIFIED",
        "UNKNOWN": "UNSPECIFIED",
    }
    result = aliases.get(raw, raw)
    return result if result in RISK_META else "UNSPECIFIED"


def normalize_spans(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping) or not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        value = [value]
    output: list[dict[str, Any]] = []
    for item in value:
        data = _mapping(item)
        if not data and isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            data = {"start": item[0], "end": item[1]}
        start = _first(data, ("start", "start_char", "begin", "offset_start"))
        end = _first(data, ("end", "end_char", "stop", "offset_end"))
        text_value = _first(data, ("text", "span_text", "value"), "")
        target = _clean_text(_first(data, ("target", "field", "source", "role"), "evidence")).lower()
        try:
            start_int = int(start) if start is not None else None
            end_int = int(end) if end is not None else None
        except (TypeError, ValueError):
            start_int = end_int = None
        if start_int is not None and end_int is not None and start_int >= 0 and end_int > start_int:
            output.append(
                {"start": start_int, "end": end_int, "text": _clean_text(text_value), "target": target}
            )
        elif _clean_text(text_value):
            output.append({"start": None, "end": None, "text": _clean_text(text_value), "target": target})
    return output


def _normalize_error_types(value: Any) -> str:
    if value is None:
        return "Not specified"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [
            _clean_text(getattr(item, "value", item))
            for item in value
            if _clean_text(getattr(item, "value", item)) not in {"", "NONE"}
        ]
        return ", ".join(parts) if parts else "Not specified"
    cleaned = _clean_text(getattr(value, "value", value), "Not specified")
    return "Not specified" if cleaned == "NONE" else cleaned


def normalize_claim(
    claim: Any,
    index: int = 0,
    review_threshold: float = 0.0,
) -> dict[str, Any]:
    data = _mapping(claim)
    raw_evidence = _first(data, ("evidence", "evidence_matches", "retrieved_evidence", "matches"))
    top_evidence: dict[str, Any] = {}
    if isinstance(raw_evidence, Sequence) and not isinstance(raw_evidence, (str, bytes)):
        top_evidence = _mapping(raw_evidence[0]) if raw_evidence else {}
    elif isinstance(raw_evidence, Mapping):
        top_evidence = _mapping(raw_evidence)
    elif isinstance(raw_evidence, str):
        top_evidence = {"text": raw_evidence}
    confidence = _bounded_number(_first(data, ("confidence", "probability", "score"), 0.0))
    label = normalize_label(
        _first(data, ("label", "prediction", "status", "nli_label")),
        confidence,
        review_threshold,
    )
    evidence_score = _bounded_number(
        _first(data, ("evidence_score", "retrieval_score", "similarity", "support_score"), 0.0)
    )
    return {
        "index": index,
        "claim_id": _clean_text(_first(data, ("claim_id", "id")), f"C{index + 1:03d}"),
        "claim_text": _clean_text(
            _first(data, ("claim_text", "claim", "text", "hypothesis", "sentence")),
            f"Claim {index + 1}",
        ),
        "label": label,
        "confidence": confidence,
        "risk_tier": normalize_risk(_first(data, ("risk_tier", "risk", "severity"))),
        "error_type": _normalize_error_types(
            _first(data, ("error_type", "error_types", "error_category", "category", "reason_code"))
        ),
        "evidence_text": _clean_text(
            _first(data, ("evidence_text", "source_text", "premise", "matched_sentence"))
            or _first(top_evidence, ("text", "evidence_text", "sentence", "premise")),
            "No evidence sentence was returned.",
        ),
        "evidence_score": evidence_score
        or _bounded_number(
            _first(top_evidence, ("score", "evidence_score", "retrieval_score", "similarity"), 0.0)
        ),
        "rationale": _clean_text(
            _first(data, ("rationale", "explanation", "reason", "notes")),
            LABEL_META[label]["description"],
        ),
        "spans": normalize_spans(
            _first(data, ("spans", "evidence_spans", "highlights", "matched_spans"))
        ),
        "probabilities": to_serializable(
            _first(data, ("probabilities", "class_probabilities"), {})
        ),
        "evidence": to_serializable(raw_evidence or []),
        "rule_flags": to_serializable(_first(data, ("rule_flags", "rules", "features"), {})),
    }


def derive_overall_label(claims: Sequence[Mapping[str, Any]]) -> str:
    labels = {normalize_label(claim.get("label")) for claim in claims}
    if "CONTRADICTED" in labels:
        return "CONTRADICTED"
    if "UNSUPPORTED" in labels:
        return "UNSUPPORTED"
    if "REVIEW" in labels or not labels:
        return "REVIEW"
    return "ENTAILED"


def normalize_audit_result(
    result: Any,
    findings: str = "",
    impression: str = "",
    review_threshold: float = 0.0,
) -> dict[str, Any]:
    data = _mapping(result)
    if not data and isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        data = {"claims": result}
    raw_claims = _first(
        data,
        ("claims", "claim_results", "sentence_results", "predictions", "results"),
        [],
    )
    if isinstance(raw_claims, Mapping):
        raw_claims = list(raw_claims.values())
    if not isinstance(raw_claims, Sequence) or isinstance(raw_claims, (str, bytes)):
        raw_claims = [raw_claims] if raw_claims else []
    claims = [normalize_claim(item, idx, review_threshold) for idx, item in enumerate(raw_claims)]
    latency_value = _first(data, ("latency_ms", "inference_ms", "elapsed_ms", "processing_time_ms"))
    try:
        latency_ms = round(float(latency_value), 1) if latency_value is not None else None
    except (TypeError, ValueError):
        latency_ms = None
    model_info = _mapping(_first(data, ("model_info", "model_details"), {}))
    metadata = _mapping(_first(data, ("metadata", "meta", "diagnostics"), {}))
    metadata.update(
        {
            "model_info": to_serializable(model_info),
            "source_summary": to_serializable(_first(data, ("summary", "audit_summary"), {})),
            "schema_version": _clean_text(_first(data, ("schema_version",), "1.0")),
        }
    )
    raw_overall = _first(data, ("overall_label", "label", "status"))
    return {
        "findings": _clean_text(_first(data, ("findings", "source", "premise")), _clean_text(findings)),
        "impression": _clean_text(
            _first(data, ("impression", "hypothesis", "candidate_summary")),
            _clean_text(impression),
        ),
        "claims": claims,
        "overall_label": normalize_label(
            raw_overall,
            _bounded_number(_first(data, ("overall_confidence", "confidence"), 1.0)),
        )
        if raw_overall is not None
        else derive_overall_label(claims),
        "model_name": _clean_text(
            _first(data, ("model_name", "model", "backend"))
            or _first(model_info, ("model_name", "name", "nli_backend", "backend")),
            "MIRROR-Rad",
        ),
        "latency_ms": latency_ms,
        "metadata": to_serializable(metadata),
    }


def summarize_claims(claims: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {label: 0 for label in LABEL_META}
    risks = {risk: 0 for risk in RISK_META}
    confidences: list[float] = []
    for item in claims:
        claim = normalize_claim(item, len(confidences))
        counts[claim["label"]] += 1
        risks[claim["risk_tier"]] += 1
        confidences.append(claim["confidence"])
    total = len(claims)
    return {
        "total": total,
        "flagged": counts["CONTRADICTED"] + counts["UNSUPPORTED"] + counts["REVIEW"],
        "supported_rate": counts["ENTAILED"] / total if total else 0.0,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "counts": counts,
        "risks": risks,
        "high_risk_flags": sum(
            1
            for item in claims
            if normalize_label(item.get("label")) != "ENTAILED"
            and normalize_risk(item.get("risk_tier")) in {"HIGH", "CRITICAL"}
        ),
    }


def confidence_text(value: Any) -> str:
    return f"{_bounded_number(value) * 100:.0f}%"


def _display_error_type(value: str) -> str:
    cleaned = _clean_text(value, "Not specified").replace("_", " ").replace("-", " ")
    return cleaned[:1].upper() + cleaned[1:]


def _span_ranges(text: str, spans: Any, target: str = "evidence") -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for span in normalize_spans(spans):
        span_target = span.get("target", "evidence")
        accepted_targets = {target, "both", ""}
        if target == "evidence":
            accepted_targets.add("source")
        elif target == "claim":
            accepted_targets.update({"hypothesis", "summary"})
        if target and span_target not in accepted_targets:
            continue
        start, end = span.get("start"), span.get("end")
        if isinstance(start, int) and isinstance(end, int) and start < len(text) and end <= len(text):
            ranges.append((start, end))
            continue
        needle = span.get("text", "")
        if needle:
            match = re.search(re.escape(needle), text, flags=re.IGNORECASE)
            if match:
                ranges.append(match.span())
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def highlight_text_html(text: Any, spans: Any = None, target: str = "evidence") -> str:
    clean = str(text or "")
    ranges = _span_ranges(clean, spans, target)
    if not ranges:
        return escape(clean)
    pieces: list[str] = []
    cursor = 0
    for start, end in ranges:
        pieces.append(escape(clean[cursor:start]))
        pieces.append(f'<mark class="evidence-highlight">{escape(clean[start:end])}</mark>')
        cursor = end
    pieces.append(escape(clean[cursor:]))
    return "".join(pieces)


def claim_card_html(claim: Mapping[str, Any], ordinal: int | None = None) -> str:
    item = normalize_claim(claim, (ordinal or 1) - 1)
    label_meta = LABEL_META[item["label"]]
    risk_meta = RISK_META[item["risk_tier"]]
    number = ordinal if ordinal is not None else item["index"] + 1
    safe_claim = highlight_text_html(item["claim_text"], item["spans"], target="claim")
    return (
        f'<article class="claim-card claim-{label_meta["class"]}" '
        f'aria-label="Claim {number}: {escape(label_meta["display"])}">'
        '<div class="claim-card-top">'
        f'<span class="claim-index">CLAIM {number:02d}</span>'
        f'<span class="status-badge status-{label_meta["class"]}">'
        f'<span aria-hidden="true">{label_meta["icon"]}</span> {escape(label_meta["display"])}</span>'
        "</div>"
        f'<p class="claim-text">{safe_claim}</p>'
        '<div class="claim-meta">'
        f'<span class="risk-pill risk-{risk_meta["class"]}">{escape(risk_meta["display"])}</span>'
        f'<span><b>{confidence_text(item["confidence"])}</b> claim score</span>'
        f'<span>{escape(_display_error_type(item["error_type"]))}</span>'
        "</div></article>"
    )


def evidence_panel_html(claim: Mapping[str, Any]) -> str:
    item = normalize_claim(claim)
    highlighted = highlight_text_html(item["evidence_text"], item["spans"])
    return (
        '<div class="evidence-panel">'
        '<div class="evidence-kicker">Retrieved source evidence</div>'
        f"<blockquote>{highlighted}</blockquote>"
        f'<div class="evidence-score">Retrieval match <b>{confidence_text(item["evidence_score"])}</b></div>'
        '<div class="rationale-label">Decision rationale</div>'
        f'<p class="rationale-text">{escape(item["rationale"])}</p>'
        "</div>"
    )


def result_json(result: Any) -> str:
    return json.dumps(to_serializable(result), ensure_ascii=False, indent=2, sort_keys=True)


def _safe_spreadsheet_cell(value: str) -> str:
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def claims_to_csv(claims: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    fieldnames = [
        "claim_number",
        "claim_text",
        "label",
        "confidence",
        "risk_tier",
        "error_type",
        "evidence_text",
        "evidence_score",
        "rationale",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for index, claim in enumerate(claims):
        item = normalize_claim(claim, index)
        writer.writerow(
            {
                "claim_number": index + 1,
                "claim_text": _safe_spreadsheet_cell(item["claim_text"]),
                "label": item["label"],
                "confidence": f'{item["confidence"]:.4f}',
                "risk_tier": item["risk_tier"],
                "error_type": _safe_spreadsheet_cell(item["error_type"]),
                "evidence_text": _safe_spreadsheet_cell(item["evidence_text"]),
                "evidence_score": f'{item["evidence_score"]:.4f}',
                "rationale": _safe_spreadsheet_cell(item["rationale"]),
            }
        )
    return buffer.getvalue()
