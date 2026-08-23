"""Open-i radiology report parsing, validation, splitting, and serialization."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .text import claim_split, normalize_text, sentence_split


class OpenIParseError(ValueError):
    """Raised when an Open-i XML report cannot be parsed."""


@dataclass(frozen=True)
class RadiologyReport:
    """A normalized, text-only view of an Open-i radiology report."""

    report_id: str
    uid: str
    pmc_id: str
    iuxr_id: str
    findings: str
    impression: str
    comparison: str = ""
    indication: str = ""
    mesh_major: tuple[str, ...] = field(default_factory=tuple)
    mesh_minor: tuple[str, ...] = field(default_factory=tuple)
    image_ids: tuple[str, ...] = field(default_factory=tuple)
    image_captions: tuple[str, ...] = field(default_factory=tuple)
    source_file: str = ""

    @property
    def is_usable(self) -> bool:
        return bool(self.findings and self.impression)

    @property
    def findings_sentences(self) -> list[str]:
        return sentence_split(self.findings)

    @property
    def impression_claims(self) -> list[str]:
        return claim_split(self.impression)

    @property
    def content_group_id(self) -> str:
        """Fingerprint exact text duplicates so they stay in one split."""

        canonical = f"{self.findings.casefold()}\n{self.impression.casefold()}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mesh_major"] = list(self.mesh_major)
        value["mesh_minor"] = list(self.mesh_minor)
        value["image_ids"] = list(self.image_ids)
        value["image_captions"] = list(self.image_captions)
        value["findings_sentences"] = self.findings_sentences
        value["impression_claims"] = self.impression_claims
        value["content_group_id"] = self.content_group_id
        return value


@dataclass(frozen=True)
class DatasetScan:
    """Result of scanning a directory, including auditable rejection reasons."""

    reports: tuple[RadiologyReport, ...]
    total_xml_files: int
    skipped_missing_findings: tuple[str, ...]
    skipped_missing_impression: tuple[str, ...]
    skipped_missing_both: tuple[str, ...]
    parse_errors: tuple[dict[str, str], ...]

    @property
    def usable_count(self) -> int:
        return len(self.reports)

    @property
    def skipped_count(self) -> int:
        return (
            len(self.skipped_missing_findings)
            + len(self.skipped_missing_impression)
            + len(self.skipped_missing_both)
            + len(self.parse_errors)
        )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return normalize_text(" ".join(element.itertext()))


def _first(root: ET.Element, local_name: str) -> ET.Element | None:
    wanted = local_name.casefold()
    return next(
        (node for node in root.iter() if _local_name(node.tag).casefold() == wanted),
        None,
    )


def _attribute_or_text(root: ET.Element, local_name: str, attribute: str) -> str:
    node = _first(root, local_name)
    if node is None:
        return ""
    return normalize_text(node.attrib.get(attribute) or _element_text(node))


def _abstract_sections(root: ET.Element) -> dict[str, str]:
    sections: defaultdict[str, list[str]] = defaultdict(list)
    for node in root.iter():
        if _local_name(node.tag).casefold() != "abstracttext":
            continue
        label = normalize_text(node.attrib.get("Label", "")).upper()
        text = _element_text(node)
        if label and text:
            sections[label].append(text)
    return {key: normalize_text(" ".join(values)) for key, values in sections.items()}


def parse_openi_xml(path: str | Path) -> RadiologyReport:
    """Parse one Open-i XML file without assuming an XML namespace."""

    xml_path = Path(path)
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise OpenIParseError(f"Could not parse {xml_path}: {exc}") from exc

    sections = _abstract_sections(root)
    report_id = _attribute_or_text(root, "IUXRId", "id") or xml_path.stem
    uid = _attribute_or_text(root, "uId", "id") or f"CXR{report_id}"
    pmc_id = _attribute_or_text(root, "pmcId", "id")
    iuxr_id = _attribute_or_text(root, "IUXRId", "id") or report_id

    major: list[str] = []
    minor: list[str] = []
    image_ids: list[str] = []
    image_captions: list[str] = []
    for node in root.iter():
        tag = _local_name(node.tag).casefold()
        if tag == "major":
            if value := _element_text(node):
                major.append(value)
        elif tag == "minor":
            if value := _element_text(node):
                minor.append(value)
        elif tag == "parentimage":
            if value := normalize_text(node.attrib.get("id")):
                image_ids.append(value)
            caption = next(
                (
                    _element_text(child)
                    for child in node.iter()
                    if _local_name(child.tag).casefold() == "caption"
                ),
                "",
            )
            if caption:
                image_captions.append(caption)

    return RadiologyReport(
        report_id=report_id,
        uid=uid,
        pmc_id=pmc_id,
        iuxr_id=iuxr_id,
        findings=sections.get("FINDINGS", ""),
        impression=sections.get("IMPRESSION", ""),
        comparison=sections.get("COMPARISON", ""),
        indication=sections.get("INDICATION", ""),
        mesh_major=tuple(dict.fromkeys(major)),
        mesh_minor=tuple(dict.fromkeys(minor)),
        image_ids=tuple(dict.fromkeys(image_ids)),
        image_captions=tuple(image_captions),
        source_file=xml_path.name,
    )


def _natural_path_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.stem)
    )


def scan_openi_dataset(directory: str | Path) -> DatasetScan:
    """Parse all XML files and retain only reports with both target sections."""

    source_dir = Path(directory)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {source_dir}")
    paths = sorted(source_dir.rglob("*.xml"), key=_natural_path_key)
    if not paths:
        raise FileNotFoundError(f"No XML files found under: {source_dir}")

    reports: list[RadiologyReport] = []
    missing_findings: list[str] = []
    missing_impression: list[str] = []
    missing_both: list[str] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            report = parse_openi_xml(path)
        except OpenIParseError as exc:
            errors.append({"source_file": path.name, "error": str(exc)})
            continue
        if report.findings and report.impression:
            reports.append(report)
        elif not report.findings and not report.impression:
            missing_both.append(path.name)
        elif not report.findings:
            missing_findings.append(path.name)
        else:
            missing_impression.append(path.name)

    ids = [report.report_id for report in reports]
    duplicate_ids = [key for key, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate report IDs found: {duplicate_ids[:10]}")
    return DatasetScan(
        reports=tuple(reports),
        total_xml_files=len(paths),
        skipped_missing_findings=tuple(missing_findings),
        skipped_missing_impression=tuple(missing_impression),
        skipped_missing_both=tuple(missing_both),
        parse_errors=tuple(errors),
    )


def load_openi_reports(
    directory: str | Path, *, require_both_sections: bool = True
) -> list[RadiologyReport]:
    """Load normalized reports.

    ``require_both_sections=False`` parses every valid XML document, which is
    useful for auditing.  Training code should keep the default.
    """

    if require_both_sections:
        return list(scan_openi_dataset(directory).reports)
    source_dir = Path(directory)
    paths = sorted(source_dir.rglob("*.xml"), key=_natural_path_key)
    return [parse_openi_xml(path) for path in paths]


def split_reports(
    reports: Sequence[RadiologyReport],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[RadiologyReport]]:
    """Make a deterministic, duplicate-aware report-level split.

    Exact Findings/Impression duplicates are grouped before assignment.  This
    prevents duplicated reports, as well as all future mutations derived from
    them, from crossing split boundaries.
    """

    ratios = {
        "train": float(train_ratio),
        "validation": float(validation_ratio),
        "test": float(test_ratio),
    }
    if any(value < 0 for value in ratios.values()) or not abs(sum(ratios.values()) - 1) < 1e-9:
        raise ValueError("train, validation, and test ratios must be non-negative and sum to 1")
    if not reports:
        return {name: [] for name in ratios}

    grouped: defaultdict[str, list[RadiologyReport]] = defaultdict(list)
    for report in reports:
        grouped[report.content_group_id].append(report)
    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)

    targets = {name: ratios[name] * len(reports) for name in ratios}
    result: dict[str, list[RadiologyReport]] = {name: [] for name in ratios}
    for group in groups:
        eligible = [name for name, ratio in ratios.items() if ratio > 0]
        destination = max(
            eligible,
            key=lambda name: (
                (targets[name] - len(result[name])) / max(targets[name], 1.0),
                targets[name] - len(result[name]),
                -list(ratios).index(name),
            ),
        )
        result[destination].extend(group)

    for values in result.values():
        values.sort(key=lambda report: _natural_path_key(Path(report.source_file)))
    _assert_split_integrity(result, expected_count=len(reports))
    return result


def _assert_split_integrity(
    splits: Mapping[str, Sequence[RadiologyReport]], *, expected_count: int
) -> None:
    ids_by_split = {
        name: {report.report_id for report in reports}
        for name, reports in splits.items()
    }
    group_by_split = {
        name: {report.content_group_id for report in reports}
        for name, reports in splits.items()
    }
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if ids_by_split[left] & ids_by_split[right]:
                raise AssertionError(f"Report leakage between {left} and {right}")
            if group_by_split[left] & group_by_split[right]:
                raise AssertionError(f"Duplicate-text leakage between {left} and {right}")
    if sum(len(values) for values in ids_by_split.values()) != expected_count:
        raise AssertionError("Split lost or duplicated reports")


def dataset_summary(scan: DatasetScan) -> dict[str, Any]:
    """Return JSON-serializable corpus statistics."""

    reports = scan.reports
    findings_sentences = sum(len(report.findings_sentences) for report in reports)
    impression_claims = sum(len(report.impression_claims) for report in reports)
    duplicate_groups = Counter(report.content_group_id for report in reports)
    return {
        "total_xml_files": scan.total_xml_files,
        "usable_reports": scan.usable_count,
        "skipped_reports": scan.skipped_count,
        "missing_findings_only": len(scan.skipped_missing_findings),
        "missing_impression_only": len(scan.skipped_missing_impression),
        "missing_both": len(scan.skipped_missing_both),
        "parse_errors": len(scan.parse_errors),
        "findings_sentences": findings_sentences,
        "impression_claims": impression_claims,
        "reports_with_images": sum(bool(report.image_ids) for report in reports),
        "image_references": sum(len(report.image_ids) for report in reports),
        "unique_content_groups": len(duplicate_groups),
        "duplicate_content_groups": sum(count > 1 for count in duplicate_groups.values()),
    }


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write UTF-8 JSON Lines and return the row count."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=isinstance(value, dict))
    if value is None:
        return ""
    return value


def write_csv(path: str | Path, records: Sequence[Mapping[str, Any]]) -> int:
    """Write a stable UTF-8 CSV, JSON-encoding nested fields."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(record) for record in records]
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return len(rows)

