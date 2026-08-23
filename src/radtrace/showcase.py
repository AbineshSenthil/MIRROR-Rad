"""Curated, de-identified teaching cases backed by the local OpenI corpus.

The source reports are copied from the leakage-controlled OpenI/NLMCXR test
split already distributed with the project. Candidate summaries are either the
original impression or an explicitly controlled alteration. These cases make
the live workflow easy to explain; they are never counted as new evaluation
evidence.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShowcaseCase:
    """One safe report pair used by the guided frontend workflow."""

    case_id: str
    title: str
    focus: str
    description: str
    findings: str
    impression: str
    origin_report_id: str
    generator: str = "controlled teaching construction"
    evidence_status: str = "controlled showcase"


SHOWCASE_CASES: tuple[ShowcaseCase, ...] = (
    ShowcaseCase(
        case_id="showcase-supported-012",
        title="Supported summary",
        focus="supported",
        description="A faithful concise impression for a normal chest report.",
        findings=(
            "Lungs are clear bilaterally. Cardiac and mediastinal silhouettes are normal. "
            "Pulmonary vasculature is normal. No pneumothorax or pleural effusion. "
            "No acute bony abnormality."
        ),
        impression="No acute cardiopulmonary abnormality.",
        origin_report_id="OpenI test report 12",
    ),
    ShowcaseCase(
        case_id="showcase-negation-012",
        title="Negation reversal",
        focus="negation",
        description="The summary changes an explicitly absent pneumothorax into a present finding.",
        findings=(
            "Lungs are clear bilaterally. Cardiac and mediastinal silhouettes are normal. "
            "Pulmonary vasculature is normal. No pneumothorax or pleural effusion. "
            "No acute bony abnormality."
        ),
        impression="A small right pneumothorax is present. No pleural effusion.",
        origin_report_id="OpenI test report 12",
    ),
    ShowcaseCase(
        case_id="showcase-laterality-025",
        title="Laterality conflict",
        focus="laterality",
        description="Left-sided disease and effusion dominance are changed to the right side.",
        findings=(
            "The heart is within normal limits in size. Surgical suture material projects over "
            "the right lung apex. The lungs are hyperlucent and hyperinflated compatible with "
            "emphysema. There is left lower lobe airspace disease identified. There is moderate "
            "left pleural effusion and small right pleural effusion. No visualized pneumothorax."
        ),
        impression=(
            "Right lower lobe airspace disease with bilateral pleural effusions, "
            "right greater than left."
        ),
        origin_report_id="OpenI test report 25",
    ),
    ShowcaseCase(
        case_id="showcase-severity-025",
        title="Severity change",
        focus="severity",
        description="Moderate and small effusions are overstated as large and moderate.",
        findings=(
            "The heart is within normal limits in size. Surgical suture material projects over "
            "the right lung apex. The lungs are hyperlucent and hyperinflated compatible with "
            "emphysema. There is left lower lobe airspace disease identified. There is moderate "
            "left pleural effusion and small right pleural effusion. No visualized pneumothorax."
        ),
        impression="Large left pleural effusion and moderate right pleural effusion.",
        origin_report_id="OpenI test report 25",
    ),
    ShowcaseCase(
        case_id="showcase-temporal-327",
        title="Temporal and severity conflict",
        focus="temporal_change",
        description="Stable mild cardiomegaly is rewritten as new and severe.",
        findings=(
            "There is stable, mild cardiomegaly with normal caliber pulmonary vasculature. "
            "There is no focal airspace disease, pneumothorax, or pleural effusion. "
            "There are no acute bony findings."
        ),
        impression="New severe cardiomegaly. No acute pulmonary findings.",
        origin_report_id="OpenI test report 327",
    ),
    ShowcaseCase(
        case_id="showcase-omission-025",
        title="Important omission",
        focus="omission",
        description="The summary preserves airspace disease but drops both pleural effusions.",
        findings=(
            "The heart is within normal limits in size. The lungs are hyperlucent and "
            "hyperinflated compatible with emphysema. There is left lower lobe airspace disease. "
            "There is moderate left pleural effusion and small right pleural effusion. "
            "No visualized pneumothorax."
        ),
        impression="Left lower lobe airspace disease.",
        origin_report_id="OpenI test report 25",
    ),
    ShowcaseCase(
        case_id="showcase-unsupported-014",
        title="Unsupported insertion",
        focus="unsupported_insertion",
        description="A pulmonary mass is added even though it is not reported in the source.",
        findings=(
            "Heart size is within normal limits. Mild hyperinflation appears similar to prior. "
            "No focal alveolar consolidation or definite pleural effusion is seen. "
            "There are no typical findings of pulmonary edema."
        ),
        impression="No acute findings. A 2 cm right upper lobe mass is present.",
        origin_report_id="OpenI test report 14",
    ),
    ShowcaseCase(
        case_id="showcase-llm-omission-417",
        title="Genuine LLM summary with omissions",
        focus="real_llm_unannotated",
        description=(
            "A locally generated Llama summary preserves cardiomegaly but drops the negative "
            "pulmonary findings; it is intentionally treated as unannotated."
        ),
        findings=(
            "There is mild cardiomegaly, similar to prior exams. No focal consolidation. "
            "No visible pleural effusion or pneumothorax."
        ),
        impression=(
            "There is mild cardiomegaly, which is a non-specific finding that may be related to "
            "various cardiac conditions. The cardiomegaly is consistent with prior findings, "
            "but the nature of the condition is unclear."
        ),
        origin_report_id="Local challenge report RC-4171663C9693",
        generator="Llama-3.2-1B-Instruct",
        evidence_status="genuine generated; unannotated",
    ),
    ShowcaseCase(
        case_id="showcase-llm-preservation-0c33",
        title="Genuine LLM preservation case",
        focus="real_llm_unannotated",
        description=(
            "A locally generated FLAN-T5 summary preserves several normal findings; no human "
            "correctness label is claimed."
        ),
        findings=(
            "The cardiac and mediastinal silhouettes are unremarkable. The lungs are well expanded "
            "and clear. There is no focal air space opacity, pneumothorax, or effusion. "
            "The bony structures of the thorax are intact with no evidence of acute abnormality."
        ),
        impression=(
            "There is no focal air space opacity, pneumothorax, or effusion. The bony structures "
            "of the thorax are intact with no evidence of acute abnormality."
        ),
        origin_report_id="Local challenge report RC-0C33AAF62307",
        generator="FLAN-T5-base",
        evidence_status="genuine generated; unannotated",
    ),
)

SHOWCASE_CASES_BY_ID = {case.case_id: case for case in SHOWCASE_CASES}


def showcase_dataset_csv() -> str:
    """Serialize the curated cases in the application's upload schema."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=("case_id", "findings", "impression", "generator", "site", "split", "subgroup"),
    )
    writer.writeheader()
    for case in SHOWCASE_CASES:
        writer.writerow(
            {
                "case_id": case.case_id,
                "findings": case.findings,
                "impression": case.impression,
                "generator": case.generator,
                "site": "OpenI NLMCXR public corpus",
                "split": case.evidence_status,
                "subgroup": case.focus,
            }
        )
    return output.getvalue()
