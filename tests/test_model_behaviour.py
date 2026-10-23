"""End-to-end behaviour of the shipped model (skipped without the checkpoint).

These lock in the review-routing fix: concise faithful impressions must not be
escalated, and invented, contradicted or missing critical findings must be.
"""

import pytest

from radtrace.showcase import SHOWCASE_CASES_BY_ID

NORMAL = (
    "The heart size and mediastinal contours are within normal limits. The lungs are clear. "
    "There is no pleural effusion or pneumothorax. Mild degenerative changes of the thoracic spine."
)

FAITHFUL = [
    (NORMAL, NORMAL),
    (NORMAL, "No acute cardiopulmonary process."),
    (NORMAL, "No acute cardiopulmonary abnormality."),
    (NORMAL, "Normal chest radiograph."),
    (NORMAL, "1. No acute cardiopulmonary process. 2. Mild degenerative changes of the thoracic spine."),
    ("Small left pneumothorax. No pleural effusion. Heart size is normal.", "Small left pneumothorax."),
]

ERRORS = [
    ("Small left pneumothorax. No pleural effusion. Heart size is normal.", "Small right pneumothorax."),
    ("Two small pulmonary nodules. No pneumothorax.", "Two small pulmonary nodules. Large right pneumothorax."),
    ("Two small pulmonary aneurysms. No pneumothorax.", "Two small pulmonary aneurysms. Large right pneumothorax."),
    (NORMAL, "Large right pneumothorax."),
    ("Small left pneumothorax. Mild cardiomegaly.", "Mild cardiomegaly."),
]


@pytest.mark.parametrize(("findings", "impression"), FAITHFUL)
def test_faithful_summaries_are_routine(auditor, findings, impression):
    result = auditor.audit(findings, impression)
    assert result.review_priority == "ROUTINE", result.priority_reasons


@pytest.mark.parametrize(("findings", "impression"), ERRORS)
def test_clinically_significant_errors_are_high(auditor, findings, impression):
    result = auditor.audit(findings, impression)
    assert result.review_priority == "HIGH"
    assert result.priority_reasons


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("showcase-supported-012", "ROUTINE"),
        ("showcase-negation-012", "HIGH"),
        ("showcase-laterality-025", "HIGH"),
        ("showcase-severity-025", "HIGH"),
        ("showcase-temporal-327", "HIGH"),
        ("showcase-omission-025", "REVIEW"),
        ("showcase-unsupported-014", "HIGH"),
        # Drops only negative findings, which a concise impression may omit.
        ("showcase-llm-omission-417", "ROUTINE"),
        ("showcase-llm-preservation-0c33", "ROUTINE"),
    ],
)
def test_showcase_cases_route_as_described(auditor, case_id, expected):
    case = SHOWCASE_CASES_BY_ID[case_id]
    assert auditor.audit(case.findings, case.impression).review_priority == expected


def test_error_burden_ranks_errors_above_faithful_summaries(auditor):
    faithful = [auditor.audit(f, i).count_aware_risk for f, i in FAITHFUL]
    errors = [auditor.audit(f, i).count_aware_risk for f, i in ERRORS]
    assert min(errors) > 0.4
    assert sum(errors) / len(errors) > sum(faithful) / len(faithful)
