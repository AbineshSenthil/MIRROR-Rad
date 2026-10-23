import pytest

from radtrace.text import claim_split, normalize_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Words ending in "ms." are not the abbreviation "Ms." (regression).
        ("Two small aneurysms. No pneumothorax.", ["Two small aneurysms.", "No pneumothorax."]),
        ("Correlate with symptoms. No pneumothorax.", ["Correlate with symptoms.", "No pneumothorax."]),
        # Numbered impressions never produce "1." / "2." claims (regression).
        (
            "1. No acute process. 2. Stable cardiomegaly.",
            ["No acute process.", "Stable cardiomegaly."],
        ),
        (
            "1. Small left pneumothorax.\n2. No pleural effusion.",
            ["Small left pneumothorax.", "No pleural effusion."],
        ),
        ("Impression: 1) Small left effusion 2) No pneumothorax", ["Small left effusion", "No pneumothorax"]),
        ("IMPRESSION: No acute cardiopulmonary process.", ["No acute cardiopulmonary process."]),
        ("T12 compression deformity. 3. Stable.", ["T12 compression deformity.", "Stable."]),
    ],
)
def test_claim_split_regressions(text, expected):
    assert claim_split(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A number ending a sentence is content, not a list marker.
        (
            "Rib fractures involving ribs 5 and 6. No pneumothorax.",
            ["Rib fractures involving ribs 5 and 6.", "No pneumothorax."],
        ),
        ("Nodule measures 1.2 cm. No effusion.", ["Nodule measures 1.2 cm.", "No effusion."]),
        (
            "Dr. Smith was notified. Atelectasis vs. pneumonia.",
            ["Dr. Smith was notified.", "Atelectasis vs. pneumonia."],
        ),
        (
            "Findings, e.g. edema, are absent. No. 2 tube removed.",
            ["Findings, e.g. edema, are absent.", "No. 2 tube removed."],
        ),
        # Commas and "or" keep negation scope intact; semicolons split.
        (
            "No edema or pleural effusion; mild cardiomegaly.",
            ["No edema or pleural effusion", "mild cardiomegaly."],
        ),
        ("Pleural effusion: no.", ["Pleural effusion: no."]),
    ],
)
def test_claim_split_preserves_clinical_meaning(text, expected):
    assert claim_split(text) == expected


def test_claim_split_handles_empty_input():
    assert claim_split("") == []
    assert claim_split(None) == []
    assert claim_split("   \n  ") == []


def test_normalize_text_marks_openi_redactions_and_keeps_case():
    assert normalize_text("Seen on XXXX  study.\r\nNo   pneumothorax") == (
        "Seen on [REDACTED] study.\nNo pneumothorax"
    )
