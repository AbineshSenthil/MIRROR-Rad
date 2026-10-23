import json

import pytest

from radtrace.privacy import scan_privacy_risk
from radtrace.upload import UploadValidationError, parse_uploaded_cases


def test_csv_upload_maps_aliases_and_generates_ids():
    payload = b"source_findings,summary,model_name\nNo pneumothorax.,Normal chest.,llama\n"
    rows = parse_uploaded_cases("pairs.csv", payload)
    assert rows == [
        {"case_id": "case-001", "findings": "No pneumothorax.", "impression": "Normal chest.", "generator": "llama"}
    ]


def test_json_and_jsonl_uploads():
    record = {"case_id": "a", "findings": "Lungs clear.", "impression": "Normal."}
    assert parse_uploaded_cases("pairs.json", json.dumps([record]).encode()) == [record]
    jsonl = "\n".join(json.dumps({**record, "case_id": f"c{i}"}) for i in range(3)).encode()
    assert [row["case_id"] for row in parse_uploaded_cases("pairs.jsonl", jsonl)] == ["c0", "c1", "c2"]


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        ("pairs.txt", b"x", "Use a .csv"),
        ("pairs.csv", b"", "empty"),
        ("pairs.csv", "findings,impression\né,x\n".encode("latin-1"), "UTF-8"),
        ("pairs.csv", b"findings,impression\nLungs clear.,\n", "missing impression"),
        ("pairs.csv", b"case_id,findings,impression\na,x,y\na,x,y\n", "Duplicate case_id"),
    ],
)
def test_upload_rejects_unsafe_or_invalid_files(filename, payload, message):
    with pytest.raises(UploadValidationError, match=message):
        parse_uploaded_cases(filename, payload)


def test_privacy_preflight_flags_identifiers_without_copying_them():
    flags = scan_privacy_risk("MRN: 12345678. Contact jane.doe@example.org or 555-123-4567.")
    assert {flag.category for flag in flags} == {
        "medical_record_number",
        "email_address",
        "telephone_number",
    }
    assert all("12345678" not in flag.guidance for flag in flags)


def test_privacy_preflight_ignores_ordinary_report_text():
    text = "Nodule measures 1.2 cm, previously 1.0 cm. No pneumothorax. Tip at T4 level."
    assert scan_privacy_risk(text) == []
