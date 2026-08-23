"""Deterministic ReXErr population construction and label provenance."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

SEMANTIC_ERROR_TYPES = frozenset(
    {
        "False prediction",
        "False negation",
        "Add medical device",
        "Change location",
        "Change severity",
        "Change position of device",
        "Change name of device",
        "Change measurement",
    }
)
LINGUISTIC_ERROR_TYPES = frozenset(
    {
        "Add typo",
        "Change to homophone",
        "Add repetition",
    }
)
REPORT_LINGUISTIC_ERROR_TYPES = frozenset(
    {
        "Add contradiction",
        "Add repetitions",
        "Add typo",
        "Change to homophone",
    }
)


def load_sentence_split(path: str | Path) -> Any:
    """Load and validate one official sentence-level split."""

    import pandas as pd

    frame = pd.read_csv(Path(path))
    required = {
        "dicom_id",
        "study_id",
        "subject_id",
        "original_sentence",
        "error_sentence",
        "error_present",
        "error_type",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ReXErr split is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["error_present"] = frame["error_present"].astype("Int64")
    frame["original_sentence"] = frame["original_sentence"].fillna("").astype(str)
    frame["error_sentence"] = frame["error_sentence"].fillna("").astype(str)
    frame["error_type"] = frame["error_type"].fillna("Not applicable").astype(str)
    frame["study_id"] = frame["study_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["evaluation_role"] = "excluded"
    clean = frame["error_present"] == 0
    semantic = (frame["error_present"] == 1) & frame["error_type"].isin(
        SEMANTIC_ERROR_TYPES
    )
    linguistic = (frame["error_present"] == 1) & frame["error_type"].isin(
        LINGUISTIC_ERROR_TYPES
    )
    neutral = frame["error_present"] == 2
    frame.loc[clean, "evaluation_role"] = "clean_control"
    frame.loc[semantic, "evaluation_role"] = "semantic_error"
    frame.loc[linguistic, "evaluation_role"] = "linguistic_error"
    frame.loc[neutral, "evaluation_role"] = "neutral_excluded"
    frame["binary_target"] = semantic.astype(int)
    frame["pair_structure"] = "substitution"
    frame.loc[frame["original_sentence"].str.strip().eq(""), "pair_structure"] = "addition"
    frame.loc[frame["error_sentence"].str.strip().eq(""), "pair_structure"] = "deletion"
    return frame


def load_report_split(path: str | Path) -> Any:
    """Load one official report-level split and parse its sampled taxonomy."""

    import pandas as pd

    frame = pd.read_csv(Path(path))
    required = {
        "dicom_id",
        "study_id",
        "subject_id",
        "original_report",
        "error_report",
        "errors_sampled",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ReXErr report split is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["original_report"] = frame["original_report"].fillna("").astype(str)
    frame["error_report"] = frame["error_report"].fillna("").astype(str)
    frame["study_id"] = frame["study_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)

    def parse_errors(value: Any) -> tuple[str, ...]:
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            parsed = []
        if not isinstance(parsed, (list, tuple)):
            return ()
        return tuple(str(item).strip() for item in parsed if str(item).strip())

    frame["sampled_error_types"] = frame["errors_sampled"].map(parse_errors)
    frame["sampled_linguistic_count"] = frame["sampled_error_types"].map(
        lambda values: sum(value in REPORT_LINGUISTIC_ERROR_TYPES for value in values)
    )
    frame["sampled_clinical_count"] = frame["sampled_error_types"].map(
        lambda values: len(values)
        - sum(value in REPORT_LINGUISTIC_ERROR_TYPES for value in values)
    )
    return frame


def build_report_evaluation_population(report_frame: Any, sentence_frame: Any) -> Any:
    """Create paired error and matched-clean report cases with post-hoc labels.

    Report-level sampled labels are intentionally not treated as exact ground
    truth because the dataset documentation notes that an attempted insertion
    may fail.  The sentence-level post-hoc labels determine whether a report has
    a semantic clinical error.  Each error report is paired with the same
    original report used as an identity clean control, balancing style, length,
    and patient context without importing new clinical labels.
    """

    import pandas as pd

    semantic_counts = report_label_counts(sentence_frame)
    merged = report_frame.merge(
        semantic_counts,
        on="study_id",
        how="inner",
        validate="one_to_one",
    )
    merged[["semantic", "linguistic"]] = merged[["semantic", "linguistic"]].astype(int)
    error_cases = pd.DataFrame(
        {
            "study_id": merged["study_id"],
            "subject_id": merged["subject_id"],
            "reference_report": merged["original_report"],
            "candidate_report": merged["error_report"],
            "target": (merged["semantic"] > 0).astype(int),
            "semantic_error_count": merged["semantic"],
            "linguistic_error_count": merged["linguistic"],
            "case_role": "error_injected",
        }
    )
    clean_cases = pd.DataFrame(
        {
            "study_id": merged["study_id"],
            "subject_id": merged["subject_id"],
            "reference_report": merged["original_report"],
            "candidate_report": merged["original_report"],
            "target": 0,
            "semantic_error_count": 0,
            "linguistic_error_count": 0,
            "case_role": "matched_clean_control",
        }
    )
    return pd.concat([error_cases, clean_cases], ignore_index=True)


def report_label_counts(sentence_frame: Any) -> Any:
    """Aggregate post-hoc sentence labels to their parent study."""

    if {"semantic", "linguistic"}.issubset(sentence_frame.columns):
        return sentence_frame[["study_id", "semantic", "linguistic"]].copy()
    return (
        sentence_frame.assign(
            semantic=(sentence_frame["evaluation_role"] == "semantic_error").astype(int),
            linguistic=(sentence_frame["evaluation_role"] == "linguistic_error").astype(int),
        )
        .groupby("study_id", as_index=False)[["semantic", "linguistic"]]
        .sum()
    )


def load_sentence_label_counts(path: str | Path, *, chunksize: int = 100_000) -> Any:
    """Stream a large sentence split into compact study-level label counts."""

    import pandas as pd

    blocks: list[Any] = []
    for chunk in pd.read_csv(
        Path(path),
        usecols=["study_id", "error_present", "error_type"],
        chunksize=max(1, int(chunksize)),
    ):
        chunk["study_id"] = chunk["study_id"].astype(str)
        chunk["error_present"] = chunk["error_present"].astype("Int64")
        chunk["error_type"] = chunk["error_type"].fillna("Not applicable").astype(str)
        chunk["semantic"] = (
            (chunk["error_present"] == 1) & chunk["error_type"].isin(SEMANTIC_ERROR_TYPES)
        ).astype(int)
        chunk["linguistic"] = (
            (chunk["error_present"] == 1) & chunk["error_type"].isin(LINGUISTIC_ERROR_TYPES)
        ).astype(int)
        blocks.append(
            chunk.groupby("study_id", as_index=False)[["semantic", "linguistic"]].sum()
        )
    if not blocks:
        return pd.DataFrame(columns=["study_id", "semantic", "linguistic"])
    return (
        pd.concat(blocks, ignore_index=True)
        .groupby("study_id", as_index=False)[["semantic", "linguistic"]]
        .sum()
    )


def build_changed_report_population(report_frame: Any, sentence_labels: Any) -> Any:
    """Build report pairs without artificial identity controls."""

    import pandas as pd

    counts = report_label_counts(sentence_labels)
    merged = report_frame.merge(counts, on="study_id", how="inner", validate="one_to_one")
    merged[["semantic", "linguistic"]] = merged[["semantic", "linguistic"]].astype(int)
    changed = merged[
        merged["original_report"].str.strip() != merged["error_report"].str.strip()
    ].copy()
    return pd.DataFrame(
        {
            "study_id": changed["study_id"],
            "subject_id": changed["subject_id"],
            "reference_report": changed["original_report"],
            "candidate_report": changed["error_report"],
            "target": (changed["semantic"] > 0).astype(int),
            "semantic_error_count": changed["semantic"],
            "linguistic_error_count": changed["linguistic"],
            "sampled_error_types": changed["sampled_error_types"],
            "case_role": "changed_error_report",
        }
    ).reset_index(drop=True)


def attach_report_context(sentence_frame: Any, report_frame: Any) -> Any:
    """Attach complete original/error reports to each aligned sentence row."""

    context = report_frame[
        ["study_id", "subject_id", "original_report", "error_report"]
    ].copy().rename(columns={"subject_id": "report_subject_id"})
    if context["study_id"].duplicated().any():
        raise ValueError("ReXErr report split contains duplicate study identifiers.")
    enriched = sentence_frame.merge(
        context,
        on="study_id",
        how="left",
        validate="many_to_one",
    )
    missing = enriched["original_report"].isna() | enriched["error_report"].isna()
    if missing.any():
        raise ValueError(
            f"ReXErr sentence/report join is missing context for {int(missing.sum())} rows."
        )
    mismatch = enriched["subject_id"].astype(str) != enriched["report_subject_id"].astype(str)
    if mismatch.any():
        raise ValueError(
            f"ReXErr sentence/report join has {int(mismatch.sum())} subject mismatches."
        )
    return enriched.drop(columns=["report_subject_id"])


def primary_semantic_population(frame: Any) -> Any:
    return frame[frame["evaluation_role"].isin(["clean_control", "semantic_error"])].copy()


def population_summary(frame: Any) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "roles": {str(key): int(value) for key, value in frame["evaluation_role"].value_counts().items()},
        "pair_structures": {
            str(key): int(value) for key, value in frame["pair_structure"].value_counts().items()
        },
        "error_types": {
            str(key): int(value)
            for key, value in frame.loc[frame["error_present"] == 1, "error_type"]
            .value_counts()
            .items()
        },
        "unique_studies": int(frame["study_id"].nunique()),
        "unique_subjects": int(frame["subject_id"].nunique()),
    }
