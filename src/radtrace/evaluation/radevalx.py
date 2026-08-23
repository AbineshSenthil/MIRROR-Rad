"""RadEvalX consensus-count adapter and established metric baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

ERROR_CATEGORIES = {
    "1": "false_finding",
    "2": "omitted_finding",
    "3": "incorrect_location_or_position",
    "4": "incorrect_severity",
    "5": "spurious_comparison",
    "6": "omitted_comparison_or_change",
    "7": "spurious_uncertainty",
    "8": "omitted_uncertainty",
}
COMMISSION_COLUMNS = ("1", "3", "4", "5", "7")
OMISSION_COLUMNS = ("2", "6", "8")
SIMILARITY_BASELINES = ("bleu4", "bleu_2", "bertscore", "CheXbert", "radgraph_f1")
ERROR_BASELINES = ("radcliq",)


def _consensus_frame(path: Path, prefix: str) -> Any:
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"report_id", "ground_truth", "M2Tr-Generation", *ERROR_CATEGORIES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"RadEvalX file is missing columns: {sorted(missing)}")
    frame = frame.copy()
    for column, category in ERROR_CATEGORIES.items():
        frame[column] = frame[column].fillna(0).astype(float)
        frame[f"{prefix}_{category}"] = frame[column]
    frame[f"{prefix}_error_count"] = frame[list(ERROR_CATEGORIES)].sum(axis=1)
    frame[f"{prefix}_commission_count"] = frame[list(COMMISSION_COLUMNS)].sum(axis=1)
    frame[f"{prefix}_omission_count"] = frame[list(OMISSION_COLUMNS)].sum(axis=1)
    return frame.drop(columns=list(ERROR_CATEGORIES))


def load_benchmark(directory: str | Path) -> Any:
    """Load the exact 100-report radiologist-consensus benchmark."""

    import pandas as pd

    root = Path(directory)
    significant = _consensus_frame(root / "RadEval_clinically_significant_errors.csv", "significant")
    insignificant = _consensus_frame(
        root / "RadEval-clinically_insignificant_errors.csv",
        "insignificant",
    )
    if len(significant) != 100 or significant["report_id"].nunique() != 100:
        raise ValueError("RadEvalX significant-error file must contain 100 unique reports.")
    join_columns = ["report_id", "ground_truth", "M2Tr-Generation"]
    merged = significant.merge(
        insignificant,
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )
    metrics = pd.read_csv(root / "metrcis_scores_m2tr.csv")
    metric_columns = {
        "report_id",
        "bleu4",
        "bleu_2",
        "bertscore",
        "CheXbert",
        "radgraph_f1",
        "radcliq",
    }
    missing = metric_columns - set(metrics.columns)
    if missing:
        raise ValueError(f"RadEvalX metric file is missing columns: {sorted(missing)}")
    merged = merged.merge(
        metrics[list(metric_columns)],
        on="report_id",
        how="left",
        validate="one_to_one",
    )
    if merged[list(metric_columns - {"report_id"})].isna().any().any():
        raise ValueError("One or more RadEvalX reports are missing established metric scores.")
    merged["significant_error_present"] = (merged["significant_error_count"] > 0).astype(int)
    merged["insignificant_error_present"] = (merged["insignificant_error_count"] > 0).astype(int)
    return merged.sort_values("report_id").reset_index(drop=True)


def baseline_risk_scores(frame: Any) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for name in SIMILARITY_BASELINES:
        output[name] = [1.0 - float(value) for value in frame[name]]
    for name in ERROR_BASELINES:
        values = [float(value) for value in frame[name]]
        lower, upper = min(values), max(values)
        denominator = upper - lower
        output[name] = [
            (value - lower) / denominator if denominator > 0 else 0.5 for value in values
        ]
    return output


def benchmark_summary(frame: Any) -> dict[str, Any]:
    return {
        "reports": len(frame),
        "significant_positive_reports": int(frame["significant_error_present"].sum()),
        "insignificant_positive_reports": int(frame["insignificant_error_present"].sum()),
        "significant_error_count": int(frame["significant_error_count"].sum()),
        "insignificant_error_count": int(frame["insignificant_error_count"].sum()),
        "significant_commission_count": int(frame["significant_commission_count"].sum()),
        "significant_omission_count": int(frame["significant_omission_count"].sum()),
        "significant_categories": {
            name: int(frame[f"significant_{name}"].sum()) for name in ERROR_CATEGORIES.values()
        },
    }
