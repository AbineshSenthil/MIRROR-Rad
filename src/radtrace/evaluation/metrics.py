"""Binary ranking, calibration, review-budget, and clustered-bootstrap metrics."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence


def _as_binary(values: Sequence[int | bool]) -> list[int]:
    return [int(bool(value)) for value in values]


def expected_calibration_error(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    targets = _as_binary(labels)
    if len(targets) != len(scores):
        raise ValueError("labels and scores must have equal length")
    if not targets:
        return 0.0
    output = 0.0
    for index in range(max(1, bins)):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            row
            for row, score in enumerate(scores)
            if lower <= score < upper or (index == bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        confidence = sum(float(scores[row]) for row in members) / len(members)
        prevalence = sum(targets[row] for row in members) / len(members)
        output += len(members) / len(targets) * abs(confidence - prevalence)
    return output


def binary_metrics(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
    decisions: Sequence[int | bool] | None = None,
) -> dict[str, float | int | None]:
    """Compute error-detection metrics with explicit score and decision paths."""

    from sklearn.metrics import (
        average_precision_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    targets = _as_binary(labels)
    bounded = [max(0.0, min(1.0, float(value))) for value in scores]
    if len(targets) != len(bounded):
        raise ValueError("labels and scores must have equal length")
    predicted = (
        _as_binary(decisions)
        if decisions is not None
        else [int(score >= threshold) for score in bounded]
    )
    if len(predicted) != len(targets):
        raise ValueError("decisions must have the same length as labels")
    tp = sum(actual == 1 and guess == 1 for actual, guess in zip(targets, predicted))
    tn = sum(actual == 0 and guess == 0 for actual, guess in zip(targets, predicted))
    fp = sum(actual == 0 and guess == 1 for actual, guess in zip(targets, predicted))
    fn = sum(actual == 1 and guess == 0 for actual, guess in zip(targets, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta2 = 4.0
    f2 = (
        (1 + beta2) * precision * recall / (beta2 * precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (tp + tn) / len(targets) if targets else 0.0
    brier = (
        sum((score - actual) ** 2 for score, actual in zip(bounded, targets)) / len(targets)
        if targets
        else 0.0
    )
    two_classes = len(set(targets)) == 2
    return {
        "n": len(targets),
        "positives": sum(targets),
        "negatives": len(targets) - sum(targets),
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
        "mcc": float(matthews_corrcoef(targets, predicted)) if targets else 0.0,
        "auroc": float(roc_auc_score(targets, bounded)) if two_classes else None,
        "auprc": float(average_precision_score(targets, bounded)) if two_classes else None,
        "brier": brier,
        "ece": expected_calibration_error(targets, bounded),
        "false_alerts_per_100_negatives": 100.0 * fp / (tn + fp) if tn + fp else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def review_budget_metrics(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    *,
    error_counts: Sequence[float] | None = None,
    budgets: Sequence[float] = (0.10, 0.20, 0.30),
) -> dict[str, dict[str, float | int]]:
    targets = _as_binary(labels)
    if len(targets) != len(scores):
        raise ValueError("labels and scores must have equal length")
    counts = [float(value) for value in error_counts] if error_counts is not None else targets
    if len(counts) != len(targets):
        raise ValueError("error_counts must have the same length as labels")
    ranking = sorted(range(len(targets)), key=lambda row: float(scores[row]), reverse=True)
    total_positive = sum(targets)
    total_errors = sum(counts)
    output: dict[str, dict[str, float | int]] = {}
    for budget in budgets:
        reviewed = min(len(ranking), max(1, math.ceil(len(ranking) * float(budget))))
        selected = ranking[:reviewed]
        captured_reports = sum(targets[index] for index in selected)
        captured_errors = sum(counts[index] for index in selected)
        output[f"{round(budget * 100)}pct"] = {
            "reviewed": reviewed,
            "report_recall": captured_reports / total_positive if total_positive else 0.0,
            "error_count_recall": captured_errors / total_errors if total_errors else 0.0,
            "precision": captured_reports / reviewed if reviewed else 0.0,
        }
    return output


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return [max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator)]


def clustered_bootstrap_intervals(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    groups: Sequence[str | int],
    metrics: Mapping[str, Callable[[Sequence[int], Sequence[float]], float | None]],
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float | int | None]]:
    """Percentile confidence intervals after sampling whole groups."""

    if not (len(labels) == len(scores) == len(groups)):
        raise ValueError("labels, scores, and groups must have equal length")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[str(group)].append(index)
    keys = sorted(grouped)
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {name: [] for name in metrics}
    targets = _as_binary(labels)
    bounded = [float(value) for value in scores]
    for _ in range(max(1, iterations)):
        sampled_keys = [rng.choice(keys) for _ in keys]
        indices = [index for key in sampled_keys for index in grouped[key]]
        sample_labels = [targets[index] for index in indices]
        sample_scores = [bounded[index] for index in indices]
        for name, function in metrics.items():
            try:
                value = function(sample_labels, sample_scores)
            except (ValueError, ZeroDivisionError):
                value = None
            if value is not None and math.isfinite(float(value)):
                distributions[name].append(float(value))

    output: dict[str, dict[str, float | int | None]] = {}
    for name, function in metrics.items():
        point = function(targets, bounded)
        values = sorted(distributions[name])
        if not values:
            output[name] = {"estimate": point, "lower": None, "upper": None, "samples": 0}
            continue
        lower_index = max(0, min(len(values) - 1, round(0.025 * (len(values) - 1))))
        upper_index = max(0, min(len(values) - 1, round(0.975 * (len(values) - 1))))
        output[name] = {
            "estimate": float(point) if point is not None else None,
            "lower": values[lower_index],
            "upper": values[upper_index],
            "samples": len(values),
        }
    return output


def paired_bootstrap_differences(
    labels: Sequence[int | bool],
    candidate_scores: Sequence[float],
    comparator_scores: Sequence[float],
    metrics: Mapping[str, Callable[[Sequence[int], Sequence[float]], float | None]],
    *,
    iterations: int = 5000,
    seed: int = 42,
) -> dict[str, dict[str, float | int | None]]:
    """Bootstrap paired metric differences on the same evaluation cases.

    A positive estimate means the candidate score is higher than the
    comparator.  Resampling paired rows—rather than bootstrapping each model
    separately—preserves their case-level correlation and gives the honest
    uncertainty needed for small external cohorts such as RadEvalX.
    """

    if not (
        len(labels) == len(candidate_scores) == len(comparator_scores)
    ):
        raise ValueError("labels and both score vectors must have equal length")
    if not labels:
        raise ValueError("paired bootstrap requires at least one row")
    targets = _as_binary(labels)
    left = [float(value) for value in candidate_scores]
    right = [float(value) for value in comparator_scores]
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {name: [] for name in metrics}
    size = len(targets)
    for _ in range(max(1, iterations)):
        indices = [rng.randrange(size) for _ in range(size)]
        sample_labels = [targets[index] for index in indices]
        sample_left = [left[index] for index in indices]
        sample_right = [right[index] for index in indices]
        for name, function in metrics.items():
            try:
                left_value = function(sample_labels, sample_left)
                right_value = function(sample_labels, sample_right)
            except (ValueError, ZeroDivisionError):
                continue
            if left_value is None or right_value is None:
                continue
            difference = float(left_value) - float(right_value)
            if math.isfinite(difference):
                distributions[name].append(difference)

    output: dict[str, dict[str, float | int | None]] = {}
    for name, function in metrics.items():
        candidate_value = function(targets, left)
        comparator_value = function(targets, right)
        estimate = (
            float(candidate_value) - float(comparator_value)
            if candidate_value is not None and comparator_value is not None
            else None
        )
        values = sorted(distributions[name])
        if not values:
            output[name] = {
                "estimate": estimate,
                "lower": None,
                "upper": None,
                "samples": 0,
            }
            continue
        lower_index = max(0, min(len(values) - 1, round(0.025 * (len(values) - 1))))
        upper_index = max(0, min(len(values) - 1, round(0.975 * (len(values) - 1))))
        output[name] = {
            "estimate": estimate,
            "lower": values[lower_index],
            "upper": values[upper_index],
            "samples": len(values),
        }
    return output
