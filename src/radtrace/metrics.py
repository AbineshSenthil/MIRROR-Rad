"""Dataset-level factuality, safety, retrieval and calibration metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .types import THREE_WAY_LABELS, Label, RiskTier


def _labels(values: Sequence[Label | str]) -> list[Label]:
    return [Label.coerce(value) for value in values]


def classification_metrics(
    y_true: Sequence[Label | str],
    y_pred: Sequence[Label | str],
) -> dict[str, Any]:
    true = _labels(y_true)
    predicted = _labels(y_pred)
    if len(true) != len(predicted):
        raise ValueError("y_true and y_pred must have equal length")
    if not true:
        return {
            "n": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "mcc": 0.0,
            "per_class": {},
            "confusion_matrix": {},
        }
    per_class: dict[str, dict[str, float | int]] = {}
    confusion: dict[str, dict[str, int]] = {
        actual.value: {guess.value: 0 for guess in Label} for actual in THREE_WAY_LABELS
    }
    for actual, guess in zip(true, predicted):
        if actual in THREE_WAY_LABELS:
            confusion[actual.value][guess.value] += 1
    f1_values: list[float] = []
    for label in THREE_WAY_LABELS:
        tp = sum(1 for actual, guess in zip(true, predicted) if actual == label and guess == label)
        fp = sum(1 for actual, guess in zip(true, predicted) if actual != label and guess == label)
        fn = sum(1 for actual, guess in zip(true, predicted) if actual == label and guess != label)
        support = sum(actual == label for actual in true)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label.value] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return {
        "n": len(true),
        "accuracy": sum(actual == guess for actual, guess in zip(true, predicted)) / len(true),
        "macro_f1": sum(f1_values) / len(f1_values),
        "mcc": _multiclass_mcc(true, predicted),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _multiclass_mcc(true: Sequence[Label], predicted: Sequence[Label]) -> float:
    categories = list(Label)
    matrix = [[0 for _ in categories] for _ in categories]
    lookup = {label: index for index, label in enumerate(categories)}
    for actual, guess in zip(true, predicted):
        matrix[lookup[actual]][lookup[guess]] += 1
    correct = sum(matrix[i][i] for i in range(len(categories)))
    total = sum(sum(row) for row in matrix)
    predicted_totals = [sum(matrix[i][j] for i in range(len(categories))) for j in range(len(categories))]
    actual_totals = [sum(matrix[i]) for i in range(len(categories))]
    numerator = correct * total - sum(p * t for p, t in zip(predicted_totals, actual_totals))
    denominator = math.sqrt(
        (total * total - sum(value * value for value in predicted_totals))
        * (total * total - sum(value * value for value in actual_totals))
    )
    return numerator / denominator if denominator else 0.0


def safety_metrics(
    y_true: Sequence[Label | str],
    y_pred: Sequence[Label | str],
    *,
    risk_tiers: Sequence[RiskTier | str] | None = None,
    probabilities: Sequence[Mapping[str, float]] | None = None,
) -> dict[str, float | int | None]:
    """Compute high-risk hallucination F2/FNR and unsafe AUCPR.

    A prediction of REVIEW is considered a successful safety alert because the
    system has abstained and sent the claim to a human rather than accepting it.
    """

    true = _labels(y_true)
    predicted = _labels(y_pred)
    if len(true) != len(predicted):
        raise ValueError("y_true and y_pred must have equal length")
    if risk_tiers is not None and len(risk_tiers) != len(true):
        raise ValueError("risk_tiers must have the same length as y_true")
    high_risk: list[bool] = []
    for index, label in enumerate(true):
        unsafe = label in (Label.CONTRADICTED, Label.UNSUPPORTED)
        if risk_tiers is not None:
            tier = RiskTier.coerce(risk_tiers[index])
            unsafe = unsafe and tier in (RiskTier.HIGH, RiskTier.CRITICAL)
        high_risk.append(unsafe)
    alerts = [label != Label.ENTAILED for label in predicted]
    tp = sum(actual and alert for actual, alert in zip(high_risk, alerts))
    fp = sum(not actual and alert for actual, alert in zip(high_risk, alerts))
    fn = sum(actual and not alert for actual, alert in zip(high_risk, alerts))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    beta2 = 4.0
    f2 = (1 + beta2) * precision * recall / (beta2 * precision + recall) if precision + recall else 0.0
    fnr = fn / (tp + fn) if tp + fn else 0.0
    auprc: float | None = None
    if probabilities is not None:
        if len(probabilities) != len(true):
            raise ValueError("probabilities must have the same length as y_true")
        scores = [
            float(item.get(Label.CONTRADICTED.value, 0.0))
            + float(item.get(Label.UNSUPPORTED.value, 0.0))
            for item in probabilities
        ]
        auprc = _average_precision(high_risk, scores)
    return {
        "high_risk_precision": precision,
        "high_risk_recall": recall,
        "high_risk_f2": f2,
        "high_risk_fnr": fnr,
        "high_risk_tp": tp,
        "high_risk_fp": fp,
        "high_risk_fn": fn,
        "unsafe_auprc": auprc,
    }


def _average_precision(targets: Sequence[bool], scores: Sequence[float]) -> float | None:
    positives = sum(targets)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, targets), key=lambda item: item[0], reverse=True)
    found = 0
    precision_sum = 0.0
    for rank, (_, target) in enumerate(ranked, start=1):
        if target:
            found += 1
            precision_sum += found / rank
    return precision_sum / positives


def calibration_metrics(
    y_true: Sequence[Label | str],
    probabilities: Sequence[Mapping[str, float]],
    *,
    n_bins: int = 10,
) -> dict[str, float | int]:
    true = _labels(y_true)
    if len(true) != len(probabilities):
        raise ValueError("y_true and probabilities must have equal length")
    if not true:
        return {"n": 0, "ece": 0.0, "brier": 0.0, "mean_confidence": 0.0}
    confidences: list[float] = []
    correct: list[bool] = []
    brier_values: list[float] = []
    for actual, row in zip(true, probabilities):
        normalized = _normalize_probability_row(row)
        predicted = max(THREE_WAY_LABELS, key=lambda item: normalized[item.value])
        confidences.append(normalized[predicted.value])
        correct.append(predicted == actual)
        brier_values.append(
            sum((normalized[label.value] - float(actual == label)) ** 2 for label in THREE_WAY_LABELS)
        )
    ece = 0.0
    for bin_index in range(max(1, n_bins)):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        indices = [
            index for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (bin_index == n_bins - 1 and confidence == 1.0)
        ]
        if not indices:
            continue
        bin_confidence = sum(confidences[index] for index in indices) / len(indices)
        bin_accuracy = sum(correct[index] for index in indices) / len(indices)
        ece += len(indices) / len(true) * abs(bin_accuracy - bin_confidence)
    return {
        "n": len(true),
        "ece": ece,
        "brier": sum(brier_values) / len(brier_values),
        "mean_confidence": sum(confidences) / len(confidences),
    }


def _normalize_probability_row(row: Mapping[str, float]) -> dict[str, float]:
    values = {
        label.value: max(0.0, float(row.get(label.value, row.get(label.value.lower(), 0.0))))
        for label in THREE_WAY_LABELS
    }
    total = sum(values.values())
    if total <= 0.0:
        return {key: 1.0 / len(values) for key in values}
    return {key: value / total for key, value in values.items()}


def retrieval_metrics(
    gold_evidence: Sequence[int | Sequence[int]],
    ranked_evidence: Sequence[Sequence[int]],
    *,
    ks: Sequence[int] = (1, 3),
) -> dict[str, float | int]:
    if len(gold_evidence) != len(ranked_evidence):
        raise ValueError("gold_evidence and ranked_evidence must have equal length")
    recalls = {int(k): [] for k in ks}
    reciprocal_ranks: list[float] = []
    evaluated = 0
    for gold, ranking in zip(gold_evidence, ranked_evidence):
        target = {int(gold)} if isinstance(gold, int) else {int(item) for item in gold}
        if not target:
            continue
        evaluated += 1
        ordered = [int(item) for item in ranking]
        for k, recall_values in recalls.items():
            recall_values.append(float(bool(target & set(ordered[:k]))))
        first = next((rank for rank, item in enumerate(ordered, start=1) if item in target), None)
        reciprocal_ranks.append(1.0 / first if first is not None else 0.0)
    result: dict[str, float | int] = {"n_evidence": evaluated}
    for k, values in recalls.items():
        result[f"evidence_recall@{k}"] = sum(values) / len(values) if values else 0.0
    result["evidence_mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return result


def summary_factuality_metrics(
    y_pred: Sequence[Label | str],
    *,
    summary_ids: Sequence[str] | None = None,
    y_true: Sequence[Label | str] | None = None,
) -> dict[str, float | int]:
    predicted = _labels(y_pred)
    if summary_ids is None:
        summary_ids = ["all"] * len(predicted)
    if len(summary_ids) != len(predicted):
        raise ValueError("summary_ids must have the same length as y_pred")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, summary_id in enumerate(summary_ids):
        groups[str(summary_id)].append(index)
    precisions: list[float] = []
    hallucination_rates: list[float] = []
    review_rates: list[float] = []
    for indices in groups.values():
        labels = [predicted[index] for index in indices]
        total = len(labels)
        precisions.append(sum(label == Label.ENTAILED for label in labels) / total)
        hallucination_rates.append(
            sum(label in (Label.CONTRADICTED, Label.UNSUPPORTED) for label in labels) / total
        )
        review_rates.append(sum(label == Label.REVIEW for label in labels) / total)
    result: dict[str, float | int] = {
        "n_summaries": len(groups),
        "mean_factual_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "mean_hallucination_rate": sum(hallucination_rates) / len(hallucination_rates) if hallucination_rates else 0.0,
        "mean_review_rate": sum(review_rates) / len(review_rates) if review_rates else 0.0,
    }
    if y_true is not None:
        true = _labels(y_true)
        if len(true) != len(predicted):
            raise ValueError("y_true must have the same length as y_pred")
        true_hallucination_rates: list[float] = []
        for indices in groups.values():
            true_hallucination_rates.append(
                sum(true[index] in (Label.CONTRADICTED, Label.UNSUPPORTED) for index in indices) / len(indices)
            )
        result["hallucination_rate_mae"] = (
            sum(abs(left - right) for left, right in zip(hallucination_rates, true_hallucination_rates))
            / len(hallucination_rates)
            if hallucination_rates else 0.0
        )
    return result


def evaluate_dataset(
    y_true: Sequence[Label | str],
    y_pred: Sequence[Label | str],
    *,
    probabilities: Sequence[Mapping[str, float]] | None = None,
    risk_tiers: Sequence[RiskTier | str] | None = None,
    gold_evidence: Sequence[int | Sequence[int]] | None = None,
    ranked_evidence: Sequence[Sequence[int]] | None = None,
    summary_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute all applicable metrics and return one JSON-ready report."""

    result: dict[str, Any] = {
        "classification": classification_metrics(y_true, y_pred),
        "safety": safety_metrics(
            y_true, y_pred, risk_tiers=risk_tiers, probabilities=probabilities
        ),
        "summary": summary_factuality_metrics(y_pred, summary_ids=summary_ids, y_true=y_true),
    }
    if probabilities is not None:
        result["calibration"] = calibration_metrics(y_true, probabilities)
    if gold_evidence is not None or ranked_evidence is not None:
        if gold_evidence is None or ranked_evidence is None:
            raise ValueError("gold_evidence and ranked_evidence must be provided together")
        result["retrieval"] = retrieval_metrics(gold_evidence, ranked_evidence)
    return result


# Common names used by evaluation notebooks.
compute_metrics = evaluate_dataset
compute_classification_metrics = classification_metrics
compute_retrieval_metrics = retrieval_metrics
compute_calibration_metrics = calibration_metrics
