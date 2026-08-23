"""Small, dependency-light helpers shared by transformer training and inference.

The utilities deliberately keep calibration outside of the neural checkpoint so
that a saved model remains a normal Hugging Face sequence classifier.  RadTrace
stores a single post-hoc temperature beside the checkpoint and applies it before
softmax at inference time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def softmax_logits(logits: Sequence[Sequence[float]], temperature: float = 1.0) -> np.ndarray:
    """Convert logits to probabilities using a positive temperature."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a finite value greater than zero")
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("logits must be a two-dimensional array with at least two classes")
    scaled = values / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(np.clip(scaled, -700.0, 700.0))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def negative_log_likelihood(probabilities: np.ndarray, targets: Sequence[int]) -> float:
    """Return multiclass mean negative log likelihood."""

    labels = np.asarray(targets, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("probabilities and targets must contain the same number of rows")
    if len(labels) == 0:
        return 0.0
    if labels.min() < 0 or labels.max() >= probabilities.shape[1]:
        raise ValueError("target index is outside the probability columns")
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: Sequence[int],
    *,
    bins: int = 15,
) -> float:
    """Compute top-label expected calibration error."""

    labels = np.asarray(targets, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("probabilities and targets must contain the same number of rows")
    if len(labels) == 0:
        return 0.0
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(np.float64)
    boundaries = np.linspace(0.0, 1.0, max(1, int(bins)) + 1)
    result = 0.0
    for index in range(len(boundaries) - 1):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask |= confidence == lower
        if mask.any():
            result += float(mask.mean()) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return result


def fit_temperature(
    logits: Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    minimum: float = 0.05,
    maximum: float = 10.0,
    search_steps: int = 401,
) -> dict[str, float]:
    """Fit scalar temperature by deterministic log-grid NLL minimisation.

    A dense one-dimensional search is robust, reproducible and avoids adding an
    optimiser dependency.  Temperature scaling changes confidence, never the
    predicted class, making it safe to add after checkpoint selection.
    """

    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if len(values) == 0:
        return {
            "temperature": 1.0,
            "nll_before": 0.0,
            "nll_after": 0.0,
            "ece_before": 0.0,
            "ece_after": 0.0,
        }
    if minimum <= 0 or maximum <= minimum or search_steps < 2:
        raise ValueError("invalid temperature search range")
    candidates = np.geomspace(minimum, maximum, int(search_steps))
    losses = [negative_log_likelihood(softmax_logits(values, float(item)), labels) for item in candidates]
    temperature = float(candidates[int(np.argmin(losses))])
    before = softmax_logits(values)
    after = softmax_logits(values, temperature)
    return {
        "temperature": temperature,
        "nll_before": negative_log_likelihood(before, labels),
        "nll_after": negative_log_likelihood(after, labels),
        "ece_before": expected_calibration_error(before, labels),
        "ece_after": expected_calibration_error(after, labels),
    }


def load_temperature(model_path: str | Path) -> tuple[float, dict[str, Any]]:
    """Load MIRROR-Rad calibration metadata, falling back safely to temperature 1."""

    directory = Path(model_path)
    if not directory.is_dir():
        return 1.0, {}
    for filename in ("calibration.json", "training_metadata.json"):
        path = directory / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            calibration = payload.get("calibration", payload)
            temperature = float(calibration.get("temperature", 1.0))
            if math.isfinite(temperature) and temperature > 0.0:
                return temperature, dict(calibration)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return 1.0, {}
