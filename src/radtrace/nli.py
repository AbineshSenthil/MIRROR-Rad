"""NLI backends and the optional learned fusion classifier."""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .clinical_rules import assess_clinical_consistency
from .transformer_utils import load_temperature
from .types import THREE_WAY_LABELS, Label


@dataclass(slots=True)
class NLIPrediction:
    label: Label
    confidence: float
    probabilities: dict[str, float]
    backend: str


class NLIPredictor(Protocol):
    name: str

    def predict(self, premise: str, hypothesis: str) -> NLIPrediction: ...


def _softmax(scores: Mapping[Label, float]) -> dict[str, float]:
    maximum = max(scores.values()) if scores else 0.0
    values = {label: math.exp(max(-50.0, min(50.0, score - maximum))) for label, score in scores.items()}
    denominator = sum(values.values()) or 1.0
    return {label.value: value / denominator for label, value in values.items()}


class HeuristicNLIPredictor:
    """Deterministic predictor used only for explicit ablation baselines."""

    name = "clinical-rules-nli"

    def predict(self, premise: str, hypothesis: str) -> NLIPrediction:
        assessment = assess_clinical_consistency(hypothesis, premise)
        feature = assessment.features
        overlap = feature.get("entity_overlap", 0.0)
        lexical = feature.get("lexical_jaccard", 0.0)
        unsupported = feature.get("unsupported_entity_ratio", 0.0)
        hard_mismatch = max(
            feature.get("negation_mismatch", 0.0),
            feature.get("laterality_mismatch", 0.0),
            feature.get("anatomy_mismatch", 0.0),
            feature.get("measurement_mismatch", 0.0),
            feature.get("temporal_mismatch", 0.0),
            feature.get("severity_mismatch", 0.0),
            feature.get("entity_substitution_mismatch", 0.0),
        )
        # Logit-like scores.  They intentionally leave ambiguous paraphrases
        # uncertain so the verifier can abstain into REVIEW.
        scores = {
            Label.ENTAILED: -0.30 + 2.45 * overlap + 2.10 * lexical - 2.7 * hard_mismatch - 1.8 * unsupported,
            Label.CONTRADICTED: -1.10 + 4.25 * hard_mismatch + 0.70 * overlap - 0.8 * unsupported,
            Label.UNSUPPORTED: -0.45 + 3.10 * unsupported + 1.45 * (1.0 - overlap) - 0.65 * lexical,
        }
        if not (hypothesis or "").strip() or not (premise or "").strip():
            scores = {Label.ENTAILED: -4.0, Label.CONTRADICTED: -4.0, Label.UNSUPPORTED: 4.0}
        probabilities = _softmax(scores)
        label = max(THREE_WAY_LABELS, key=lambda item: probabilities[item.value])
        return NLIPrediction(label, probabilities[label.value], probabilities, self.name)

    def predict_many(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = 16
    ) -> list[NLIPrediction]:
        del batch_size
        return [self.predict(premise, hypothesis) for premise, hypothesis in pairs]


class HuggingFaceNLIPredictor:
    """Optional transformer sequence-classification backend.

    Loading is lazy and defaults to ``local_files_only=True``.  Therefore the
    application never unexpectedly downloads a large model during inference.
    ``available`` and ``load_error`` expose initialization state so operational
    callers can fail closed; the fallback is retained only for explicit ablations.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        local_files_only: bool = True,
        device: str | int = "auto",
        max_length: int = 384,
        temperature: float | None = None,
        label_mapping: Mapping[str | int, Label | str] | None = None,
        fallback: NLIPredictor | None = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.local_files_only = bool(local_files_only)
        self.device = device
        self.effective_device: str | int = "cpu"
        self.max_length = int(max_length)
        loaded_temperature, calibration_metadata = load_temperature(self.model_name_or_path)
        self.temperature = float(temperature if temperature is not None else loaded_temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be a finite value greater than zero")
        self.calibration_metadata = calibration_metadata
        self.label_mapping = {
            str(key): Label.coerce(value) for key, value in (label_mapping or {}).items()
        }
        self.fallback = fallback or HeuristicNLIPredictor()
        model_reference = Path(self.model_name_or_path)
        if model_reference.is_absolute():
            parts = model_reference.parts
            artifact_index = next(
                (index for index, part in enumerate(parts) if part.casefold() == "artifacts"),
                None,
            )
            public_reference = (
                Path(*parts[artifact_index:]).as_posix()
                if artifact_index is not None
                else model_reference.name
            )
        else:
            public_reference = self.model_name_or_path.replace("\\", "/")
        self.name = f"huggingface:{public_reference}"
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self.load_error: str | None = None

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self.load_error is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, local_files_only=self.local_files_only
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path, local_files_only=self.local_files_only
            )
            self._torch = torch
            if self.device == "auto":
                self.effective_device = "cuda" if torch.cuda.is_available() else "cpu"
            elif isinstance(self.device, int):
                self.effective_device = f"cuda:{self.device}" if torch.cuda.is_available() else "cpu"
            elif str(self.device).startswith("cuda") and not torch.cuda.is_available():
                self.effective_device = "cpu"
            else:
                self.effective_device = self.device
            if self.effective_device != "cpu":
                self._model.to(self.effective_device)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001 - any backend/load failure must fall back safely
            self.load_error = f"{type(exc).__name__}: {exc}"
            self._tokenizer = None
            self._model = None
            self._torch = None

    def predict(self, premise: str, hypothesis: str) -> NLIPrediction:
        return self.predict_many([(premise, hypothesis)], batch_size=1)[0]

    def predict_many(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int = 16,
    ) -> list[NLIPrediction]:
        """Predict multiple premise/hypothesis pairs with batched inference."""

        if not pairs:
            return []
        self._ensure_loaded()
        if self._model is None:
            output = []
            for premise, hypothesis in pairs:
                prediction = self.fallback.predict(premise, hypothesis)
                output.append(
                    NLIPrediction(
                        prediction.label,
                        prediction.confidence,
                        prediction.probabilities,
                        f"{prediction.backend} (HF fallback)",
                    )
                )
            return output
        output: list[NLIPrediction] = []
        for start in range(0, len(pairs), max(1, int(batch_size))):
            chunk = pairs[start : start + max(1, int(batch_size))]
            inputs = self._tokenizer(
                [item[0] for item in chunk],
                [item[1] for item in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            if self.effective_device != "cpu":
                inputs = {key: value.to(self.effective_device) for key, value in inputs.items()}
            use_amp = str(self.effective_device).startswith("cuda")
            with self._torch.inference_mode(), self._torch.autocast(
                device_type="cuda", dtype=self._torch.float16, enabled=use_amp
            ):
                logits = self._model(**inputs).logits.float() / self.temperature
                rows = self._torch.softmax(logits, dim=-1).detach().cpu().tolist()
            output.extend(self._map_probabilities(row) for row in rows)
        return output

    def _map_probabilities(self, raw_probabilities: Sequence[float]) -> NLIPrediction:
        mapped = {label.value: 0.0 for label in THREE_WAY_LABELS}
        id2label = getattr(self._model.config, "id2label", {}) or {}
        for index, probability in enumerate(raw_probabilities):
            raw_label = str(id2label.get(index, f"LABEL_{index}"))
            label = self._map_label(index, raw_label)
            mapped[label.value] += float(probability)
        total = sum(mapped.values()) or 1.0
        mapped = {key: value / total for key, value in mapped.items()}
        label = max(THREE_WAY_LABELS, key=lambda item: mapped[item.value])
        return NLIPrediction(label, mapped[label.value], mapped, self.name)

    def _map_label(self, index: int, raw_label: str) -> Label:
        for key in (str(index), raw_label, raw_label.lower(), raw_label.upper()):
            if key in self.label_mapping:
                return self.label_mapping[key]
        normalized = raw_label.lower()
        if "entail" in normalized:
            return Label.ENTAILED
        if "contra" in normalized:
            return Label.CONTRADICTED
        if "neutral" in normalized or "unknown" in normalized or "unsupported" in normalized:
            return Label.UNSUPPORTED
        # Most MNLI checkpoints use contradiction, neutral, entailment for 0/1/2.
        return (Label.CONTRADICTED, Label.UNSUPPORTED, Label.ENTAILED)[min(index, 2)]


class _HeldOutProbabilityCalibrator:
    """Serializable wrapper applying multinomial sigmoid recalibration."""

    def __init__(self, estimator: Any, calibrator: Any) -> None:
        self.estimator = estimator
        self.calibrator = calibrator
        self.classes_ = calibrator.classes_

    def predict_proba(self, features: Sequence[Mapping[str, float]]) -> Any:
        import numpy as np

        probabilities = np.asarray(self.estimator.predict_proba(list(features)), dtype=float)
        transformed = np.log(np.clip(probabilities, 1e-9, 1.0))
        return self.calibrator.predict_proba(transformed)


class SklearnFusionClassifier:
    """Learn how to fuse retrieval, NLI and rule signals.

    This class imports scikit-learn only when ``fit`` is called, so inference
    with the offline heuristic remains available in minimal installations.
    """

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = int(random_state)
        self.model: Any = None
        self.calibrated = False
        self.training_metadata: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def fit(
        self,
        features: Sequence[Mapping[str, float]],
        labels: Sequence[Label | str],
        *,
        calibrate: bool = True,
        class_weight: str | Mapping[str, float] | None = "balanced",
    ) -> SklearnFusionClassifier:
        if not features or len(features) != len(labels):
            raise ValueError("features and labels must be non-empty and have equal length")
        try:
            from sklearn.calibration import CalibratedClassifierCV  # type: ignore
            from sklearn.feature_extraction import DictVectorizer  # type: ignore
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.pipeline import Pipeline  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Training learned fusion requires scikit-learn. Install project training dependencies first."
            ) from exc

        y = [Label.coerce(value).value for value in labels]
        if Label.REVIEW.value in y:
            raise ValueError("REVIEW is an abstention output and cannot be a training label")
        unique_counts = {item: y.count(item) for item in set(y)}
        if len(unique_counts) < 2:
            raise ValueError("At least two target classes are required to train fusion")
        estimator = LogisticRegression(
            max_iter=1500,
            class_weight=class_weight,
            random_state=self.random_state,
        )
        if calibrate and min(unique_counts.values()) >= 3:
            classifier: Any = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
            self.calibrated = True
        else:
            classifier = estimator
            self.calibrated = False
        self.model = Pipeline([("vectorizer", DictVectorizer(sparse=True)), ("classifier", classifier)])
        self.model.fit(list(features), y)
        self.training_metadata = {
            "n_examples": len(y),
            "class_counts": unique_counts,
            "calibrated": self.calibrated,
            "feature_names": list(self.model.named_steps["vectorizer"].get_feature_names_out()),
        }
        return self

    def calibrate_on(
        self,
        features: Sequence[Mapping[str, float]],
        labels: Sequence[Label | str],
        *,
        method: str = "sigmoid",
        split_name: str = "held-out calibration",
    ) -> SklearnFusionClassifier:
        """Calibrate a fitted fusion model on an explicitly separate split.

        The previous implementation calibrated with random row-level folds,
        which could place claims from one study on both sides of a fold.  A
        frozen estimator plus a subject-disjoint calibration partition removes
        that shortcut and makes the split provenance explicit in metadata.
        """

        if self.model is None:
            raise RuntimeError("Fit fusion before calibration")
        if not features or len(features) != len(labels):
            raise ValueError("calibration features and labels must be non-empty and aligned")
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        y = [Label.coerce(value).value for value in labels]
        if Label.REVIEW.value in y or len(set(y)) < 2:
            raise ValueError("Calibration requires at least two non-REVIEW classes")
        if method != "sigmoid":
            raise ValueError("Held-out fusion calibration currently supports sigmoid only")
        base_probabilities = np.asarray(self.model.predict_proba(list(features)), dtype=float)
        calibration_inputs = np.log(np.clip(base_probabilities, 1e-9, 1.0))
        calibrator = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=self.random_state,
        )
        calibrator.fit(calibration_inputs, y)
        self.model = _HeldOutProbabilityCalibrator(self.model, calibrator)
        self.calibrated = True
        self.training_metadata["calibrated"] = True
        self.training_metadata["calibration"] = {
            "method": method,
            "split": split_name,
            "n_examples": len(y),
            "class_counts": dict(Counter(y)),
            "group_policy": "separate subject-disjoint partition",
        }
        return self

    def predict_proba(self, features: Sequence[Mapping[str, float]]) -> list[dict[str, float]]:
        if self.model is None:
            raise RuntimeError("Fusion classifier is not fitted")
        values = self.model.predict_proba(list(features))
        classes = [str(item) for item in self.model.classes_]
        output: list[dict[str, float]] = []
        for row in values:
            mapping = {label.value: 0.0 for label in THREE_WAY_LABELS}
            for label, probability in zip(classes, row):
                mapping[Label.coerce(label).value] = float(probability)
            total = sum(mapping.values()) or 1.0
            output.append({key: value / total for key, value in mapping.items()})
        return output

    def predict_proba_one(self, features: Mapping[str, float]) -> dict[str, float]:
        return self.predict_proba([features])[0]

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted fusion classifier")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "random_state": self.random_state,
            "calibrated": self.calibrated,
            "training_metadata": self.training_metadata,
            "model": self.model,
        }
        with destination.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> SklearnFusionClassifier:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        instance = cls(random_state=int(payload.get("random_state", 42)))
        instance.model = payload["model"]
        instance.calibrated = bool(payload.get("calibrated", False))
        instance.training_metadata = dict(payload.get("training_metadata", {}))
        return instance

    def metadata_json(self) -> str:
        return json.dumps(self.training_metadata, indent=2, sort_keys=True)


def build_nli_predictor(
    model_name_or_path: str | None = None,
    *,
    local_files_only: bool = True,
    device: str | int = "auto",
    max_length: int = 384,
    temperature: float | None = None,
) -> NLIPredictor:
    if model_name_or_path:
        return HuggingFaceNLIPredictor(
            model_name_or_path,
            local_files_only=local_files_only,
            device=device,
            max_length=max_length,
            temperature=temperature,
        )
    return HeuristicNLIPredictor()


# Convenient aliases used in early notebooks.
HFSequenceClassifierNLI = HuggingFaceNLIPredictor
FusionClassifier = SklearnFusionClassifier
