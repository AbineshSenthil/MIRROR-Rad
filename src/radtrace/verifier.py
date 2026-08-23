"""End-to-end evidence-grounded radiology summary verifier."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .clinical_rules import (
    CRITICAL_ENTITIES,
    RuleAssessment,
    assess_clinical_consistency,
    combine_assessments,
)
from .nli import NLIPrediction, NLIPredictor, SklearnFusionClassifier, build_nli_predictor
from .retrieval import TfidfEvidenceRetriever, split_sentences
from .types import (
    THREE_WAY_LABELS,
    AuditResult,
    AuditSummary,
    ClaimAudit,
    ErrorType,
    EvidenceMatch,
    Label,
    RiskTier,
    TextSpan,
)


@dataclass(slots=True)
class _ClaimAnalysis:
    features: dict[str, float]
    evidence: list[EvidenceMatch]
    assessment: RuleAssessment
    nli_predictions: list[NLIPrediction]
    global_nli_prediction: NLIPrediction


class RadTraceVerifier:
    """Audit Impression claims against the source Findings section.

    The default configuration is fully offline and transparent.  Pass a local
    Hugging Face checkpoint to ``nli_model`` or an already-fitted fusion model
    to upgrade the same interface without changing the UI.
    """

    ARTIFACT_VERSION = 1
    FEATURE_SCHEMA_VERSION = "sentence-plus-global-nli-v3-semantic-delta"
    DEFAULT_RULE_OVERRIDE_THRESHOLDS: ClassVar[dict[str, float]] = {
        "negation_mismatch": 0.30,
        "laterality_mismatch": 0.30,
        "anatomy_mismatch": 0.55,
        "measurement_mismatch": 0.30,
        "severity_mismatch": 0.55,
        "temporal_mismatch": 0.55,
        "entity_substitution_mismatch": 0.50,
        "device_identity_mismatch": 0.50,
        "device_position_mismatch": 0.50,
    }

    def __init__(
        self,
        *,
        top_k: int = 3,
        review_threshold: float = 0.52,
        nli_model: str | None = None,
        nli_predictor: NLIPredictor | None = None,
        fusion_model: SklearnFusionClassifier | None = None,
        local_files_only: bool = True,
        nli_device: str | int = "auto",
        nli_max_length: int = 384,
        nli_temperature: float | None = None,
        apply_rule_overrides: bool = True,
        rule_override_thresholds: Mapping[str, float] | None = None,
    ) -> None:
        if not 0.0 <= review_threshold <= 1.0:
            raise ValueError("review_threshold must be between 0 and 1")
        self.top_k = int(top_k)
        self.review_threshold = float(review_threshold)
        self.nli_model = nli_model
        self.local_files_only = bool(local_files_only)
        self.nli_device = nli_device
        self.nli_max_length = int(nli_max_length)
        self.nli_temperature = nli_temperature
        self.apply_rule_overrides = bool(apply_rule_overrides)
        self.rule_override_thresholds = {
            **self.DEFAULT_RULE_OVERRIDE_THRESHOLDS,
            **{key: float(value) for key, value in (rule_override_thresholds or {}).items()},
        }
        self.retriever = TfidfEvidenceRetriever(top_k=self.top_k)
        self.nli_predictor = nli_predictor or build_nli_predictor(
            nli_model,
            local_files_only=self.local_files_only,
            device=self.nli_device,
            max_length=self.nli_max_length,
            temperature=self.nli_temperature,
        )
        self.fusion_model = fusion_model

    def audit(self, findings: str, impression: str) -> AuditResult:
        """Audit one source/candidate pair with batched claim inference."""

        return self.audit_many([(findings, impression)])[0]

    def audit_many(
        self,
        report_pairs: Sequence[tuple[str, str]],
        *,
        nli_batch_size: int = 32,
    ) -> list[AuditResult]:
        """Audit many reports while sharing the neural inference batches.

        Each input tuple is ``(source_findings, candidate_impression)``.  The
        returned list preserves input order and keeps report boundaries in the
        claim identifiers.  This is the production path used for dataset
        auditing and public-benchmark evaluation.
        """

        normalized: list[tuple[str, str, list[str]]] = []
        flattened: list[tuple[str, str]] = []
        for findings, impression in report_pairs:
            source = (findings or "").strip()
            candidate = (impression or "").strip()
            claims = split_sentences(candidate)
            normalized.append((source, candidate, claims))
            flattened.extend((source, claim) for claim in claims)

        audited_claims = self.audit_claims_batch(
            flattened,
            nli_batch_size=nli_batch_size,
        )
        output: list[AuditResult] = []
        cursor = 0
        for report_index, (findings, impression, claims_text) in enumerate(
            normalized,
            start=1,
        ):
            count = len(claims_text)
            claims = audited_claims[cursor : cursor + count]
            cursor += count
            for claim_index, claim in enumerate(claims, start=1):
                claim.claim_id = f"R{report_index:05d}-C{claim_index:03d}"
            output.append(
                AuditResult(
                    findings=findings,
                    impression=impression,
                    claims=claims,
                    summary=self._summarize(claims),
                    model_info=self._model_info(),
                )
            )
        return output

    def audit_claims_batch(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        nli_batch_size: int = 32,
    ) -> list[ClaimAudit]:
        """Audit already-segmented ``(source, claim)`` pairs in one batch."""

        return [
            claim
            for claim, _ in self.audit_claims_with_features_batch(
                pairs,
                nli_batch_size=nli_batch_size,
            )
        ]

    def audit_claims_with_features_batch(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        nli_batch_size: int = 32,
    ) -> list[tuple[ClaimAudit, dict[str, float]]]:
        """Audit segmented pairs and retain the exact fusion feature vectors."""

        if not pairs:
            return []
        normalized = [((source or "").strip(), (claim or "").strip()) for source, claim in pairs]
        analyses = self._analyse_claims(normalized, nli_batch_size=max(1, nli_batch_size))
        return [
            (
                self._claim_from_analysis(
                    claim,
                    claim_id=f"C{index:06d}",
                    analysis=analysis,
                ),
                dict(analysis.features),
            )
            for index, ((_, claim), analysis) in enumerate(
                zip(normalized, analyses, strict=True),
                start=1,
            )
        ]

    def _model_info(self) -> dict[str, object]:
        return {
            "retriever": "tfidf-unigram-bigram",
            "top_k": self.top_k,
            "nli_backend": getattr(
                self.nli_predictor,
                "name",
                type(self.nli_predictor).__name__,
            ),
            "fusion": "learned-logistic-calibrated"
            if self.fusion_model is not None and self.fusion_model.is_fitted
            else "transparent-weighted-fusion",
            "review_threshold": self.review_threshold,
            "rule_overrides": self.apply_rule_overrides,
            "rule_override_thresholds": self.rule_override_thresholds,
            "offline": self.local_files_only,
            "nli_device": getattr(self.nli_predictor, "effective_device", "cpu"),
            "nli_temperature": getattr(self.nli_predictor, "temperature", 1.0),
            "feature_schema": self.FEATURE_SCHEMA_VERSION,
        }

    def extract_claim_features(self, findings: str, claim: str) -> dict[str, float]:
        """Public training hook returning exactly the inference-time features."""

        return dict(self._analyse_claim(findings, claim).features)

    def extract_claim_features_batch(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        nli_batch_size: int = 16,
    ) -> list[dict[str, float]]:
        """Extract inference-identical features with batched neural NLI calls."""

        return [
            dict(analysis.features)
            for analysis in self._analyse_claims(pairs, nli_batch_size=nli_batch_size)
        ]

    def _analyse_claim(self, findings: str, claim: str) -> _ClaimAnalysis:
        return self._analyse_claims([(findings, claim)], nli_batch_size=max(1, self.top_k))[0]

    def _analyse_claims(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        nli_batch_size: int,
    ) -> list[_ClaimAnalysis]:
        prepared: list[
            tuple[list[EvidenceMatch], list[RuleAssessment], list[int], int]
        ] = []
        flattened_pairs: list[tuple[str, str]] = []
        for findings, claim in pairs:
            evidence = self.retriever.retrieve(claim, findings, top_k=self.top_k)
            assessments = [assess_clinical_consistency(claim, match.text) for match in evidence]
            sentence_prediction_indices: list[int] = []
            for match in evidence:
                sentence_prediction_indices.append(len(flattened_pairs))
                flattened_pairs.append((match.text, claim))
            # The encoder was trained on full Findings→claim pairs.  Retain the
            # sentence pairs for evidence explanations, then add one global
            # pair so learned fusion can use the model in its trained context.
            # For a one-sentence source the evidence and global pairs are
            # identical, so reuse the prediction rather than doubling work.
            if (
                len(evidence) == 1
                and evidence[0].text.strip() == findings.strip()
                and sentence_prediction_indices
            ):
                global_prediction_index = sentence_prediction_indices[0]
            else:
                global_prediction_index = len(flattened_pairs)
                flattened_pairs.append((findings, claim))
            prepared.append(
                (
                    evidence,
                    assessments,
                    sentence_prediction_indices,
                    global_prediction_index,
                )
            )

        predict_many = getattr(self.nli_predictor, "predict_many", None)
        if callable(predict_many):
            flattened_predictions = predict_many(
                flattened_pairs, batch_size=max(1, int(nli_batch_size))
            )
        else:
            flattened_predictions = [
                self.nli_predictor.predict(premise, hypothesis)
                for premise, hypothesis in flattened_pairs
            ]

        output: list[_ClaimAnalysis] = []
        for evidence, assessments, sentence_indices, global_index in prepared:
            predictions = [flattened_predictions[index] for index in sentence_indices]
            global_prediction = flattened_predictions[global_index]
            for match, prediction in zip(evidence, predictions):
                match.nli_probabilities = dict(prediction.probabilities)
            assessment = combine_assessments(assessments)
            features = self._aggregate_features(evidence, assessments, predictions, assessment)
            for label in THREE_WAY_LABELS:
                features[f"nli_global_{label.value.lower()}"] = float(
                    global_prediction.probabilities.get(label.value, 0.0)
                )
            output.append(
                _ClaimAnalysis(
                    features,
                    evidence,
                    assessment,
                    predictions,
                    global_prediction,
                )
            )
        return output

    def _aggregate_features(
        self,
        evidence: list[EvidenceMatch],
        assessments: list[RuleAssessment],
        predictions: list[NLIPrediction],
        best_assessment: RuleAssessment,
    ) -> dict[str, float]:
        features = {f"rule_{key}": float(value) for key, value in best_assessment.features.items()}
        features.update(
            {
                "lexical_jaccard": best_assessment.features.get("lexical_jaccard", 0.0),
                "retrieval_top_score": evidence[0].score if evidence else 0.0,
                "retrieval_raw_top_score": evidence[0].raw_score if evidence else 0.0,
                "retrieval_slot_masked_top_score": (
                    evidence[0].slot_masked_score if evidence else 0.0
                ),
                "retrieval_slot_gain": (
                    max(0.0, evidence[0].slot_masked_score - evidence[0].raw_score)
                    if evidence
                    else 0.0
                ),
                "retrieval_second_score": evidence[1].score if len(evidence) > 1 else 0.0,
                "retrieval_margin": evidence[0].score - evidence[1].score if len(evidence) > 1 else (evidence[0].score if evidence else 0.0),
                "evidence_count": float(len(evidence)),
            }
        )
        for label in THREE_WAY_LABELS:
            weighted = [
                prediction.probabilities.get(label.value, 0.0) * (0.35 + 0.65 * match.score)
                for match, prediction in zip(evidence, predictions)
            ]
            features[f"nli_{label.value.lower()}"] = max(weighted, default=0.0)
            features[f"nli_mean_{label.value.lower()}"] = sum(weighted) / len(weighted) if weighted else 0.0
        # Serious contradictions in any highly relevant sentence should remain
        # visible, even if another sentence has better overall entity overlap.
        for name in (
            "negation_mismatch", "laterality_mismatch", "anatomy_mismatch",
            "severity_mismatch", "temporal_mismatch", "measurement_mismatch",
            "entity_substitution_mismatch", "device_identity_mismatch",
            "device_position_mismatch",
        ):
            features[f"any_rule_{name}"] = max(
                (item.features.get(name, 0.0) * (0.35 + 0.65 * match.score)
                 for item, match in zip(assessments, evidence)),
                default=0.0,
            )
        return features

    def _audit_claim(self, findings: str, claim: str, *, claim_id: str) -> ClaimAudit:
        analysis = self._analyse_claim(findings, claim)
        return self._claim_from_analysis(claim, claim_id=claim_id, analysis=analysis)

    def _claim_from_analysis(
        self,
        claim: str,
        *,
        claim_id: str,
        analysis: _ClaimAnalysis,
    ) -> ClaimAudit:
        if self.fusion_model is not None and self.fusion_model.is_fitted:
            probabilities = self.fusion_model.predict_proba_one(analysis.features)
        else:
            probabilities = self._heuristic_fusion(analysis.features)
        if self.apply_rule_overrides:
            probabilities = self._apply_high_precision_rule_override(probabilities, analysis.features)
        raw_label = max(THREE_WAY_LABELS, key=lambda item: probabilities.get(item.value, 0.0))
        confidence = float(probabilities[raw_label.value])
        # Empty Findings is definitely unsupported rather than merely uncertain.
        label = raw_label if confidence >= self.review_threshold or not analysis.evidence else Label.REVIEW
        if not analysis.evidence:
            label = Label.UNSUPPORTED
            confidence = float(probabilities[Label.UNSUPPORTED.value])

        error_types = list(analysis.assessment.violations)
        spans = list(analysis.assessment.spans)
        if label == Label.ENTAILED:
            error_types = []
            spans = []
        elif label == Label.UNSUPPORTED:
            if ErrorType.UNSUPPORTED not in error_types:
                error_types.insert(0, ErrorType.UNSUPPORTED)
        elif label == Label.CONTRADICTED and not error_types:
            if analysis.features.get("any_rule_negation_mismatch", 0.0) >= self.rule_override_thresholds["negation_mismatch"]:
                error_types = [ErrorType.NEGATION]
            elif (
                analysis.features.get("any_rule_laterality_mismatch", 0.0)
                >= self.rule_override_thresholds["laterality_mismatch"]
                or analysis.features.get("any_rule_anatomy_mismatch", 0.0)
                >= self.rule_override_thresholds["anatomy_mismatch"]
            ):
                error_types = [ErrorType.LATERALITY]
            elif (
                analysis.features.get("any_rule_device_identity_mismatch", 0.0)
                >= self.rule_override_thresholds["device_identity_mismatch"]
                or analysis.features.get("any_rule_device_position_mismatch", 0.0)
                >= self.rule_override_thresholds["device_position_mismatch"]
                or analysis.features.get("any_rule_measurement_mismatch", 0.0)
                >= self.rule_override_thresholds["measurement_mismatch"]
            ):
                error_types = [ErrorType.MEASUREMENT]
            elif analysis.features.get("any_rule_severity_mismatch", 0.0) >= self.rule_override_thresholds["severity_mismatch"]:
                error_types = [ErrorType.SEVERITY]
            elif analysis.features.get("any_rule_temporal_mismatch", 0.0) >= self.rule_override_thresholds["temporal_mismatch"]:
                error_types = [ErrorType.TEMPORAL]
            else:
                error_types = [ErrorType.ENTITY]
        elif label == Label.REVIEW:
            if ErrorType.LOW_CONFIDENCE not in error_types:
                error_types.append(ErrorType.LOW_CONFIDENCE)
        if label != Label.ENTAILED and not spans:
            spans = [TextSpan(0, len(claim), claim, error_types[0] if error_types else ErrorType.ENTITY)]

        risk = self._risk_for(label, analysis.assessment)
        explanation = self._explain(label, confidence, analysis)
        return ClaimAudit(
            claim_id=claim_id,
            claim=claim,
            label=label,
            confidence=confidence,
            probabilities={key: float(value) for key, value in probabilities.items()},
            risk_tier=risk,
            error_types=error_types,
            evidence=analysis.evidence,
            spans=spans,
            rule_flags={
                **analysis.assessment.details,
                "violations": [item.value for item in analysis.assessment.violations],
                "features": analysis.assessment.features,
            },
            explanation=explanation,
        )

    @staticmethod
    def _heuristic_fusion(features: Mapping[str, float]) -> dict[str, float]:
        entail = (
            0.62 * features.get("nli_entailed", 0.0)
            + 0.18 * features.get("retrieval_top_score", 0.0)
            + 0.20 * features.get("rule_entity_overlap", 0.0)
        )
        strongest_conflict = max(
            features.get("any_rule_negation_mismatch", 0.0),
            features.get("any_rule_laterality_mismatch", 0.0),
            0.90 * features.get("any_rule_anatomy_mismatch", 0.0),
            features.get("any_rule_measurement_mismatch", 0.0),
            features.get("any_rule_device_identity_mismatch", 0.0),
            features.get("any_rule_device_position_mismatch", 0.0),
            0.85 * features.get("any_rule_entity_substitution_mismatch", 0.0),
            0.75 * features.get("any_rule_temporal_mismatch", 0.0),
            0.70 * features.get("any_rule_severity_mismatch", 0.0),
        )
        contradicted = (
            0.64 * features.get("nli_contradicted", 0.0)
            + 0.30 * strongest_conflict
            + 0.06 * features.get("rule_entity_overlap", 0.0)
        )
        unsupported = (
            0.62 * features.get("nli_unsupported", 0.0)
            + 0.28 * features.get("rule_unsupported_entity_ratio", 0.0)
            + 0.10 * (1.0 - features.get("retrieval_top_score", 0.0))
        )
        # Explicit conflicts suppress a superficially high lexical entailment.
        entail *= max(0.10, 1.0 - 0.78 * strongest_conflict)
        if features.get("evidence_count", 0.0) == 0.0:
            entail, contradicted, unsupported = 0.01, 0.01, 0.98
        raw = {
            Label.ENTAILED.value: max(0.001, entail),
            Label.CONTRADICTED.value: max(0.001, contradicted),
            Label.UNSUPPORTED.value: max(0.001, unsupported),
        }
        total = sum(raw.values())
        return {key: value / total for key, value in raw.items()}

    def _apply_high_precision_rule_override(
        self, probabilities: Mapping[str, float], features: Mapping[str, float]
    ) -> dict[str, float]:
        """Enforce safety-critical conflicts that have unambiguous structure.

        Learned fusion remains responsible for nuanced entity, severity and
        temporal cases.  Polarity, entity-local laterality and normalized
        measurement conflicts are deterministic and should not be accepted as
        entailments merely because their words are otherwise very similar.
        """

        triggered = [
            features.get(f"any_rule_{name}", 0.0)
            for name, threshold in self.rule_override_thresholds.items()
            if features.get(f"any_rule_{name}", 0.0) >= threshold
        ]
        if not triggered:
            return {key: float(value) for key, value in probabilities.items()}
        contradicted_probability = float(probabilities.get(Label.CONTRADICTED.value, 0.0))
        unsupported_probability = float(probabilities.get(Label.UNSUPPORTED.value, 0.0))
        if (
            self.fusion_model is not None
            and self.fusion_model.is_fitted
            and unsupported_probability >= 0.45
            and unsupported_probability >= contradicted_probability + 0.05
        ):
            # A structural mismatch proves that the claim is unsafe, but it does
            # not by itself distinguish a contradiction from an unsupported
            # addition. Preserve a confident learned unsupported decision so a
            # device/addition rule cannot silently change the error topology.
            # When learned fusion is unavailable, deterministic rules remain the
            # fail-safe path for explicit substitutions.
            return {key: float(value) for key, value in probabilities.items()}
        conflict = max(triggered)
        target = max(
            contradicted_probability,
            min(0.97, 0.62 + 0.35 * conflict),
        )
        other_labels = (Label.ENTAILED.value, Label.UNSUPPORTED.value)
        other_total = sum(float(probabilities.get(label, 0.0)) for label in other_labels)
        remainder = 1.0 - target
        if other_total <= 0.0:
            other_values = {label: remainder / 2.0 for label in other_labels}
        else:
            other_values = {
                label: remainder * float(probabilities.get(label, 0.0)) / other_total
                for label in other_labels
            }
        return {**other_values, Label.CONTRADICTED.value: target}

    @staticmethod
    def _risk_for(label: Label, assessment: RuleAssessment) -> RiskTier:
        if label == Label.ENTAILED:
            return RiskTier.LOW
        risk = assessment.risk_tier
        entities = set(assessment.details.get("claim_entities", []))
        if label == Label.UNSUPPORTED and entities & CRITICAL_ENTITIES:
            return RiskTier.HIGH
        if label == Label.CONTRADICTED and risk.rank < RiskTier.HIGH.rank:
            return RiskTier.HIGH
        if label == Label.REVIEW and risk.rank < RiskTier.MODERATE.rank:
            return RiskTier.MODERATE
        if label == Label.UNSUPPORTED and risk == RiskTier.LOW:
            return RiskTier.MODERATE
        return risk

    @staticmethod
    def _explain(label: Label, confidence: float, analysis: _ClaimAnalysis) -> str:
        if not analysis.evidence:
            return "No Findings sentence was available to support this claim."
        evidence = analysis.evidence[0]
        violation_names = [item.value.replace("_", " ").lower() for item in analysis.assessment.violations]
        if label == Label.ENTAILED:
            return f"The claim is supported by retrieved Findings evidence (similarity {evidence.score:.2f})."
        if label == Label.CONTRADICTED:
            identity_pairs = analysis.assessment.details.get("device_identity_mismatch_pairs", [])
            if identity_pairs:
                changed = ", ".join(f"{left} vs {right}" for left, right in identity_pairs[:2])
                return f"The claim changes the device identity relative to the evidence ({changed})."
            position_changes = analysis.assessment.details.get("device_position_mismatches", [])
            if position_changes:
                first = position_changes[0]
                claim_positions = ", ".join(first.get("claim_positions", []))
                evidence_positions = ", ".join(first.get("evidence_positions", []))
                return (
                    "The claim changes a device position relative to the evidence "
                    f"({claim_positions} vs {evidence_positions})."
                )
            reason = ", ".join(violation_names) if violation_names else "a semantic conflict"
            return f"The claim conflicts with the retrieved evidence through {reason}."
        if label == Label.UNSUPPORTED:
            entities = analysis.assessment.details.get("unsupported_entities", [])
            detail = f" Unmatched concepts: {', '.join(entities)}." if entities else ""
            return f"No sufficiently supporting evidence was found.{detail}".strip()
        return f"The best class confidence ({confidence:.2f}) is below the review threshold; human review is recommended."

    @staticmethod
    def _summarize(claims: list[ClaimAudit]) -> AuditSummary:
        counts = {label.value: 0 for label in Label}
        for claim in claims:
            counts[claim.label.value] += 1
        total = len(claims)
        factual_precision = counts[Label.ENTAILED.value] / total if total else 0.0
        hallucination_rate = (
            counts[Label.CONTRADICTED.value] + counts[Label.UNSUPPORTED.value]
        ) / total if total else 0.0
        review_rate = counts[Label.REVIEW.value] / total if total else 0.0
        highest_risk = max((claim.risk_tier for claim in claims), key=lambda item: item.rank, default=RiskTier.LOW)
        label_weight = {
            Label.ENTAILED: 0.0,
            Label.REVIEW: 0.35,
            Label.UNSUPPORTED: 0.65,
            Label.CONTRADICTED: 1.0,
        }
        tier_multiplier = {
            RiskTier.LOW: 0.5,
            RiskTier.MODERATE: 0.75,
            RiskTier.HIGH: 0.9,
            RiskTier.CRITICAL: 1.0,
        }
        risk_score = (
            100.0 * sum(label_weight[item.label] * tier_multiplier[item.risk_tier] for item in claims) / total
            if total else 0.0
        )
        return AuditSummary(
            total_claims=total,
            label_counts=counts,
            factual_precision=factual_precision,
            hallucination_rate=hallucination_rate,
            review_rate=review_rate,
            highest_risk=highest_risk,
            risk_weighted_score=risk_score,
        )

    def save_artifact(self, directory: str | Path) -> Path:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        config = {
            "artifact_version": self.ARTIFACT_VERSION,
            "feature_schema": self.FEATURE_SCHEMA_VERSION,
            "top_k": self.top_k,
            "review_threshold": self.review_threshold,
            "nli_model": str(self.nli_model).replace("\\", "/")
            if self.nli_model
            else None,
            "local_files_only": self.local_files_only,
            "nli_device": self.nli_device,
            "nli_max_length": self.nli_max_length,
            "nli_temperature": self.nli_temperature,
            "apply_rule_overrides": self.apply_rule_overrides,
            "rule_override_thresholds": self.rule_override_thresholds,
            "has_fusion_model": bool(self.fusion_model is not None and self.fusion_model.is_fitted),
        }
        (destination / "verifier_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        if config["has_fusion_model"]:
            assert self.fusion_model is not None
            self.fusion_model.save(destination / "fusion_model.pkl")
        return destination

    @classmethod
    def load_artifact(
        cls,
        directory: str | Path,
        *,
        device_override: str | int | None = None,
    ) -> RadTraceVerifier:
        source = Path(directory).resolve()
        config = json.loads((source / "verifier_config.json").read_text(encoding="utf-8"))
        if int(config.get("artifact_version", 0)) != cls.ARTIFACT_VERSION:
            raise ValueError("Unsupported verifier artifact schema")
        nli_model = config.get("nli_model")
        if nli_model:
            normalized = Path(str(nli_model).replace("\\", "/"))
            if not normalized.is_absolute():
                project_root = next(
                    (
                        parent
                        for parent in (source, *source.parents)
                        if (parent / "src" / "radtrace").is_dir()
                    ),
                    Path.cwd(),
                )
                normalized = project_root / normalized
            normalized = normalized.resolve()
            if bool(config.get("local_files_only", True)) and not normalized.is_dir():
                raise FileNotFoundError(f"Configured clinical NLI checkpoint was not found: {normalized}")
            nli_model = str(normalized)
        fusion_path = source / "fusion_model.pkl"
        fusion = SklearnFusionClassifier.load(fusion_path) if fusion_path.exists() else None
        return cls(
            top_k=int(config.get("top_k", 3)),
            review_threshold=float(config.get("review_threshold", 0.52)),
            nli_model=nli_model,
            local_files_only=bool(config.get("local_files_only", True)),
            nli_device=device_override
            if device_override is not None
            else config.get("nli_device", "auto"),
            nli_max_length=int(config.get("nli_max_length", 384)),
            nli_temperature=config.get("nli_temperature"),
            fusion_model=fusion,
            apply_rule_overrides=bool(config.get("apply_rule_overrides", True)),
            rule_override_thresholds=config.get("rule_override_thresholds"),
        )

    # Concise aliases for callers that think of the entire directory as a model.
    save = save_artifact
    load = load_artifact
