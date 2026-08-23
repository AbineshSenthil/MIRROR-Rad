"""Lightweight, offline TF-IDF evidence retrieval for radiology text."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from .text import claim_split
from .types import EvidenceMatch

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", re.IGNORECASE)
_CLINICAL_ALIASES = {
    "effusions": "effusion",
    "opacities": "opacity",
    "infiltrates": "infiltrate",
    "lungs": "lung",
    "pleurae": "pleura",
    "cardiomediastinal": "cardiomediastinum",
    "enlarged": "enlargement",
}


def split_sentences(text: str) -> list[str]:
    """Segment clinical prose without case-sensitive boundary assumptions.

    Open-i reports frequently contain newlines, semicolons and terse fragments;
    treating those as boundaries gives the retriever more precise evidence than
    a general prose sentence splitter.  The shared clinical text utility also
    protects decimal measurements and abbreviations, and correctly segments
    de-capitalized generated reports.
    """

    return claim_split(text)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer((text or "").lower()):
        token = match.group(0).strip("./-")
        if not token:
            continue
        tokens.append(_CLINICAL_ALIASES.get(token, token))
    return tokens


def _features(text: str, ngram_range: tuple[int, int]) -> list[str]:
    tokens = tokenize(text)
    values: list[str] = []
    lower, upper = ngram_range
    for n in range(max(1, lower), max(1, upper) + 1):
        values.extend("_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return values


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    numerator = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class TfidfEvidenceRetriever:
    """Rank Findings sentences for a candidate Impression claim.

    The implementation is intentionally dependency-free.  It uses smoothed IDF,
    sublinear term frequency and unigram/bigram features.  A new tiny index is
    built for each report, which is fast and avoids cross-patient leakage.
    """

    def __init__(
        self,
        top_k: int = 3,
        *,
        ngram_range: tuple[int, int] = (1, 2),
        min_score: float = 0.0,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.top_k = int(top_k)
        self.ngram_range = ngram_range
        self.min_score = float(min_score)

    def retrieve(
        self,
        query: str,
        source: str | Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[EvidenceMatch]:
        sentences = split_sentences(source) if isinstance(source, str) else [str(item).strip() for item in source]
        sentences = [sentence for sentence in sentences if sentence]
        if not sentences or not (query or "").strip():
            return []

        document_terms = [_features(sentence, self.ngram_range) for sentence in sentences]
        query_terms = _features(query, self.ngram_range)
        document_frequency: Counter[str] = Counter()
        for terms in document_terms:
            document_frequency.update(set(terms))
        count = len(sentences)
        idf = {
            term: math.log((count + 1.0) / (frequency + 1.0)) + 1.0
            for term, frequency in document_frequency.items()
        }
        unseen_idf = math.log(count + 1.0) + 1.0

        def vector(terms: Iterable[str]) -> dict[str, float]:
            frequencies = Counter(terms)
            return {
                term: (1.0 + math.log(frequency)) * idf.get(term, unseen_idf)
                for term, frequency in frequencies.items()
            }

        query_vector = vector(query_terms)
        query_tokens = set(tokenize(query))
        scored: list[tuple[float, int, str]] = []
        for index, (sentence, terms) in enumerate(zip(sentences, document_terms)):
            score = _cosine(query_vector, vector(terms))
            # Small clinically useful tie-breaker for exact measurements and
            # polarity/laterality terms; it never dominates TF-IDF similarity.
            sentence_tokens = set(tokenize(sentence))
            exact_markers = query_tokens & sentence_tokens & {
                "left", "right", "bilateral", "no", "without", "stable",
                "increased", "decreased", "cm", "mm",
            }
            score = min(1.0, score + 0.01 * len(exact_markers))
            if score >= self.min_score:
                scored.append((score, index, sentence))

        scored.sort(key=lambda item: (-item[0], item[1]))
        limit = min(int(top_k or self.top_k), len(scored))
        return [
            EvidenceMatch(sentence_index=index, text=sentence, score=float(score), rank=rank)
            for rank, (score, index, sentence) in enumerate(scored[:limit], start=1)
        ]

    def batch_retrieve(
        self,
        queries: Sequence[str],
        source: str | Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[list[EvidenceMatch]]:
        return [self.retrieve(query, source, top_k=top_k) for query in queries]


# Backwards-friendly concise alias for notebooks and external integrations.
EvidenceRetriever = TfidfEvidenceRetriever
