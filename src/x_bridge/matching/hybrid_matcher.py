from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from x_bridge.config import HybridWeights
from x_bridge.domain.enums import MatchingMethod
from x_bridge.domain.models import SchemaField
from x_bridge.matching.lexical_matcher import LexicalMatcher
from x_bridge.matching.semantic_matcher import SemanticMatcher
from x_bridge.matching.type_compatibility import build_type_compatibility_matrix


@dataclass(frozen=True)
class SimilarityBreakdown:
    similarity: NDArray[np.float64]
    type_compatibility: NDArray[np.float64]
    lexical_similarity: NDArray[np.float64] | None = None
    semantic_similarity: NDArray[np.float64] | None = None


class HybridMatcher:
    method = MatchingMethod.HYBRID

    def __init__(
        self,
        lexical_matcher: LexicalMatcher,
        semantic_matcher: SemanticMatcher,
        weights: HybridWeights | None = None,
    ) -> None:
        self._lexical_matcher = lexical_matcher
        self._semantic_matcher = semantic_matcher
        self._weights = weights or HybridWeights()

    def similarity_breakdown(
        self, source_fields: tuple[SchemaField, ...], target_fields: tuple[SchemaField, ...]
    ) -> SimilarityBreakdown:
        lexical_scores = self._lexical_matcher.similarity_matrix(source_fields, target_fields)
        semantic_scores = self._semantic_matcher.similarity_matrix(source_fields, target_fields)
        type_scores = build_type_compatibility_matrix(source_fields, target_fields)

        hybrid_scores = (
            self._weights.lexical_weight * lexical_scores
            + self._weights.semantic_weight * semantic_scores
            + self._weights.type_weight * type_scores
        )
        return SimilarityBreakdown(
            similarity=np.clip(hybrid_scores, 0.0, 1.0),
            type_compatibility=type_scores,
            lexical_similarity=lexical_scores,
            semantic_similarity=semantic_scores,
        )

    def similarity_matrix(
        self, source_fields: tuple[SchemaField, ...], target_fields: tuple[SchemaField, ...]
    ) -> NDArray[np.float64]:
        return self.similarity_breakdown(source_fields, target_fields).similarity
