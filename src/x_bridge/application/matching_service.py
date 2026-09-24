from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from x_bridge.config import MatchingConfig
from x_bridge.domain.enums import MappingStatus, MatchingMethod
from x_bridge.domain.models import FieldMapping, MappingResult, SchemaDefinition, SchemaField
from x_bridge.matching.confidence_scorer import ConfidenceScorer, compute_score_margins
from x_bridge.matching.hybrid_matcher import SimilarityBreakdown
from x_bridge.matching.type_compatibility import build_type_compatibility_matrix


@runtime_checkable
class ComponentAwareMatcher(Protocol):
    def similarity_breakdown(
        self, source_fields: tuple[SchemaField, ...], target_fields: tuple[SchemaField, ...]
    ) -> SimilarityBreakdown: ...


class FieldSimilarityMatcher(Protocol):
    method: MatchingMethod

    def similarity_matrix(
        self, source_fields: tuple[SchemaField, ...], target_fields: tuple[SchemaField, ...]
    ) -> NDArray[np.float64]: ...


def _build_similarity_breakdown(
    matcher: FieldSimilarityMatcher,
    source_fields: tuple[SchemaField, ...],
    target_fields: tuple[SchemaField, ...],
) -> SimilarityBreakdown:
    if isinstance(matcher, ComponentAwareMatcher):
        return matcher.similarity_breakdown(source_fields, target_fields)

    similarity_scores = matcher.similarity_matrix(source_fields, target_fields)
    type_scores = build_type_compatibility_matrix(source_fields, target_fields)
    component_scores = (
        {"lexical_similarity": similarity_scores}
        if matcher.method is MatchingMethod.LEXICAL
        else {"semantic_similarity": similarity_scores}
    )
    return SimilarityBreakdown(
        similarity=similarity_scores, type_compatibility=type_scores, **component_scores
    )


def _rank_candidate_pairs(
    confidence_scores: NDArray[np.float64],
    source_fields: tuple[SchemaField, ...],
    target_fields: tuple[SchemaField, ...],
    minimum_confidence: float,
) -> list[tuple[int, int]]:
    candidate_pairs = [
        (source_index, target_index)
        for source_index in range(len(source_fields))
        for target_index in range(len(target_fields))
        if confidence_scores[source_index, target_index] >= minimum_confidence
    ]
    return sorted(
        candidate_pairs,
        key=lambda pair: (
            -confidence_scores[pair[0], pair[1]],
            source_fields[pair[0]].path,
            target_fields[pair[1]].path,
        ),
    )


class MatchingService:
    def __init__(
        self,
        matcher: FieldSimilarityMatcher,
        confidence_scorer: ConfidenceScorer | None = None,
        config: MatchingConfig | None = None,
    ) -> None:
        self._matcher = matcher
        self._config = config or MatchingConfig()
        self._confidence_scorer = confidence_scorer or ConfidenceScorer(self._config.confidence)

    def _classify(self, confidence_score: float) -> MappingStatus:
        if confidence_score >= self._config.thresholds.automatic_threshold:
            return MappingStatus.AUTOMATIC
        if confidence_score >= self._config.thresholds.review_threshold:
            return MappingStatus.REVIEW_REQUIRED
        return MappingStatus.UNMATCHED

    def match(
        self, source_schema: SchemaDefinition, target_schema: SchemaDefinition
    ) -> MappingResult:
        source_fields = source_schema.fields
        target_fields = target_schema.fields
        breakdown = _build_similarity_breakdown(self._matcher, source_fields, target_fields)
        score_margins = compute_score_margins(breakdown.similarity)
        confidence_scores = self._confidence_scorer.confidence_matrix(
            breakdown.similarity, breakdown.type_compatibility, score_margins
        )

        ranked_pairs = _rank_candidate_pairs(
            confidence_scores,
            source_fields,
            target_fields,
            self._config.thresholds.review_threshold,
        )

        assigned_source_indices: set[int] = set()
        assigned_target_indices: set[int] = set()
        selected_mappings: list[FieldMapping] = []

        for source_index, target_index in ranked_pairs:
            if source_index in assigned_source_indices:
                continue
            if not self._config.allow_many_to_one and target_index in assigned_target_indices:
                continue

            confidence_score = float(confidence_scores[source_index, target_index])
            mapping_status = self._classify(confidence_score)
            selected_mappings.append(
                FieldMapping(
                    source_field=source_fields[source_index].path,
                    target_field=target_fields[target_index].path,
                    similarity_score=float(breakdown.similarity[source_index, target_index]),
                    confidence_score=confidence_score,
                    matching_method=self._matcher.method,
                    status=mapping_status,
                    is_accepted=mapping_status is MappingStatus.AUTOMATIC,
                    lexical_similarity=_component_value(
                        breakdown.lexical_similarity, source_index, target_index
                    ),
                    semantic_similarity=_component_value(
                        breakdown.semantic_similarity, source_index, target_index
                    ),
                    type_compatibility=float(
                        breakdown.type_compatibility[source_index, target_index]
                    ),
                    score_margin=float(score_margins[source_index, target_index]),
                )
            )
            assigned_source_indices.add(source_index)
            assigned_target_indices.add(target_index)

        selected_mappings.sort(key=lambda mapping: mapping.source_field)
        mapped_target_paths = {mapping.target_field for mapping in selected_mappings}

        return MappingResult(
            matching_method=self._matcher.method,
            source_schema_name=source_schema.name,
            target_schema_name=target_schema.name,
            mappings=tuple(selected_mappings),
            unmatched_source_fields=tuple(
                source_field.path
                for source_index, source_field in enumerate(source_fields)
                if source_index not in assigned_source_indices
            ),
            unmatched_target_fields=tuple(
                target_field.path
                for target_field in target_fields
                if target_field.path not in mapped_target_paths
            ),
        )


def _component_value(
    component_scores: NDArray[np.float64] | None, source_index: int, target_index: int
) -> float | None:
    if component_scores is None:
        return None
    return float(component_scores[source_index, target_index])
