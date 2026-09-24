import numpy as np
import pytest

from x_bridge.application.matching_service import MatchingService
from x_bridge.config import (
    ConfidenceSettings,
    HybridWeights,
    MatchingConfig,
    SemanticSettings,
    ThresholdSettings,
)
from x_bridge.domain.enums import FieldDataType, MappingStatus, MatchingMethod
from x_bridge.domain.models import SchemaDefinition, SchemaField
from x_bridge.matching.confidence_scorer import ConfidenceScorer, compute_score_margins
from x_bridge.matching.hybrid_matcher import HybridMatcher
from x_bridge.matching.lexical_matcher import LexicalMatcher
from x_bridge.matching.semantic_matcher import SemanticMatcher


def _build_hybrid_matcher(encoder, weights: HybridWeights | None = None) -> HybridMatcher:
    return HybridMatcher(
        LexicalMatcher(),
        SemanticMatcher(
            encoder,
            SemanticSettings(cosine_floor=0.0, include_description_in_representation=False),
        ),
        weights or HybridWeights(),
    )


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        HybridWeights(lexical_weight=0.5, semantic_weight=0.5, type_weight=0.5)


def test_hybrid_score_equals_weighted_sum_of_components(
    source_schema_definition, target_schema_definition, aligned_encoder
) -> None:
    weights = HybridWeights()
    hybrid_matcher = _build_hybrid_matcher(aligned_encoder, weights)
    breakdown = hybrid_matcher.similarity_breakdown(
        source_schema_definition.fields, target_schema_definition.fields
    )
    expected_scores = (
        weights.lexical_weight * breakdown.lexical_similarity
        + weights.semantic_weight * breakdown.semantic_similarity
        + weights.type_weight * breakdown.type_compatibility
    )
    assert breakdown.similarity == pytest.approx(expected_scores)


def test_score_margin_is_difference_to_runner_up() -> None:
    similarity_scores = np.array([[0.90, 0.60, 0.20]])
    score_margins = compute_score_margins(similarity_scores)
    assert score_margins[0, 0] == pytest.approx(0.30)
    assert score_margins[0, 1] == pytest.approx(0.0)
    assert score_margins[0, 2] == pytest.approx(0.0)


def test_confidence_rewards_margin_and_penalizes_type_mismatch() -> None:
    confidence_scorer = ConfidenceScorer(
        ConfidenceSettings(margin_saturation=0.10, margin_bonus_weight=0.10, type_penalty_weight=0.15)
    )
    similarity_scores = np.array([[0.80]])
    compatible_confidence = confidence_scorer.confidence_matrix(
        similarity_scores, np.array([[1.0]]), np.array([[0.10]])
    )
    incompatible_confidence = confidence_scorer.confidence_matrix(
        similarity_scores, np.array([[0.0]]), np.array([[0.10]])
    )
    ambiguous_confidence = confidence_scorer.confidence_matrix(
        similarity_scores, np.array([[1.0]]), np.array([[0.0]])
    )

    assert compatible_confidence[0, 0] == pytest.approx(0.88)
    assert incompatible_confidence[0, 0] == pytest.approx(0.76)
    assert ambiguous_confidence[0, 0] == pytest.approx(0.80)


def test_confidence_never_exceeds_one() -> None:
    confidence_scorer = ConfidenceScorer()
    confidence_scores = confidence_scorer.confidence_matrix(
        np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
    )
    assert confidence_scores[0, 0] == pytest.approx(1.0)


def test_thresholds_classify_automatic_review_and_unmatched(
    source_schema_definition, target_schema_definition, aligned_encoder
) -> None:
    config = MatchingConfig(
        thresholds=ThresholdSettings(automatic_threshold=0.90, review_threshold=0.70),
        semantic=SemanticSettings(cosine_floor=0.0, include_description_in_representation=False),
    )
    matching_service = MatchingService(_build_hybrid_matcher(aligned_encoder), config=config)
    mapping_result = matching_service.match(source_schema_definition, target_schema_definition)

    for mapping in mapping_result.mappings:
        if mapping.confidence_score >= 0.90:
            assert mapping.status is MappingStatus.AUTOMATIC
            assert mapping.is_accepted is True
        else:
            assert mapping.status is MappingStatus.REVIEW_REQUIRED
            assert mapping.is_accepted is False
        assert mapping.confidence_score >= 0.70


def test_low_scoring_source_fields_remain_unmatched(aligned_encoder) -> None:
    config = MatchingConfig(
        thresholds=ThresholdSettings(automatic_threshold=0.99, review_threshold=0.98),
        semantic=SemanticSettings(cosine_floor=0.0, include_description_in_representation=False),
    )
    source_schema = SchemaDefinition(
        name="source",
        fields=(SchemaField(name="code_agence", path="code_agence", data_type=FieldDataType.STRING),),
    )
    target_schema = SchemaDefinition(
        name="target",
        fields=(
            SchemaField(name="loyaltyPoints", path="loyaltyPoints", data_type=FieldDataType.INTEGER),
        ),
    )
    mapping_result = MatchingService(
        _build_hybrid_matcher(aligned_encoder), config=config
    ).match(source_schema, target_schema)

    assert mapping_result.mappings == ()
    assert mapping_result.unmatched_source_fields == ("code_agence",)
    assert mapping_result.unmatched_target_fields == ("loyaltyPoints",)


def test_duplicate_target_is_resolved_in_favour_of_higher_confidence(aligned_encoder) -> None:
    source_schema = SchemaDefinition(
        name="source",
        fields=(
            SchemaField(name="customerName", path="customerName", data_type=FieldDataType.STRING),
            SchemaField(name="customer_name", path="customer_name", data_type=FieldDataType.STRING),
        ),
    )
    target_schema = SchemaDefinition(
        name="target",
        fields=(
            SchemaField(name="customerName", path="customerName", data_type=FieldDataType.STRING),
            SchemaField(name="unrelatedFlag", path="unrelatedFlag", data_type=FieldDataType.BOOLEAN),
        ),
    )
    config = MatchingConfig(
        thresholds=ThresholdSettings(automatic_threshold=0.90, review_threshold=0.10),
        semantic=SemanticSettings(cosine_floor=0.0, include_description_in_representation=False),
    )
    mapping_result = MatchingService(
        _build_hybrid_matcher(aligned_encoder), config=config
    ).match(source_schema, target_schema)

    assigned_targets = [mapping.target_field for mapping in mapping_result.mappings]
    assert len(assigned_targets) == len(set(assigned_targets))


def test_many_to_one_allowed_when_configured(aligned_encoder) -> None:
    source_schema = SchemaDefinition(
        name="source",
        fields=(
            SchemaField(name="customerName", path="customerName", data_type=FieldDataType.STRING),
            SchemaField(name="customer_name", path="customer_name", data_type=FieldDataType.STRING),
        ),
    )
    target_schema = SchemaDefinition(
        name="target",
        fields=(
            SchemaField(name="customerName", path="customerName", data_type=FieldDataType.STRING),
        ),
    )
    config = MatchingConfig(
        thresholds=ThresholdSettings(automatic_threshold=0.90, review_threshold=0.10),
        semantic=SemanticSettings(cosine_floor=0.0, include_description_in_representation=False),
        allow_many_to_one=True,
    )
    mapping_result = MatchingService(
        _build_hybrid_matcher(aligned_encoder), config=config
    ).match(source_schema, target_schema)

    assert [mapping.target_field for mapping in mapping_result.mappings] == [
        "customerName",
        "customerName",
    ]


def test_matching_is_deterministic(
    source_schema_definition, target_schema_definition, aligned_encoder
) -> None:
    matching_service = MatchingService(_build_hybrid_matcher(aligned_encoder))
    first_result = matching_service.match(source_schema_definition, target_schema_definition)
    second_result = matching_service.match(source_schema_definition, target_schema_definition)
    assert first_result == second_result


def test_single_method_matchers_report_their_own_method(
    source_schema_definition, target_schema_definition, aligned_encoder
) -> None:
    lexical_result = MatchingService(LexicalMatcher()).match(
        source_schema_definition, target_schema_definition
    )
    semantic_result = MatchingService(
        SemanticMatcher(
            aligned_encoder,
            SemanticSettings(cosine_floor=0.0, include_description_in_representation=False),
        )
    ).match(source_schema_definition, target_schema_definition)

    assert lexical_result.matching_method is MatchingMethod.LEXICAL
    assert semantic_result.matching_method is MatchingMethod.SEMANTIC
    assert all(mapping.semantic_similarity is None for mapping in lexical_result.mappings)
    assert all(mapping.lexical_similarity is None for mapping in semantic_result.mappings)
