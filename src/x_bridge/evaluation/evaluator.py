import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from x_bridge.domain.enums import MappingStatus
from x_bridge.domain.models import CoverageMetrics, EvaluationResult, MappingResult


class GroundTruthMapping(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_field: str
    target_field: str


def load_ground_truth(ground_truth_path: Path) -> tuple[GroundTruthMapping, ...]:
    raw_mappings = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    return tuple(GroundTruthMapping.model_validate(entry) for entry in raw_mappings)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _compute_coverage(mapping_result: MappingResult) -> CoverageMetrics:
    source_field_count = mapping_result.source_field_count
    return CoverageMetrics(
        automatic_mapping_rate=_safe_ratio(
            len(mapping_result.automatic_mappings), source_field_count
        ),
        review_required_rate=_safe_ratio(
            len(mapping_result.review_required_mappings), source_field_count
        ),
        unmatched_rate=_safe_ratio(
            len(mapping_result.unmatched_source_fields), source_field_count
        ),
    )


class MappingEvaluator:
    def __init__(self, include_review_required: bool = False) -> None:
        self._include_review_required = include_review_required

    def _predicted_pairs(self, mapping_result: MappingResult) -> dict[str, str]:
        accepted_statuses = {MappingStatus.AUTOMATIC}
        if self._include_review_required:
            accepted_statuses.add(MappingStatus.REVIEW_REQUIRED)
        return {
            mapping.source_field: mapping.target_field
            for mapping in mapping_result.mappings
            if mapping.status in accepted_statuses
        }

    def evaluate(
        self,
        mapping_result: MappingResult,
        ground_truth_mappings: tuple[GroundTruthMapping, ...],
    ) -> EvaluationResult:
        expected_target_by_source = {
            ground_truth_mapping.source_field: ground_truth_mapping.target_field
            for ground_truth_mapping in ground_truth_mappings
        }
        predicted_target_by_source = self._predicted_pairs(mapping_result)

        true_positive_pairs = [
            (source_field, target_field)
            for source_field, target_field in predicted_target_by_source.items()
            if expected_target_by_source.get(source_field) == target_field
        ]
        incorrect_mappings = tuple(
            (source_field, target_field)
            for source_field, target_field in sorted(predicted_target_by_source.items())
            if expected_target_by_source.get(source_field) != target_field
        )
        missed_mappings = tuple(
            (source_field, target_field)
            for source_field, target_field in sorted(expected_target_by_source.items())
            if predicted_target_by_source.get(source_field) != target_field
        )

        true_positive_count = len(true_positive_pairs)
        false_positive_count = len(incorrect_mappings)
        false_negative_count = len(missed_mappings)
        true_negative_count = sum(
            1
            for source_field in mapping_result.unmatched_source_fields
            if source_field not in expected_target_by_source
        )

        precision = _safe_ratio(true_positive_count, true_positive_count + false_positive_count)
        recall = _safe_ratio(true_positive_count, true_positive_count + false_negative_count)
        f1_score = (
            2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        )
        accuracy = _safe_ratio(
            true_positive_count + true_negative_count, mapping_result.source_field_count
        )

        return EvaluationResult(
            matching_method=mapping_result.matching_method,
            true_positives=true_positive_count,
            false_positives=false_positive_count,
            false_negatives=false_negative_count,
            true_negatives=true_negative_count,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            coverage=_compute_coverage(mapping_result),
            incorrect_mappings=incorrect_mappings,
            missed_mappings=missed_mappings,
        )
