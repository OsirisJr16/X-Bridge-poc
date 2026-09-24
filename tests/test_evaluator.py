import json
from pathlib import Path

import pytest

from x_bridge.domain.enums import MappingStatus, MatchingMethod
from x_bridge.domain.models import FieldMapping, MappingResult
from x_bridge.evaluation.evaluator import (
    GroundTruthMapping,
    MappingEvaluator,
    load_ground_truth,
)


def _mapping(
    source_field: str,
    target_field: str,
    confidence_score: float,
    status: MappingStatus,
) -> FieldMapping:
    return FieldMapping(
        source_field=source_field,
        target_field=target_field,
        similarity_score=confidence_score,
        confidence_score=confidence_score,
        matching_method=MatchingMethod.HYBRID,
        status=status,
        is_accepted=status is MappingStatus.AUTOMATIC,
        type_compatibility=1.0,
        score_margin=0.2,
    )


@pytest.fixture
def mapping_result_fixture() -> MappingResult:
    return MappingResult(
        matching_method=MatchingMethod.HYBRID,
        source_schema_name="source",
        target_schema_name="target",
        mappings=(
            _mapping("nom_client", "customerName", 0.94, MappingStatus.AUTOMATIC),
            _mapping("date_naissance", "createdAt", 0.92, MappingStatus.AUTOMATIC),
            _mapping("devise", "currency", 0.80, MappingStatus.REVIEW_REQUIRED),
        ),
        unmatched_source_fields=("code_agence", "solde_compte"),
        unmatched_target_fields=("loyaltyPoints",),
    )


@pytest.fixture
def ground_truth_fixture() -> tuple[GroundTruthMapping, ...]:
    return (
        GroundTruthMapping(source_field="nom_client", target_field="customerName"),
        GroundTruthMapping(source_field="date_naissance", target_field="birthDate"),
        GroundTruthMapping(source_field="devise", target_field="currency"),
        GroundTruthMapping(source_field="solde_compte", target_field="accountBalance"),
    )


def test_metrics_on_accepted_mappings_only(mapping_result_fixture, ground_truth_fixture) -> None:
    evaluation_result = MappingEvaluator().evaluate(mapping_result_fixture, ground_truth_fixture)

    assert evaluation_result.true_positives == 1
    assert evaluation_result.false_positives == 1
    assert evaluation_result.false_negatives == 3
    assert evaluation_result.true_negatives == 1
    assert evaluation_result.precision == pytest.approx(0.5)
    assert evaluation_result.recall == pytest.approx(0.25)
    assert evaluation_result.f1_score == pytest.approx(1 / 3)
    assert evaluation_result.accuracy == pytest.approx(2 / 5)


def test_including_review_required_changes_metrics(
    mapping_result_fixture, ground_truth_fixture
) -> None:
    evaluation_result = MappingEvaluator(include_review_required=True).evaluate(
        mapping_result_fixture, ground_truth_fixture
    )
    assert evaluation_result.true_positives == 2
    assert evaluation_result.false_negatives == 2
    assert evaluation_result.recall == pytest.approx(0.5)


def test_coverage_rates_sum_to_one(mapping_result_fixture, ground_truth_fixture) -> None:
    coverage = MappingEvaluator().evaluate(mapping_result_fixture, ground_truth_fixture).coverage
    coverage_total = (
        coverage.automatic_mapping_rate
        + coverage.review_required_rate
        + coverage.unmatched_rate
    )
    assert coverage_total == pytest.approx(1.0)
    assert coverage.automatic_mapping_rate == pytest.approx(2 / 5)
    assert coverage.unmatched_rate == pytest.approx(2 / 5)


def test_incorrect_and_missed_mappings_are_reported(
    mapping_result_fixture, ground_truth_fixture
) -> None:
    evaluation_result = MappingEvaluator().evaluate(mapping_result_fixture, ground_truth_fixture)
    assert evaluation_result.incorrect_mappings == (("date_naissance", "createdAt"),)
    assert ("date_naissance", "birthDate") in evaluation_result.missed_mappings
    assert ("devise", "currency") in evaluation_result.missed_mappings


def test_perfect_prediction_yields_unit_metrics() -> None:
    mapping_result = MappingResult(
        matching_method=MatchingMethod.HYBRID,
        source_schema_name="source",
        target_schema_name="target",
        mappings=(_mapping("devise", "currency", 0.95, MappingStatus.AUTOMATIC),),
        unmatched_source_fields=(),
        unmatched_target_fields=(),
    )
    ground_truth_mappings = (GroundTruthMapping(source_field="devise", target_field="currency"),)
    evaluation_result = MappingEvaluator().evaluate(mapping_result, ground_truth_mappings)

    assert evaluation_result.precision == pytest.approx(1.0)
    assert evaluation_result.recall == pytest.approx(1.0)
    assert evaluation_result.f1_score == pytest.approx(1.0)
    assert evaluation_result.accuracy == pytest.approx(1.0)


def test_empty_prediction_yields_zero_metrics_without_error() -> None:
    mapping_result = MappingResult(
        matching_method=MatchingMethod.LEXICAL,
        source_schema_name="source",
        target_schema_name="target",
        mappings=(),
        unmatched_source_fields=("devise",),
        unmatched_target_fields=("currency",),
    )
    ground_truth_mappings = (GroundTruthMapping(source_field="devise", target_field="currency"),)
    evaluation_result = MappingEvaluator().evaluate(mapping_result, ground_truth_mappings)

    assert evaluation_result.precision == pytest.approx(0.0)
    assert evaluation_result.recall == pytest.approx(0.0)
    assert evaluation_result.f1_score == pytest.approx(0.0)


def test_load_ground_truth_parses_project_file() -> None:
    ground_truth_path = Path(__file__).resolve().parents[1] / "data" / "ground_truth.json"
    ground_truth_mappings = load_ground_truth(ground_truth_path)
    assert len(ground_truth_mappings) == 17
    assert GroundTruthMapping(source_field="nom_client", target_field="customerName") in (
        ground_truth_mappings
    )


def test_load_ground_truth_rejects_malformed_entries(tmp_path: Path) -> None:
    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(json.dumps([{"source_field": "a"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_ground_truth(ground_truth_path)
