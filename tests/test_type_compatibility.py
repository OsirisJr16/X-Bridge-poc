import pytest

from x_bridge.domain.enums import FieldDataType
from x_bridge.domain.models import SchemaField
from x_bridge.matching.type_compatibility import (
    build_type_compatibility_matrix,
    compute_type_compatibility,
)


@pytest.mark.parametrize(
    ("source_type", "target_type", "expected_compatibility"),
    [
        (FieldDataType.STRING, FieldDataType.STRING, 1.0),
        (FieldDataType.INTEGER, FieldDataType.INTEGER, 1.0),
        (FieldDataType.NUMBER, FieldDataType.NUMBER, 1.0),
        (FieldDataType.BOOLEAN, FieldDataType.BOOLEAN, 1.0),
        (FieldDataType.INTEGER, FieldDataType.NUMBER, 0.90),
        (FieldDataType.STRING, FieldDataType.INTEGER, 0.0),
        (FieldDataType.STRING, FieldDataType.BOOLEAN, 0.0),
        (FieldDataType.UNKNOWN, FieldDataType.STRING, 0.50),
    ],
)
def test_type_compatibility_values(
    source_type: FieldDataType, target_type: FieldDataType, expected_compatibility: float
) -> None:
    assert compute_type_compatibility(source_type, target_type) == pytest.approx(
        expected_compatibility
    )


def test_type_compatibility_is_symmetric() -> None:
    assert compute_type_compatibility(
        FieldDataType.INTEGER, FieldDataType.NUMBER
    ) == compute_type_compatibility(FieldDataType.NUMBER, FieldDataType.INTEGER)


def test_matrix_shape_and_values(source_schema_definition, target_schema_definition) -> None:
    compatibility_scores = build_type_compatibility_matrix(
        source_schema_definition.fields, target_schema_definition.fields
    )
    assert compatibility_scores.shape == (3, 3)
    assert compatibility_scores[0, 0] == pytest.approx(1.0)
    assert compatibility_scores[0, 2] == pytest.approx(0.0)


def test_matrix_handles_empty_field_tuple() -> None:
    string_field = SchemaField(name="a", path="a", data_type=FieldDataType.STRING)
    assert build_type_compatibility_matrix((string_field,), ()).shape == (1, 0)
