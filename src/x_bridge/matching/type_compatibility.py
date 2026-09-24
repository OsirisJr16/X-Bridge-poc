import numpy as np
from numpy.typing import NDArray

from x_bridge.domain.enums import FieldDataType
from x_bridge.domain.models import SchemaField

IDENTICAL_TYPE_COMPATIBILITY = 1.0
NUMERIC_FAMILY_COMPATIBILITY = 0.90
UNKNOWN_TYPE_COMPATIBILITY = 0.50
INCOMPATIBLE_TYPE_COMPATIBILITY = 0.0

_NUMERIC_TYPES = frozenset({FieldDataType.INTEGER, FieldDataType.NUMBER})


def compute_type_compatibility(
    source_type: FieldDataType, target_type: FieldDataType
) -> float:
    if FieldDataType.UNKNOWN in (source_type, target_type):
        return UNKNOWN_TYPE_COMPATIBILITY
    if source_type is target_type:
        return IDENTICAL_TYPE_COMPATIBILITY
    if source_type in _NUMERIC_TYPES and target_type in _NUMERIC_TYPES:
        return NUMERIC_FAMILY_COMPATIBILITY
    return INCOMPATIBLE_TYPE_COMPATIBILITY


def build_type_compatibility_matrix(
    source_fields: tuple[SchemaField, ...], target_fields: tuple[SchemaField, ...]
) -> NDArray[np.float64]:
    return np.array(
        [
            [
                compute_type_compatibility(source_field.data_type, target_field.data_type)
                for target_field in target_fields
            ]
            for source_field in source_fields
        ],
        dtype=np.float64,
    ).reshape(len(source_fields), len(target_fields))
