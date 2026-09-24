from pydantic import BaseModel, ConfigDict, Field

from x_bridge.domain.enums import FieldDataType, MappingStatus, MatchingMethod


class SchemaField(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    path: str
    data_type: FieldDataType
    description: str | None = None
    is_required: bool = False


class SchemaDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    fields: tuple[SchemaField, ...]

    def field_paths(self) -> tuple[str, ...]:
        return tuple(schema_field.path for schema_field in self.fields)


class FieldMapping(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_field: str
    target_field: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    matching_method: MatchingMethod
    status: MappingStatus
    is_accepted: bool
    lexical_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    type_compatibility: float = Field(ge=0.0, le=1.0)
    score_margin: float = Field(ge=0.0, le=1.0)


class MappingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    matching_method: MatchingMethod
    source_schema_name: str
    target_schema_name: str
    mappings: tuple[FieldMapping, ...]
    unmatched_source_fields: tuple[str, ...]
    unmatched_target_fields: tuple[str, ...]

    @property
    def automatic_mappings(self) -> tuple[FieldMapping, ...]:
        return tuple(
            mapping for mapping in self.mappings if mapping.status is MappingStatus.AUTOMATIC
        )

    @property
    def review_required_mappings(self) -> tuple[FieldMapping, ...]:
        return tuple(
            mapping for mapping in self.mappings if mapping.status is MappingStatus.REVIEW_REQUIRED
        )

    @property
    def source_field_count(self) -> int:
        return len(self.mappings) + len(self.unmatched_source_fields)


class CoverageMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    automatic_mapping_rate: float
    review_required_rate: float
    unmatched_rate: float


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    matching_method: MatchingMethod
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    coverage: CoverageMetrics
    incorrect_mappings: tuple[tuple[str, str], ...]
    missed_mappings: tuple[tuple[str, str], ...]
