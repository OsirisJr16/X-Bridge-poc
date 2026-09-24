from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


class HybridWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    lexical_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    type_weight: float = Field(default=0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> "HybridWeights":
        weights_total = self.lexical_weight + self.semantic_weight + self.type_weight
        if abs(weights_total - 1.0) > 1e-9:
            raise ValueError(f"hybrid weights must sum to 1.0, got {weights_total}")
        return self


class LexicalSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    token_overlap_weight: float = Field(default=0.60, ge=0.0, le=1.0)
    character_sequence_weight: float = Field(default=0.40, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> "LexicalSettings":
        weights_total = self.token_overlap_weight + self.character_sequence_weight
        if abs(weights_total - 1.0) > 1e-9:
            raise ValueError(f"lexical weights must sum to 1.0, got {weights_total}")
        return self


class ThresholdSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    automatic_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_threshold_ordering(self) -> "ThresholdSettings":
        if self.review_threshold > self.automatic_threshold:
            raise ValueError("review_threshold must not exceed automatic_threshold")
        return self


class SemanticSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str = DEFAULT_EMBEDDING_MODEL
    query_prefix: str = "query: "
    cosine_floor: float = Field(default=0.70, ge=0.0, lt=1.0)
    include_description_in_representation: bool = True


class ConfidenceSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    margin_saturation: float = Field(default=0.15, gt=0.0, le=1.0)
    margin_bonus_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    type_penalty_weight: float = Field(default=0.15, ge=0.0, le=1.0)


class MatchingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    weights: HybridWeights = HybridWeights()
    lexical: LexicalSettings = LexicalSettings()
    thresholds: ThresholdSettings = ThresholdSettings()
    semantic: SemanticSettings = SemanticSettings()
    confidence: ConfidenceSettings = ConfidenceSettings()
    allow_many_to_one: bool = False
