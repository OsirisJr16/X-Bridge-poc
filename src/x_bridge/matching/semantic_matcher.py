import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

from x_bridge.config import SemanticSettings
from x_bridge.domain.enums import MatchingMethod
from x_bridge.domain.models import SchemaField
from x_bridge.infrastructure.embedding_service import TextEncoder
from x_bridge.matching.text_normalization import normalize_identifier


def build_field_representation(schema_field: SchemaField, include_description: bool) -> str:
    representation_parts = [
        f"field: {normalize_identifier(schema_field.name)}",
        f"type: {schema_field.data_type.value}",
    ]
    if include_description and schema_field.description:
        representation_parts.append(f"description: {schema_field.description}")
    return " | ".join(representation_parts)


def rescale_cosine_similarity(
    cosine_scores: NDArray[np.float64], cosine_floor: float
) -> NDArray[np.float64]:
    rescaled_scores = (cosine_scores - cosine_floor) / (1.0 - cosine_floor)
    return np.clip(rescaled_scores, 0.0, 1.0)


class SemanticMatcher:
    method = MatchingMethod.SEMANTIC

    def __init__(self, encoder: TextEncoder, settings: SemanticSettings | None = None) -> None:
        self._encoder = encoder
        self._settings = settings or SemanticSettings()

    def _encode_fields(self, schema_fields: tuple[SchemaField, ...]) -> NDArray[np.float64]:
        representations = [
            self._settings.query_prefix
            + build_field_representation(
                schema_field, self._settings.include_description_in_representation
            )
            for schema_field in schema_fields
        ]
        return self._encoder.encode_texts(representations)

    def similarity_matrix(
        self, source_fields: tuple[SchemaField, ...], target_fields: tuple[SchemaField, ...]
    ) -> NDArray[np.float64]:
        source_embeddings = self._encode_fields(source_fields)
        target_embeddings = self._encode_fields(target_fields)
        cosine_scores = cosine_similarity(source_embeddings, target_embeddings)
        return rescale_cosine_similarity(cosine_scores, self._settings.cosine_floor)
