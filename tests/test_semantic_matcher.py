import numpy as np
import pytest

from x_bridge.config import SemanticSettings
from x_bridge.domain.enums import FieldDataType
from x_bridge.domain.models import SchemaField
from x_bridge.matching.semantic_matcher import (
    SemanticMatcher,
    build_field_representation,
    rescale_cosine_similarity,
)


def test_representation_includes_normalized_name_type_and_description() -> None:
    schema_field = SchemaField(
        name="date_naissance",
        path="date_naissance",
        data_type=FieldDataType.STRING,
        description="Date de naissance du client",
    )
    representation = build_field_representation(schema_field, include_description=True)
    assert representation == (
        "field: date naissance | type: string | description: Date de naissance du client"
    )


def test_representation_can_exclude_description() -> None:
    schema_field = SchemaField(
        name="birthDate",
        path="birthDate",
        data_type=FieldDataType.STRING,
        description="Date of birth",
    )
    assert build_field_representation(schema_field, include_description=False) == (
        "field: birth date | type: string"
    )


def test_rescale_cosine_similarity_maps_floor_to_zero_and_one_to_one() -> None:
    cosine_scores = np.array([[0.60, 0.70, 0.85, 1.00]])
    rescaled = rescale_cosine_similarity(cosine_scores, cosine_floor=0.70)
    assert rescaled[0, 0] == pytest.approx(0.0)
    assert rescaled[0, 1] == pytest.approx(0.0)
    assert rescaled[0, 2] == pytest.approx(0.5)
    assert rescaled[0, 3] == pytest.approx(1.0)


def test_aligned_fields_receive_highest_semantic_similarity(
    source_schema_definition, target_schema_definition, aligned_encoder
) -> None:
    semantic_matcher = SemanticMatcher(
        aligned_encoder, SemanticSettings(cosine_floor=0.0, include_description_in_representation=False)
    )
    similarity_scores = semantic_matcher.similarity_matrix(
        source_schema_definition.fields, target_schema_definition.fields
    )
    assert similarity_scores.argmax(axis=1).tolist() == [0, 1, 2]
    assert np.diag(similarity_scores) == pytest.approx(np.ones(3))


def test_encoder_cache_avoids_re_encoding_identical_fields(
    source_schema_definition, aligned_encoder
) -> None:
    from x_bridge.infrastructure.embedding_service import SentenceTransformerEncoder

    class CountingEncoder(SentenceTransformerEncoder):
        def __init__(self) -> None:
            super().__init__(model_name="unused")
            self.encoded_batches: list[list[str]] = []

        def _load_model(self):
            encoded_batches = self.encoded_batches

            class StubModel:
                def encode(self, texts, **_: object):
                    encoded_batches.append(list(texts))
                    return np.eye(len(texts), 4, dtype=np.float64)

            return StubModel()

    counting_encoder = CountingEncoder()
    representations = ["query: field a", "query: field b", "query: field a"]
    counting_encoder.encode_texts(representations)
    counting_encoder.encode_texts(representations)

    assert counting_encoder.encoded_batches == [["query: field a", "query: field b"]]
    assert counting_encoder.cached_text_count == 2
