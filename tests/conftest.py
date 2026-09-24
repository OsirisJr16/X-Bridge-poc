from collections.abc import Sequence
from zlib import crc32

import numpy as np
import pytest
from numpy.typing import NDArray

from x_bridge.domain.enums import FieldDataType
from x_bridge.domain.models import SchemaDefinition, SchemaField

EMBEDDING_DIMENSION = 8
_RESERVED_BASIS_DIMENSIONS = 3


class DeterministicEncoder:
    def __init__(self, vector_by_text: dict[str, Sequence[float]] | None = None) -> None:
        self._vector_by_text = vector_by_text or {}
        self.encode_call_count = 0
        self.encoded_text_count = 0

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self.encode_call_count += 1
        self.encoded_text_count += len(texts)
        embeddings = np.array([self._vector_for(text) for text in texts], dtype=np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.where(norms == 0.0, 1.0, norms)

    def _vector_for(self, text: str) -> list[float]:
        for known_text, known_vector in self._vector_by_text.items():
            if known_text in text:
                padded_vector = list(known_vector)
                padded_vector.extend([0.0] * (EMBEDDING_DIMENSION - len(padded_vector)))
                return padded_vector
        return self._orthogonal_fallback_vector(text)

    @staticmethod
    def _orthogonal_fallback_vector(text: str) -> list[float]:
        digest = crc32(text.encode("utf-8"))
        fallback_vector = [0.0] * EMBEDDING_DIMENSION
        for dimension_index in range(_RESERVED_BASIS_DIMENSIONS, EMBEDDING_DIMENSION):
            fallback_vector[dimension_index] = float((digest >> dimension_index) & 1) + 0.01
        return fallback_vector


@pytest.fixture
def source_schema_definition() -> SchemaDefinition:
    return SchemaDefinition(
        name="source",
        fields=(
            SchemaField(
                name="nom_client",
                path="nom_client",
                data_type=FieldDataType.STRING,
                description="Nom de famille du client",
                is_required=True,
            ),
            SchemaField(
                name="date_naissance",
                path="date_naissance",
                data_type=FieldDataType.STRING,
                description="Date de naissance du client",
            ),
            SchemaField(
                name="montant_total",
                path="montant_total",
                data_type=FieldDataType.NUMBER,
                description="Montant total des commandes",
            ),
        ),
    )


@pytest.fixture
def target_schema_definition() -> SchemaDefinition:
    return SchemaDefinition(
        name="target",
        fields=(
            SchemaField(
                name="customerName",
                path="customerName",
                data_type=FieldDataType.STRING,
                description="Family name of the customer",
                is_required=True,
            ),
            SchemaField(
                name="birthDate",
                path="birthDate",
                data_type=FieldDataType.STRING,
                description="Customer date of birth",
            ),
            SchemaField(
                name="totalAmount",
                path="totalAmount",
                data_type=FieldDataType.NUMBER,
                description="Total amount of the orders",
            ),
        ),
    )


@pytest.fixture
def aligned_encoder() -> DeterministicEncoder:
    return DeterministicEncoder(
        {
            "nom client": (1.0, 0.0, 0.0),
            "customer name": (1.0, 0.0, 0.0),
            "date naissance": (0.0, 1.0, 0.0),
            "birth date": (0.0, 1.0, 0.0),
            "montant total": (0.0, 0.0, 1.0),
            "total amount": (0.0, 0.0, 1.0),
        }
    )
