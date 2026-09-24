from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from x_bridge.config import LexicalSettings
from x_bridge.domain.enums import MatchingMethod
from x_bridge.domain.models import SchemaField
from x_bridge.matching.text_normalization import (
    normalize_identifier,
    split_identifier_into_tokens,
)


@lru_cache(maxsize=4096)
def _character_sequence_similarity(normalized_source: str, normalized_target: str) -> float:
    if not normalized_source or not normalized_target:
        return 0.0
    return SequenceMatcher(None, normalized_source, normalized_target).ratio()


def _token_overlap_similarity(
    source_tokens: tuple[str, ...], target_tokens: tuple[str, ...]
) -> float:
    source_token_set = frozenset(source_tokens)
    target_token_set = frozenset(target_tokens)
    if not source_token_set or not target_token_set:
        return 0.0
    intersection_size = len(source_token_set & target_token_set)
    union_size = len(source_token_set | target_token_set)
    return intersection_size / union_size


class LexicalMatcher:
    method = MatchingMethod.LEXICAL

    def __init__(self, settings: LexicalSettings | None = None) -> None:
        self._settings = settings or LexicalSettings()

    def similarity(self, source_field: SchemaField, target_field: SchemaField) -> float:
        source_tokens = split_identifier_into_tokens(source_field.name)
        target_tokens = split_identifier_into_tokens(target_field.name)
        token_similarity = _token_overlap_similarity(source_tokens, target_tokens)
        sequence_similarity = _character_sequence_similarity(
            normalize_identifier(source_field.name), normalize_identifier(target_field.name)
        )
        blended_similarity = (
            self._settings.token_overlap_weight * token_similarity
            + self._settings.character_sequence_weight * sequence_similarity
        )
        return float(np.clip(blended_similarity, 0.0, 1.0))

    def similarity_matrix(
        self, source_fields: tuple[SchemaField, ...], target_fields: tuple[SchemaField, ...]
    ) -> NDArray[np.float64]:
        return np.array(
            [
                [self.similarity(source_field, target_field) for target_field in target_fields]
                for source_field in source_fields
            ],
            dtype=np.float64,
        ).reshape(len(source_fields), len(target_fields))
