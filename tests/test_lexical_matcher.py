import pytest

from x_bridge.domain.enums import FieldDataType
from x_bridge.domain.models import SchemaField
from x_bridge.matching.lexical_matcher import LexicalMatcher
from x_bridge.matching.text_normalization import (
    normalize_identifier,
    split_identifier_into_tokens,
)


@pytest.mark.parametrize(
    ("identifier", "expected_normalized"),
    [
        ("customerName", "customer name"),
        ("nom_client", "nom client"),
        ("birthDate", "birth date"),
        ("date_naissance", "date naissance"),
        ("customer-postal-code", "customer postal code"),
        ("HTTPResponseCode", "http response code"),
        ("address.city", "address city"),
        ("field2Name", "field 2 name"),
    ],
)
def test_normalize_identifier_handles_naming_variations(
    identifier: str, expected_normalized: str
) -> None:
    assert normalize_identifier(identifier) == expected_normalized


def test_split_identifier_expands_known_abbreviations() -> None:
    assert split_identifier_into_tokens("num_tel") == ("number", "telephone")


def test_split_identifier_drops_stop_tokens() -> None:
    assert split_identifier_into_tokens("date_de_naissance") == ("date", "naissance")


def _string_field(field_name: str) -> SchemaField:
    return SchemaField(name=field_name, path=field_name, data_type=FieldDataType.STRING)


def test_identical_names_score_one() -> None:
    lexical_matcher = LexicalMatcher()
    similarity = lexical_matcher.similarity(_string_field("email"), _string_field("email"))
    assert similarity == pytest.approx(1.0)


def test_similarity_is_symmetric_and_bounded() -> None:
    lexical_matcher = LexicalMatcher()
    forward_similarity = lexical_matcher.similarity(
        _string_field("adresse_email"), _string_field("email")
    )
    reverse_similarity = lexical_matcher.similarity(
        _string_field("email"), _string_field("adresse_email")
    )
    assert forward_similarity == pytest.approx(reverse_similarity)
    assert 0.0 <= forward_similarity <= 1.0


def test_shared_token_scores_above_unrelated_names() -> None:
    lexical_matcher = LexicalMatcher()
    shared_token_similarity = lexical_matcher.similarity(
        _string_field("date_naissance"), _string_field("birthDate")
    )
    unrelated_similarity = lexical_matcher.similarity(
        _string_field("code_agence"), _string_field("loyaltyPoints")
    )
    assert shared_token_similarity > unrelated_similarity


def test_similarity_matrix_has_expected_shape(
    source_schema_definition, target_schema_definition
) -> None:
    lexical_matcher = LexicalMatcher()
    similarity_scores = lexical_matcher.similarity_matrix(
        source_schema_definition.fields, target_schema_definition.fields
    )
    assert similarity_scores.shape == (3, 3)
    assert similarity_scores.min() >= 0.0
    assert similarity_scores.max() <= 1.0
