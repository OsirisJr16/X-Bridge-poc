import re
from types import MappingProxyType

_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_DIGIT_BOUNDARY = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_NON_ALPHANUMERIC = re.compile(r"[^0-9a-z]+")

ABBREVIATION_EXPANSIONS = MappingProxyType(
    {
        "addr": "address",
        "amt": "amount",
        "desc": "description",
        "dob": "date of birth",
        "id": "identifier",
        "nb": "number",
        "nbr": "number",
        "no": "number",
        "num": "number",
        "pct": "percent",
        "qte": "quantite",
        "qty": "quantity",
        "tel": "telephone",
        "ts": "timestamp",
    }
)

_STOP_TOKENS = frozenset({"the", "a", "an", "de", "du", "la", "le", "les", "des", "d", "l"})


def split_identifier_into_tokens(identifier: str) -> tuple[str, ...]:
    spaced_identifier = _CAMEL_CASE_BOUNDARY.sub(" ", identifier)
    spaced_identifier = _DIGIT_BOUNDARY.sub(" ", spaced_identifier)
    spaced_identifier = _NON_ALPHANUMERIC.sub(" ", spaced_identifier.lower())

    expanded_tokens: list[str] = []
    for raw_token in spaced_identifier.split():
        if raw_token in _STOP_TOKENS:
            continue
        expanded_tokens.extend(ABBREVIATION_EXPANSIONS.get(raw_token, raw_token).split())
    return tuple(expanded_tokens)


def normalize_identifier(identifier: str) -> str:
    return " ".join(split_identifier_into_tokens(identifier))


def normalize_field_path(field_path: str) -> str:
    return normalize_identifier(field_path)
