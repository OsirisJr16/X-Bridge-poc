from enum import StrEnum


class FieldDataType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    NULL = "null"
    UNKNOWN = "unknown"

    @classmethod
    def from_schema_keyword(cls, keyword: str | None) -> "FieldDataType":
        if keyword is None:
            return cls.UNKNOWN
        try:
            return cls(keyword.lower())
        except ValueError:
            return cls.UNKNOWN


class MatchingMethod(StrEnum):
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class MappingStatus(StrEnum):
    AUTOMATIC = "automatic"
    REVIEW_REQUIRED = "review_required"
    UNMATCHED = "unmatched"
