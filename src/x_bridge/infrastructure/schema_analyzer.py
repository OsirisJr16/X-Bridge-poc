import json
from pathlib import Path
from typing import Any

from x_bridge.domain.enums import FieldDataType
from x_bridge.domain.models import SchemaDefinition, SchemaField

PATH_SEPARATOR = "."
_CONTAINER_TYPES = frozenset({FieldDataType.OBJECT, FieldDataType.ARRAY})


def _resolve_declared_type(property_schema: dict[str, Any]) -> FieldDataType:
    declared_type = property_schema.get("type")
    if isinstance(declared_type, list):
        non_null_types = [keyword for keyword in declared_type if keyword != "null"]
        declared_type = non_null_types[0] if non_null_types else "null"
    return FieldDataType.from_schema_keyword(declared_type)


def _join_path(parent_path: str, field_name: str) -> str:
    return f"{parent_path}{PATH_SEPARATOR}{field_name}" if parent_path else field_name


def _extract_fields(
    object_schema: dict[str, Any], parent_path: str = ""
) -> list[SchemaField]:
    properties = object_schema.get("properties", {})
    required_names = frozenset(object_schema.get("required", []))
    extracted_fields: list[SchemaField] = []

    for field_name, property_schema in properties.items():
        data_type = _resolve_declared_type(property_schema)
        field_path = _join_path(parent_path, field_name)

        if data_type is FieldDataType.OBJECT and "properties" in property_schema:
            extracted_fields.extend(_extract_fields(property_schema, field_path))
            continue

        if data_type is FieldDataType.ARRAY and "properties" in property_schema.get("items", {}):
            extracted_fields.extend(_extract_fields(property_schema["items"], field_path))
            continue

        extracted_fields.append(
            SchemaField(
                name=field_name,
                path=field_path,
                data_type=data_type,
                description=property_schema.get("description"),
                is_required=field_name in required_names,
            )
        )

    return extracted_fields


def analyze_json_schema(json_schema: dict[str, Any], schema_name: str | None = None) -> SchemaDefinition:
    resolved_name = schema_name or json_schema.get("title") or "schema"
    return SchemaDefinition(name=resolved_name, fields=tuple(_extract_fields(json_schema)))


def load_schema_from_file(schema_path: Path) -> SchemaDefinition:
    json_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return analyze_json_schema(json_schema, schema_name=json_schema.get("title") or schema_path.stem)


def is_container_type(data_type: FieldDataType) -> bool:
    return data_type in _CONTAINER_TYPES
