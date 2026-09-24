import json
from pathlib import Path

from x_bridge.domain.enums import FieldDataType
from x_bridge.infrastructure.schema_analyzer import analyze_json_schema, load_schema_from_file

NESTED_SCHEMA = {
    "title": "Customer",
    "type": "object",
    "required": ["customerId"],
    "properties": {
        "customerId": {"type": "string", "description": "Unique identifier"},
        "loyaltyPoints": {"type": "integer"},
        "address": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {"type": "string"},
                "geo": {
                    "type": "object",
                    "properties": {"latitude": {"type": "number"}},
                },
            },
        },
        "nickname": {"type": ["string", "null"]},
        "unknownField": {},
    },
}


def test_analyze_extracts_leaf_fields_with_full_paths() -> None:
    schema_definition = analyze_json_schema(NESTED_SCHEMA)
    assert schema_definition.field_paths() == (
        "customerId",
        "loyaltyPoints",
        "address.city",
        "address.geo.latitude",
        "nickname",
        "unknownField",
    )


def test_analyze_preserves_name_type_description_and_required() -> None:
    schema_definition = analyze_json_schema(NESTED_SCHEMA)
    fields_by_path = {field.path: field for field in schema_definition.fields}

    assert fields_by_path["customerId"].description == "Unique identifier"
    assert fields_by_path["customerId"].is_required is True
    assert fields_by_path["loyaltyPoints"].data_type is FieldDataType.INTEGER
    assert fields_by_path["address.city"].name == "city"
    assert fields_by_path["address.city"].is_required is True
    assert fields_by_path["address.geo.latitude"].data_type is FieldDataType.NUMBER


def test_nullable_and_missing_types_resolve_predictably() -> None:
    fields_by_path = {
        field.path: field for field in analyze_json_schema(NESTED_SCHEMA).fields
    }
    assert fields_by_path["nickname"].data_type is FieldDataType.STRING
    assert fields_by_path["unknownField"].data_type is FieldDataType.UNKNOWN


def test_load_schema_from_file_uses_title_as_name(tmp_path: Path) -> None:
    schema_path = tmp_path / "customer.json"
    schema_path.write_text(json.dumps(NESTED_SCHEMA), encoding="utf-8")
    assert load_schema_from_file(schema_path).name == "Customer"


def test_project_schemas_expose_expected_field_counts() -> None:
    data_directory = Path(__file__).resolve().parents[1] / "data"
    source_schema = load_schema_from_file(data_directory / "source_schema.json")
    target_schema = load_schema_from_file(data_directory / "target_schema.json")

    assert len(source_schema.fields) == 20
    assert len(target_schema.fields) == 20
    assert "adresse.code_postal" in source_schema.field_paths()
    assert "address.postalCode" in target_schema.field_paths()
