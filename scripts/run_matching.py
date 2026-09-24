import argparse
import sys
from pathlib import Path

from x_bridge.application.matcher_factory import build_matcher
from x_bridge.application.matching_service import MatchingService
from x_bridge.application.reporting import render_mapping_json, render_mapping_table
from x_bridge.config import MatchingConfig, ThresholdSettings
from x_bridge.domain.enums import MatchingMethod
from x_bridge.domain.models import MappingResult
from x_bridge.infrastructure.schema_analyzer import load_schema_from_file

DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description="Run X-Bridge schema matching")
    argument_parser.add_argument(
        "--method",
        type=MatchingMethod,
        choices=tuple(MatchingMethod),
        default=MatchingMethod.HYBRID,
    )
    argument_parser.add_argument(
        "--source", type=Path, default=DEFAULT_DATA_DIRECTORY / "source_schema.json"
    )
    argument_parser.add_argument(
        "--target", type=Path, default=DEFAULT_DATA_DIRECTORY / "target_schema.json"
    )
    argument_parser.add_argument("--output-format", choices=("table", "json"), default="table")
    argument_parser.add_argument("--output-file", type=Path, default=None)
    argument_parser.add_argument("--automatic-threshold", type=float, default=None)
    argument_parser.add_argument("--review-threshold", type=float, default=None)
    argument_parser.add_argument("--allow-many-to-one", action="store_true")
    return argument_parser


def build_config_from_arguments(arguments: argparse.Namespace) -> MatchingConfig:
    default_thresholds = ThresholdSettings()
    thresholds = ThresholdSettings(
        automatic_threshold=arguments.automatic_threshold
        if arguments.automatic_threshold is not None
        else default_thresholds.automatic_threshold,
        review_threshold=arguments.review_threshold
        if arguments.review_threshold is not None
        else default_thresholds.review_threshold,
    )
    return MatchingConfig(thresholds=thresholds, allow_many_to_one=arguments.allow_many_to_one)


def run_matching(arguments: argparse.Namespace) -> MappingResult:
    config = build_config_from_arguments(arguments)
    matcher = build_matcher(arguments.method, config)
    matching_service = MatchingService(matcher, config=config)
    return matching_service.match(
        load_schema_from_file(arguments.source), load_schema_from_file(arguments.target)
    )


def main() -> int:
    arguments = build_argument_parser().parse_args()
    mapping_result = run_matching(arguments)
    rendered_output = (
        render_mapping_json(mapping_result)
        if arguments.output_format == "json"
        else render_mapping_table(mapping_result)
    )

    if arguments.output_file:
        arguments.output_file.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_file.write_text(rendered_output + "\n", encoding="utf-8")
        print(f"written to {arguments.output_file}")
    else:
        print(rendered_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
