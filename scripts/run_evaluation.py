import argparse
import sys
from pathlib import Path

from x_bridge.application.matcher_factory import build_matcher
from x_bridge.application.matching_service import MatchingService
from x_bridge.application.reporting import render_comparison_table, render_evaluation_report
from x_bridge.config import MatchingConfig
from x_bridge.domain.enums import MatchingMethod
from x_bridge.domain.models import EvaluationResult
from x_bridge.evaluation.evaluator import MappingEvaluator, load_ground_truth
from x_bridge.infrastructure.embedding_service import SentenceTransformerEncoder
from x_bridge.infrastructure.schema_analyzer import load_schema_from_file

DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description="Evaluate X-Bridge matching experiments")
    argument_parser.add_argument(
        "--method",
        type=MatchingMethod,
        choices=tuple(MatchingMethod),
        action="append",
        default=None,
    )
    argument_parser.add_argument(
        "--source", type=Path, default=DEFAULT_DATA_DIRECTORY / "source_schema.json"
    )
    argument_parser.add_argument(
        "--target", type=Path, default=DEFAULT_DATA_DIRECTORY / "target_schema.json"
    )
    argument_parser.add_argument(
        "--ground-truth", type=Path, default=DEFAULT_DATA_DIRECTORY / "ground_truth.json"
    )
    argument_parser.add_argument("--include-review-required", action="store_true")
    argument_parser.add_argument("--detailed", action="store_true")
    return argument_parser


def evaluate_methods(arguments: argparse.Namespace) -> tuple[EvaluationResult, ...]:
    config = MatchingConfig()
    source_schema = load_schema_from_file(arguments.source)
    target_schema = load_schema_from_file(arguments.target)
    ground_truth_mappings = load_ground_truth(arguments.ground_truth)
    evaluator = MappingEvaluator(include_review_required=arguments.include_review_required)
    shared_encoder = SentenceTransformerEncoder(config.semantic.model_name)

    evaluation_results: list[EvaluationResult] = []
    for matching_method in arguments.method or list(MatchingMethod):
        matcher = build_matcher(matching_method, config, encoder=shared_encoder)
        mapping_result = MatchingService(matcher, config=config).match(
            source_schema, target_schema
        )
        evaluation_results.append(evaluator.evaluate(mapping_result, ground_truth_mappings))
    return tuple(evaluation_results)


def main() -> int:
    arguments = build_argument_parser().parse_args()
    evaluation_results = evaluate_methods(arguments)

    print("X-Bridge Experiment Comparison")
    print("=" * 30)
    print()
    print(render_comparison_table(evaluation_results))

    if arguments.detailed:
        for evaluation_result in evaluation_results:
            print()
            print(render_evaluation_report(evaluation_result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
