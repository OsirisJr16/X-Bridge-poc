import json

from x_bridge.domain.models import EvaluationResult, MappingResult

_SOURCE_COLUMN_WIDTH = 24
_TARGET_COLUMN_WIDTH = 24
_METHOD_COLUMN_WIDTH = 12
_SCORE_COLUMN_WIDTH = 11


def render_mapping_table(mapping_result: MappingResult) -> str:
    header = (
        f"{'Source':<{_SOURCE_COLUMN_WIDTH}}"
        f"{'Target':<{_TARGET_COLUMN_WIDTH}}"
        f"{'Similarity':<{_SCORE_COLUMN_WIDTH}}"
        f"{'Confidence':<{_SCORE_COLUMN_WIDTH}}"
        "Status"
    )
    separator = "-" * (len(header) + 8)
    rows = [
        f"{mapping.source_field:<{_SOURCE_COLUMN_WIDTH}}"
        f"{mapping.target_field:<{_TARGET_COLUMN_WIDTH}}"
        f"{mapping.similarity_score:<{_SCORE_COLUMN_WIDTH}.2f}"
        f"{mapping.confidence_score:<{_SCORE_COLUMN_WIDTH}.2f}"
        f"{mapping.status.value}"
        for mapping in mapping_result.mappings
    ]

    summary_lines = [
        "Status:",
        f"{len(mapping_result.automatic_mappings)} automatic mappings",
        f"{len(mapping_result.review_required_mappings)} mappings requiring review",
        f"{len(mapping_result.unmatched_source_fields)} unmatched source fields",
        f"{len(mapping_result.unmatched_target_fields)} unmatched target fields",
    ]
    if mapping_result.unmatched_source_fields:
        summary_lines.append(
            "Unmatched source: " + ", ".join(mapping_result.unmatched_source_fields)
        )
    if mapping_result.unmatched_target_fields:
        summary_lines.append(
            "Unmatched target: " + ", ".join(mapping_result.unmatched_target_fields)
        )

    title = f"X-Bridge Semantic Matching ({mapping_result.matching_method.value})"
    return "\n".join(
        [title, "=" * len(title), "", header, separator, *rows, "", *summary_lines]
    )


def render_mapping_json(mapping_result: MappingResult) -> str:
    return json.dumps(
        [mapping.model_dump(mode="json") for mapping in mapping_result.mappings],
        indent=2,
        ensure_ascii=False,
    )


def render_evaluation_report(evaluation_result: EvaluationResult) -> str:
    coverage = evaluation_result.coverage
    return "\n".join(
        [
            f"Method: {evaluation_result.matching_method.value}",
            "",
            "Model performance",
            f"  true positives     {evaluation_result.true_positives}",
            f"  false positives    {evaluation_result.false_positives}",
            f"  false negatives    {evaluation_result.false_negatives}",
            f"  true negatives     {evaluation_result.true_negatives}",
            f"  precision          {evaluation_result.precision:.3f}",
            f"  recall             {evaluation_result.recall:.3f}",
            f"  f1 score           {evaluation_result.f1_score:.3f}",
            f"  accuracy           {evaluation_result.accuracy:.3f}",
            "",
            "Matching coverage",
            f"  automatic rate     {coverage.automatic_mapping_rate:.3f}",
            f"  review rate        {coverage.review_required_rate:.3f}",
            f"  unmatched rate     {coverage.unmatched_rate:.3f}",
            "",
            "Human validation requirements",
            (
                f"  mappings to review {len(evaluation_result.incorrect_mappings)} incorrect, "
                f"{len(evaluation_result.missed_mappings)} missed"
            ),
        ]
    )


def render_comparison_table(evaluation_results: tuple[EvaluationResult, ...]) -> str:
    header = (
        f"{'Method':<{_METHOD_COLUMN_WIDTH}}"
        f"{'Precision':<{_SCORE_COLUMN_WIDTH}}"
        f"{'Recall':<{_SCORE_COLUMN_WIDTH}}"
        f"{'F1':<{_SCORE_COLUMN_WIDTH}}"
        f"{'Auto rate':<{_SCORE_COLUMN_WIDTH}}"
        "Review rate"
    )
    rows = [
        f"{evaluation_result.matching_method.value:<{_METHOD_COLUMN_WIDTH}}"
        f"{evaluation_result.precision:<{_SCORE_COLUMN_WIDTH}.3f}"
        f"{evaluation_result.recall:<{_SCORE_COLUMN_WIDTH}.3f}"
        f"{evaluation_result.f1_score:<{_SCORE_COLUMN_WIDTH}.3f}"
        f"{evaluation_result.coverage.automatic_mapping_rate:<{_SCORE_COLUMN_WIDTH}.3f}"
        f"{evaluation_result.coverage.review_required_rate:.3f}"
        for evaluation_result in evaluation_results
    ]
    return "\n".join([header, "-" * (len(header) + 4), *rows])
