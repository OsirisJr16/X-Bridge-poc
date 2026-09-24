import numpy as np
from numpy.typing import NDArray

from x_bridge.config import ConfidenceSettings


def compute_runner_up_scores(similarity_scores: NDArray[np.float64]) -> NDArray[np.float64]:
    target_count = similarity_scores.shape[1]
    if target_count < 2:
        return np.zeros_like(similarity_scores)

    best_target_indices = similarity_scores.argmax(axis=1)
    descending_scores = np.sort(similarity_scores, axis=1)[:, ::-1]
    best_scores = descending_scores[:, 0]
    second_best_scores = descending_scores[:, 1]

    is_best_candidate = np.zeros_like(similarity_scores, dtype=bool)
    is_best_candidate[np.arange(similarity_scores.shape[0]), best_target_indices] = True
    return np.where(
        is_best_candidate, second_best_scores[:, np.newaxis], best_scores[:, np.newaxis]
    )


def compute_score_margins(similarity_scores: NDArray[np.float64]) -> NDArray[np.float64]:
    runner_up_scores = compute_runner_up_scores(similarity_scores)
    return np.clip(similarity_scores - runner_up_scores, 0.0, 1.0)


class ConfidenceScorer:
    def __init__(self, settings: ConfidenceSettings | None = None) -> None:
        self._settings = settings or ConfidenceSettings()

    def confidence_matrix(
        self,
        similarity_scores: NDArray[np.float64],
        type_compatibility_scores: NDArray[np.float64],
        score_margins: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        margin_factor = np.clip(score_margins / self._settings.margin_saturation, 0.0, 1.0)
        ranking_adjustment = self._settings.margin_bonus_weight * margin_factor
        type_penalty = self._settings.type_penalty_weight * (1.0 - type_compatibility_scores)
        adjusted_scores = similarity_scores * (1.0 + ranking_adjustment - type_penalty)
        return np.clip(adjusted_scores, 0.0, 1.0)
