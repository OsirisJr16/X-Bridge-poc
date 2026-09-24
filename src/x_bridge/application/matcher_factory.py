from x_bridge.application.matching_service import FieldSimilarityMatcher
from x_bridge.config import MatchingConfig
from x_bridge.domain.enums import MatchingMethod
from x_bridge.infrastructure.embedding_service import SentenceTransformerEncoder, TextEncoder
from x_bridge.matching.hybrid_matcher import HybridMatcher
from x_bridge.matching.lexical_matcher import LexicalMatcher
from x_bridge.matching.semantic_matcher import SemanticMatcher


def build_matcher(
    matching_method: MatchingMethod,
    config: MatchingConfig,
    encoder: TextEncoder | None = None,
) -> FieldSimilarityMatcher:
    if matching_method is MatchingMethod.LEXICAL:
        return LexicalMatcher(config.lexical)

    resolved_encoder = encoder or SentenceTransformerEncoder(config.semantic.model_name)
    semantic_matcher = SemanticMatcher(resolved_encoder, config.semantic)
    if matching_method is MatchingMethod.SEMANTIC:
        return semantic_matcher

    return HybridMatcher(LexicalMatcher(config.lexical), semantic_matcher, config.weights)
