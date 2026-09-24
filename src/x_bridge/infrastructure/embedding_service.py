from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from x_bridge.config import DEFAULT_EMBEDDING_MODEL


class TextEncoder(Protocol):
    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float64]: ...


class SentenceTransformerEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._embedding_cache: dict[str, NDArray[np.float64]] = {}

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float64]:
        uncached_texts = [text for text in dict.fromkeys(texts) if text not in self._embedding_cache]
        if uncached_texts:
            computed_embeddings = self._load_model().encode(
                uncached_texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for text, embedding in zip(uncached_texts, computed_embeddings):
                self._embedding_cache[text] = np.asarray(embedding, dtype=np.float64)

        return np.vstack([self._embedding_cache[text] for text in texts])

    @property
    def cached_text_count(self) -> int:
        return len(self._embedding_cache)
