"""
Embedding model wrapper — uses Ollama's local nomic-embed-text model.

Provides a simple interface for single-string and batch embedding that
both ChromaDB ingestion and query-time retrieval can call.
"""

from typing import Optional

import httpx

from backend.config import settings


class EmbeddingModel:
    """Wraps Ollama /api/embed endpoint for the configured embedding model."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.EMBEDDING_MODEL
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        """Embed a single string, returning the vector. Results are cached."""
        if text in self._cache:
            return self._cache[text]

        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns {"embeddings": [[...vector...]]}
        vector = data["embeddings"][0]
        self._cache[text] = vector
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings. Ollama supports batch via list input."""
        # Check cache first
        uncached_indices = [i for i, t in enumerate(texts) if t not in self._cache]
        uncached_texts = [texts[i] for i in uncached_indices]

        if uncached_texts:
            resp = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": uncached_texts},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            vectors = data["embeddings"]
            for idx, vec in zip(uncached_indices, vectors):
                self._cache[texts[idx]] = vec

        return [self._cache[t] for t in texts]

    def clear_cache(self) -> None:
        self._cache.clear()
