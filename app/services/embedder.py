from abc import ABC, abstractmethod
import numpy as np
from openai import OpenAI
from app.config import settings


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder(EmbeddingProvider):
    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


def get_embedder() -> EmbeddingProvider:
    return OpenAIEmbedder()


def embed_in_batches(texts: list[str], batch_size: int = 100) -> np.ndarray:
    embedder = get_embedder()
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_vectors.extend(embedder.embed_batch(batch))
    return np.array(all_vectors, dtype=np.float32)
