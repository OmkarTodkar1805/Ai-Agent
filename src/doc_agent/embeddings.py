"""Local embedding model.

Embeddings run in-process through FastEmbed's ONNX runtime rather than a hosted
API. That removes the per-request quota that would otherwise cap a public demo,
and keeps the container small: no torch, no CUDA wheels.
"""

from functools import lru_cache

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384


@lru_cache(maxsize=1)
def model() -> TextEmbedding:
    """Load the embedding model once and reuse it.

    The first call downloads roughly 130MB of ONNX weights, so a container
    should warm this at startup rather than on the first user request.
    """
    return TextEmbedding(model_name=MODEL_NAME)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages for storage."""
    return [vector.tolist() for vector in model().embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed one question for search.

    Index and query must use the same model: vectors from different models are
    not comparable, and the failure is silent, returning plausible-looking but
    meaningless neighbours.
    """
    return next(iter(model().query_embed([text]))).tolist()
