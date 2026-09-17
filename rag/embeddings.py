from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from rag.chunker import DocumentChunk


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model(
    model_name: str = MODEL_NAME,
) -> SentenceTransformer:
    """
    Load and cache the embedding model.

    The model is loaded only once during the application
    lifetime.
    """

    return SentenceTransformer(model_name)


def embed_texts(
    texts: list[str],
    model_name: str = MODEL_NAME,
) -> list[list[float]]:
    """
    Convert text strings into embedding vectors.
    """

    if not texts:
        return []

    model = get_embedding_model(model_name)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_chunks(
    chunks: list[DocumentChunk],
    model_name: str = MODEL_NAME,
) -> list[list[float]]:
    """
    Convert DocumentChunk objects into embedding vectors.
    """

    texts = [chunk.text for chunk in chunks]

    return embed_texts(
        texts,
        model_name=model_name,
    )