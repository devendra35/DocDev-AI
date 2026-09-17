from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from rag.chunker import DocumentChunk


class VectorStoreError(Exception):
    """Base exception for vector store errors."""


class FAISSVectorStore:
    """
    FAISS-based vector store for DocDev AI.

    Stores:
    - FAISS index for vector similarity search
    - Document chunks and metadata separately as JSON
    """

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0.")

        self.dimension = dimension

        # Inner product works well with normalized embeddings
        # and is equivalent to cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)

        self.chunks: list[DocumentChunk] = []

    @property
    def size(self) -> int:
        """Return the number of vectors in the store."""
        return self.index.ntotal

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add document chunks and their embeddings to FAISS.
        """

        if not chunks:
            raise ValueError("No chunks provided.")

        if not embeddings:
            raise ValueError("No embeddings provided.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        vectors = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2-dimensional array."
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"received {vectors.shape[1]}."
            )

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Search for the most similar document chunks.
        """

        if self.size == 0:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_vector = np.asarray(
            [query_embedding],
            dtype=np.float32,
        )

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"received {query_vector.shape[1]}."
            )

        scores, indices = self.index.search(
            query_vector,
            min(top_k, self.size),
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index == -1:
                continue

            results.append(
                (
                    self.chunks[index],
                    float(score),
                )
            )

        return results

    def save(self, directory: str | Path) -> None:
        """
        Save FAISS index and chunk metadata to disk.
        """

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = directory / "index.faiss"
        metadata_path = directory / "chunks.json"

        faiss.write_index(
            self.index,
            str(index_path),
        )

        metadata = [
            {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "file_type": chunk.file_type,
                "page": chunk.page,
            }
            for chunk in self.chunks
        ]

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "FAISSVectorStore":
        """
        Load a previously saved FAISS vector store.
        """

        directory = Path(directory)

        index_path = directory / "index.faiss"
        metadata_path = directory / "chunks.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found: {metadata_path}"
            )

        index = faiss.read_index(
            str(index_path)
        )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        store = cls(
            dimension=index.d
        )

        store.index = index

        store.chunks = [
            DocumentChunk(
                text=item["text"],
                chunk_id=item["chunk_id"],
                source=item["source"],
                file_type=item["file_type"],
                page=item["page"],
            )
            for item in metadata
        ]

        if store.size != len(store.chunks):
            raise VectorStoreError(
                "FAISS index size does not match "
                "stored chunk metadata."
            )

        return store