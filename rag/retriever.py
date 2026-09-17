from __future__ import annotations

from dataclasses import dataclass

from rag.embeddings import embed_texts
from rag.vectorstore import FAISSVectorStore
from rag.chunker import DocumentChunk


@dataclass(frozen=True)
class RetrievedChunk:
    """
    Represents a chunk returned by similarity search.
    """

    chunk: DocumentChunk
    score: float

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def page(self) -> int | None:
        return self.chunk.page

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> dict:
        return self.chunk.metadata


class Retriever:
    """
    Retrieves relevant document chunks
    from the FAISS vector store.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        top_k: int = 4,
    ):
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query.
        """

        query = query.strip()

        if not query:
            return []

        query_embedding = embed_texts(
            [query]
        )[0]

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        return [
            RetrievedChunk(
                chunk=chunk,
                score=score,
            )
            for chunk, score in results
        ]


def create_retriever(
    vector_store: FAISSVectorStore,
    top_k: int = 4,
) -> Retriever:
    """
    Create a configured retriever.
    """

    return Retriever(
        vector_store=vector_store,
        top_k=top_k,
    )