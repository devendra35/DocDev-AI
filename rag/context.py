from __future__ import annotations

from rag.retriever import RetrievedChunk


def build_context(
    results: list[RetrievedChunk],
) -> str:
    """
    Convert retrieved document chunks into
    a structured context for the LLM.
    """

    if not results:
        return ""

    context_parts: list[str] = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        source = result.source
        page = result.page

        if page is not None:
            location = f"{source}, Page {page}"
        else:
            location = source

        context_parts.append(
            f"[Source {index}: {location}]\n"
            f"{result.text}"
        )

    return "\n\n".join(context_parts)


def build_source_list(
    results: list[RetrievedChunk],
) -> list[dict]:
    """
    Extract source information from retrieved chunks.
    """

    sources = []

    for result in results:
        sources.append(
            {
                "source": result.source,
                "page": result.page,
                "chunk_id": result.chunk.chunk_id,
                "score": result.score,
            }
        )

    return sources