from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.cleaner import clean_text
from rag.loader import LoadedDocument


@dataclass(frozen=True)
class DocumentChunk:
    """
    Represents a chunk ready for embedding.
    """

    text: str
    chunk_id: int
    source: str
    file_type: str
    page: int | None = None

    @property
    def metadata(self) -> dict:
        return {
            "source": self.source,
            "file_type": self.file_type,
            "page": self.page,
            "chunk_id": self.chunk_id,
        }


def create_text_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> RecursiveCharacterTextSplitter:
    """
    Create the text splitter used by DocDev AI.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
        length_function=len,
        is_separator_regex=False,
    )


def chunk_document(
    document: LoadedDocument,
    splitter: RecursiveCharacterTextSplitter,
    starting_id: int = 0,
) -> list[DocumentChunk]:
    """
    Clean and split one loaded document.
    """

    cleaned_text = clean_text(document.text)

    if not cleaned_text:
        return []

    text_chunks = splitter.split_text(cleaned_text)

    chunks = []

    for index, text in enumerate(text_chunks):
        text = text.strip()

        if not text:
            continue

        chunks.append(
            DocumentChunk(
                text=text,
                chunk_id=starting_id + index,
                source=document.source,
                file_type=document.file_type,
                page=document.page,
            )
        )

    return chunks


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    """
    Clean and chunk multiple loaded documents.
    """

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: list[DocumentChunk] = []

    for document in documents:
        starting_id = len(all_chunks)

        chunks = chunk_document(
            document=document,
            splitter=splitter,
            starting_id=starting_id,
        )

        all_chunks.extend(chunks)

    return all_chunks