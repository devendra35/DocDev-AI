from __future__ import annotations

from pathlib import Path

from rag.loader import load_document
from rag.chunker import chunk_documents
from rag.embeddings import embed_chunks
from rag.vectorstore import FAISSVectorStore
from rag.document_registry import DocumentRegistry


class IngestionError(Exception):
    """Raised when document ingestion fails."""


def ingest_document(
    file_path: str | Path,
    vectorstore_path: str | Path = "vectorstore",
) -> dict:
    """
    Load, chunk, embed, and add a document to the vector store.

    Duplicate documents are detected using SHA-256 hashing.
    """

    file_path = Path(file_path)
    vectorstore_path = Path(vectorstore_path)

    if not file_path.exists():
        raise IngestionError(
            f"File not found: {file_path}"
        )

    registry = DocumentRegistry(
        vectorstore_path / "documents.json"
    )

    try:
       
        #  Calculate document hash
       
        file_hash = registry.calculate_hash(file_path)

    
        # Check duplicate
       
        if registry.contains_hash(file_hash):
            return {
                "file": file_path.name,
                "file_type": file_path.suffix.lower(),
                "status": "skipped",
                "reason": "Document already exists.",
                "chunks_added": 0,
                "total_chunks": 0,
                "hash": file_hash,
            }

 
        #  Load document
        
        documents = load_document(file_path)

        if not documents:
            raise IngestionError(
                "No content could be extracted from the document."
            )

        #  Create chunks
      
        chunks = chunk_documents(documents)

        if not chunks:
            raise IngestionError(
                "No chunks were created from the document."
            )

      
        #  Generate embeddings
       
        embeddings = embed_chunks(chunks)

        if not embeddings:
            raise IngestionError(
                "No embeddings were generated."
            )

        dimension = len(embeddings[0])

      
        #  Load existing vector store or create new one
       
        index_file = vectorstore_path / "index.faiss"
        chunks_file = vectorstore_path / "chunks.json"

        if index_file.exists() and chunks_file.exists():
            vector_store = FAISSVectorStore.load(
                vectorstore_path
            )

            if vector_store.dimension != dimension:
                raise IngestionError(
                    "Embedding dimension does not match "
                    "the existing vector store."
                )

            previous_chunks = vector_store.size

        else:
            vector_store = FAISSVectorStore(
                dimension=dimension
            )

            previous_chunks = 0

       
        # Add document to vector store
      
        vector_store.add(
            chunks=chunks,
            embeddings=embeddings,
        )

       
        #  Save vector store
        
        vector_store.save(vectorstore_path)

      
        #  Register document
       
        registry.add(
            file_path=file_path,
            file_hash=file_hash,
            chunks=len(chunks),
            file_type=file_path.suffix.lower(),
        )

       
        #  Return result
    
        return {
            "file": file_path.name,
            "file_type": file_path.suffix.lower(),
            "status": "added",
            "documents": len(documents),
            "chunks_added": len(chunks),
            "previous_chunks": previous_chunks,
            "total_chunks": vector_store.size,
            "embedding_dimension": dimension,
            "hash": file_hash,
            "vectorstore_path": str(vectorstore_path),
        }

    except IngestionError:
        raise

    except Exception as exc:
        raise IngestionError(
            f"Document ingestion failed: {exc}"
        ) from exc


def ingest_documents(
    file_paths: list[str | Path],
    vectorstore_path: str | Path = "vectorstore",
) -> list[dict]:
    """
    Ingest multiple documents with duplicate protection.
    """

    if not file_paths:
        raise IngestionError(
            "No documents were provided."
        )

    results = []

    for file_path in file_paths:
        result = ingest_document(
            file_path=file_path,
            vectorstore_path=vectorstore_path,
        )

        results.append(result)

    return results