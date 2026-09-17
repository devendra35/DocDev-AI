from __future__ import annotations

from pathlib import Path
import shutil

from rag.document_registry import DocumentRegistry
from rag.vectorstore import FAISSVectorStore


class DocumentManager:
    """Manage documents registered in DocDev AI."""

    def __init__(
        self,
        vectorstore_path: str | Path = "vectorstore",
    ):
        self.vectorstore_path = Path(vectorstore_path)

        self.registry = DocumentRegistry(
            self.vectorstore_path / "documents.json"
        )

    def list_documents(self) -> list[dict]:
        """Return all registered documents."""

        return self.registry.list_documents()

    def search_documents(
        self,
        query: str,
    ) -> list[dict]:
        """Search documents by filename or file type."""

        query = query.strip().lower()

        if not query:
            return self.list_documents()

        results = []

        for document in self.list_documents():
            filename = document.get(
                "filename",
                "",
            ).lower()

            file_type = document.get(
                "file_type",
                "",
            ).lower()

            if (
                query in filename
                or query in file_type
            ):
                results.append(document)

        return results

    def document_count(self) -> int:
        """Return the number of registered documents."""

        return len(self.list_documents())

    def total_chunks(self) -> int:
        """Return the total number of stored chunks."""

        return sum(
            document.get("chunks", 0)
            for document in self.list_documents()
        )

    def statistics(self) -> dict:
        """
        Return document and vector store statistics.
        """

        index_file = (
            self.vectorstore_path
            / "index.faiss"
        )

        chunks_file = (
            self.vectorstore_path
            / "chunks.json"
        )

        documents = self.list_documents()

        stats = {
            "documents": len(documents),
            "registered_chunks": sum(
                document.get("chunks", 0)
                for document in documents
            ),
            "vector_store_exists": (
                index_file.exists()
                and chunks_file.exists()
            ),
            "vector_dimension": None,
            "vector_store_chunks": 0,
            "status": "Empty",
        }

        if (
            index_file.exists()
            and chunks_file.exists()
        ):
            try:
                vector_store = (
                    FAISSVectorStore.load(
                        self.vectorstore_path
                    )
                )

                stats["vector_dimension"] = (
                    vector_store.dimension
                )

                stats["vector_store_chunks"] = (
                    vector_store.size
                )

                if vector_store.size > 0:
                    stats["status"] = "Ready"

            except Exception:
                stats["status"] = "Invalid"

        return stats

    def delete_document(
        self,
        file_hash: str,
    ) -> dict:
        """
        Delete a document and rebuild the FAISS index.

        The document is identified by its SHA-256 hash.
        """

        if not file_hash:
            raise ValueError(
                "Document hash cannot be empty."
            )

        if file_hash not in self.registry.documents:
            raise ValueError(
                "Document not found in registry."
            )

        document = self.registry.documents[file_hash]

        index_file = (
            self.vectorstore_path
            / "index.faiss"
        )

        chunks_file = (
            self.vectorstore_path
            / "chunks.json"
        )

        if (
            not index_file.exists()
            or not chunks_file.exists()
        ):
            raise RuntimeError(
                "Vector store does not exist."
            )

        vector_store = FAISSVectorStore.load(
            self.vectorstore_path
        )

        remaining_chunks = [
            chunk
            for chunk in vector_store.chunks
            if chunk.source
            != document["filename"]
        ]

        removed_chunks = (
            vector_store.size
            - len(remaining_chunks)
        )

        if removed_chunks == 0:
            raise RuntimeError(
                "No chunks belonging to this document "
                "were found in the vector store."
            )

        del self.registry.documents[file_hash]

        if remaining_chunks:

            from rag.embeddings import embed_chunks

            embeddings = embed_chunks(
                remaining_chunks
            )

            dimension = len(embeddings[0])

            new_vector_store = FAISSVectorStore(
                dimension=dimension
            )

            new_vector_store.add(
                chunks=remaining_chunks,
                embeddings=embeddings,
            )

            new_vector_store.save(
                self.vectorstore_path
            )

            self.registry._save()

        else:

            shutil.rmtree(
                self.vectorstore_path
            )

            self.vectorstore_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.registry = DocumentRegistry(
                self.vectorstore_path
                / "documents.json"
            )

        return {
            "filename": document["filename"],
            "hash": file_hash,
            "chunks_removed": removed_chunks,
            "remaining_documents": (
                self.document_count()
            ),
            "remaining_chunks": (
                self.total_chunks()
            ),
        }