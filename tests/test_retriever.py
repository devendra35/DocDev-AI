from pathlib import Path

from rag.chunker import chunk_documents
from rag.embeddings import embed_chunks
from rag.loader import load_document
from rag.retriever import create_retriever
from rag.vectorstore import FAISSVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "uploads"
    / "test.pdf"
)

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "vectorstore"
    / "retriever_test"
)


def main():
    print("DocDev AI - Retriever Test")
    print("=" * 50)

    # --------------------------------------------------
    #  Load document
    # --------------------------------------------------

    print("\n[1] Loading document...")

    documents = load_document(TEST_FILE)

    print(
        f"[OK] Loaded document units: "
        f"{len(documents)}"
    )

    # --------------------------------------------------
    #  Create chunks
    # --------------------------------------------------

    print("\n[2] Creating chunks...")

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        chunk_overlap=50,
    )

    print(
        f"[OK] Created chunks: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------
    #  Generate embeddings
    # --------------------------------------------------

    print("\n[3] Generating embeddings...")

    embeddings = embed_chunks(chunks)

    print(
        f"[OK] Generated embeddings: "
        f"{len(embeddings)}"
    )

    dimension = len(embeddings[0])

    # --------------------------------------------------
    #  Build FAISS store
    # --------------------------------------------------

    print("\n[4] Building FAISS vector store...")

    store = FAISSVectorStore(
        dimension=dimension
    )

    store.add(
        chunks,
        embeddings,
    )

    print(
        f"[OK] Stored vectors: "
        f"{store.size}"
    )

    # --------------------------------------------------
    #  Save store
    # --------------------------------------------------

    print("\n[5] Saving vector store...")

    store.save(VECTORSTORE_DIR)

    print("[OK] Vector store saved.")

    # --------------------------------------------------
    #  Load store
    # --------------------------------------------------

    print("\n[6] Loading vector store...")

    loaded_store = FAISSVectorStore.load(
        VECTORSTORE_DIR
    )

    print(
        f"[OK] Loaded vectors: "
        f"{loaded_store.size}"
    )

    # --------------------------------------------------
    #  Create retriever
    # --------------------------------------------------

    print("\n[7] Creating retriever...")

    retriever = create_retriever(
        vector_store=loaded_store,
        top_k=4,
    )

    print("[OK] Retriever created.")

    # --------------------------------------------------
    #  Test retrieval
    # --------------------------------------------------

    print("\n[8] Testing retrieval...")

    query = (
        "What does DocDev AI use "
        "for document search?"
    )

    print(f"Query: {query}")

    results = retriever.retrieve(query)

    print(
        f"[OK] Retrieved chunks: "
        f"{len(results)}"
    )

    if not results:
        raise RuntimeError(
            "Retriever returned no results."
        )

    # --------------------------------------------------
    #  Display results
    # --------------------------------------------------

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(f"\n--- Result {rank} ---")

        print(
            f"Similarity Score: "
            f"{result.score:.4f}"
        )

        print(
            f"Source: "
            f"{result.source}"
        )

        print(
            f"Page: "
            f"{result.page}"
        )

        print(
            f"Chunk ID: "
            f"{result.chunk.chunk_id}"
        )

        print(
            f"Text: "
            f"{result.text[:250]}"
        )

        print(
            f"Metadata: "
            f"{result.metadata}"
        )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    assert len(results) <= 4

    assert all(
        result.text.strip()
        for result in results
    )

    assert all(
        result.source == "test.pdf"
        for result in results
    )

    assert all(
        isinstance(result.score, float)
        for result in results
    )

    # --------------------------------------------------
    # Empty query test
    # --------------------------------------------------

    print("\n[9] Testing empty query...")

    empty_results = retriever.retrieve("")

    assert empty_results == []

    print("[OK] Empty query handled correctly.")

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("RETRIEVER TEST PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()