from pathlib import Path

from rag.chunker import chunk_documents
from rag.embeddings import embed_chunks
from rag.loader import load_document
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
    / "test_store"
)


def main():
    print("DocDev AI - FAISS Vector Store Test")
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

    print(
        f"[OK] Embedding dimension: "
        f"{dimension}"
    )

    # --------------------------------------------------
    #  Create vector store
    # --------------------------------------------------

    print("\n[4] Creating FAISS vector store...")

    store = FAISSVectorStore(
        dimension=dimension
    )

    store.add(
        chunks,
        embeddings,
    )

    print(
        f"[OK] Vectors stored: "
        f"{store.size}"
    )

    assert store.size == len(chunks)

    # --------------------------------------------------
    #  Similarity search
    # --------------------------------------------------

    print("\n[5] Testing similarity search...")

    query = (
        "What does DocDev AI use for "
        "document search?"
    )

    from rag.embeddings import embed_texts

    query_embedding = embed_texts(
        [query]
    )[0]

    results = store.search(
        query_embedding,
        top_k=2,
    )

    print(
        f"[OK] Search results: "
        f"{len(results)}"
    )

    if not results:
        raise RuntimeError(
            "FAISS search returned no results."
        )

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        print(f"\nResult {rank}")
        print(f"Score: {score:.4f}")
        print(f"Source: {chunk.source}")
        print(f"Page: {chunk.page}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Text: {chunk.text[:200]}")

    # --------------------------------------------------
    #  Save vector store
    # --------------------------------------------------

    print("\n[6] Saving vector store...")

    store.save(VECTORSTORE_DIR)

    print(
        f"[OK] Saved to: "
        f"{VECTORSTORE_DIR}"
    )

    assert (
        VECTORSTORE_DIR
        / "index.faiss"
    ).exists()

    assert (
        VECTORSTORE_DIR
        / "chunks.json"
    ).exists()

    # --------------------------------------------------
    #  Load vector store
    # --------------------------------------------------

    print("\n[7] Loading vector store...")

    loaded_store = FAISSVectorStore.load(
        VECTORSTORE_DIR
    )

    print(
        f"[OK] Loaded vectors: "
        f"{loaded_store.size}"
    )

    assert loaded_store.size == store.size
    assert len(loaded_store.chunks) == len(
        store.chunks
    )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("FAISS VECTOR STORE TEST PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()