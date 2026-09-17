from pathlib import Path

from rag.chunker import chunk_documents
from rag.embeddings import embed_chunks
from rag.loader import load_document


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "uploads" / "test.pdf"


def main():
    print("DocDev AI - Embedding Test")
    

    print("\nLoading document...")

    documents = load_document(TEST_FILE)

    print(f"[OK] Loaded document units: {len(documents)}")

    print("\nCreating chunks...")

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        chunk_overlap=50,
    )

    print(f"[OK] Created chunks: {len(chunks)}")

    print("\nGenerating embeddings...")
    print("The first run may take some time because the model is downloaded.")

    embeddings = embed_chunks(chunks)

    print(f"[OK] Embeddings created: {len(embeddings)}")

    if not embeddings:
        raise RuntimeError("No embeddings were created.")

    vector_dimension = len(embeddings[0])

    print(f"[OK] Vector dimension: {vector_dimension}")

    assert len(embeddings) == len(chunks)

    assert all(
        len(vector) == vector_dimension
        for vector in embeddings
    )

    assert vector_dimension == 384

   
    print("Embedding test passed!")


if __name__ == "__main__":
    main()