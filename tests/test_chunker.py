from pathlib import Path

from rag.chunker import chunk_documents
from rag.loader import load_document


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "uploads" / "test.pdf"


def main():
    print("DocDev AI - Chunking Test")
    

    documents = load_document(TEST_FILE)

    print(f"[OK] Loaded document units: {len(documents)}")

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        chunk_overlap=50,
    )

    print(f"[OK] Created chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError("No chunks were created.")

    for chunk in chunks:
        print("\n--- Chunk ---")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print(f"Page: {chunk.page}")
        print(f"Characters: {len(chunk.text)}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Text: {chunk.text[:200]}...")

    # Validate IDs
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert chunk_ids == list(range(len(chunks)))

    # Validate content
    assert all(chunk.text.strip() for chunk in chunks)

    # Validate metadata
    assert all(chunk.source == "test.pdf" for chunk in chunks)

   
    print("Chunking test passed!")


if __name__ == "__main__":
    main()