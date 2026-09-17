from rag.ingestion import ingest_documents
from rag.vectorstore import FAISSVectorStore


def main():
    print("DocDev AI - Multi-Document Ingestion Test")
    

    files = [
        "data/uploads/test.pdf",
        "data/uploads/test.docx",
        "data/uploads/test.txt",
    ]

    print("\n[1] Documents to ingest:")

    for file in files:
        print(f" - {file}")

    print("\n[2] Ingesting documents...")

    results = ingest_documents(files)

    print("[OK] All documents ingested.")

   
    print("INGESTION RESULTS")
  

    for result in results:
        print(
            f"{result['file']} -> "
            f"{result['chunks_added']} chunks"
        )

    print("\n[3] Loading final vector store...")

    vector_store = FAISSVectorStore.load(
        "vectorstore"
    )

    print(
        f"[OK] Final vector store contains "
        f"{vector_store.size} chunks."
    )

    if vector_store.size < 3:
        raise RuntimeError(
            "Expected at least 3 chunks in the "
            "multi-document vector store."
        )

   
    print("MULTI-DOCUMENT INGESTION TEST PASSED!")
   


if __name__ == "__main__":
    main()