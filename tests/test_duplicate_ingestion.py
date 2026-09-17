from pathlib import Path

from rag.ingestion import ingest_document


def main():
    print("DocDev AI - Duplicate Ingestion Test")
   

    file_path = Path("data/uploads/test.txt")

    # Use a separate test vector store
    vectorstore_path = Path(
        "vectorstore/duplicate_test"
    )

    print("\n[1] First ingestion...")

    result1 = ingest_document(
        file_path=file_path,
        vectorstore_path=vectorstore_path,
    )

    print(result1)

    if result1["status"] != "added":
        raise RuntimeError(
            "First ingestion should add the document."
        )

    print("[OK] First ingestion added.")

    print("\n[2] Second ingestion...")

    result2 = ingest_document(
        file_path=file_path,
        vectorstore_path=vectorstore_path,
    )

    print(result2)

    if result2["status"] != "skipped":
        raise RuntimeError(
            "Second ingestion should be skipped."
        )

    print("[OK] Duplicate document skipped.")

   
    print("DUPLICATE INGESTION TEST PASSED!")
 


if __name__ == "__main__":
    main()