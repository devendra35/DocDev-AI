from rag.document_manager import DocumentManager


def main():
    print("DocDev AI - Document Delete Test")
    

    manager = DocumentManager(
        "vectorstore/duplicate_test"
    )

    documents = manager.list_documents()

    if not documents:
        raise RuntimeError(
            "No documents available for deletion test."
        )

    print("\n[1] Documents before deletion:")

    for document in documents:
        print(document)

    document = documents[0]

    file_hash = document["hash"]

    print(
        f"\n[2] Deleting: "
        f"{document['filename']}"
    )

    result = manager.delete_document(
        file_hash
    )

    print(result)

    print("\n[3] Checking registry...")

    remaining = manager.list_documents()

    if remaining:
        raise RuntimeError(
            "Document was not completely deleted."
        )

    print("[OK] Document removed.")

    print("\n[4] Final statistics:")

    print(
        f"Documents: "
        f"{manager.document_count()}"
    )

    print(
        f"Chunks: "
        f"{manager.total_chunks()}"
    )

    
    print("DOCUMENT DELETE TEST PASSED!")
   


if __name__ == "__main__":
    main()