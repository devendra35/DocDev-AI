from rag.document_manager import DocumentManager


def main():
    print("DocDev AI - Document Manager Test")
  

    manager = DocumentManager(
        "vectorstore/duplicate_test"
    )

    print("\n[1] Listing documents...")

    documents = manager.list_documents()

    for document in documents:
        print(document)

    print(
        f"\n[OK] Documents: "
        f"{manager.document_count()}"
    )

    print(
        f"[OK] Total chunks: "
        f"{manager.total_chunks()}"
    )

    if not documents:
        raise RuntimeError(
            "No registered documents found."
        )

  
    print("DOCUMENT MANAGER TEST PASSED!")
   


if __name__ == "__main__":
    main()