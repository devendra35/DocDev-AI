from rag.document_manager import DocumentManager


def main():
    print("DocDev AI - Document Search Test")
   

    manager = DocumentManager(
        "vectorstore/duplicate_test"
    )

    print("\n[1] Checking available documents...")

    documents = manager.list_documents()

    if not documents:
        raise RuntimeError(
            "No documents available for search test."
        )

    for document in documents:
        print(document)

    filename = documents[0]["filename"]

    print(
        f"\n[2] Searching for: {filename}"
    )

    results = manager.search_documents(
        filename
    )

    for result in results:
        print(result)

    if not results:
        raise RuntimeError(
            "Document search returned no results."
        )

    print(
        f"[OK] Found {len(results)} document(s)."
    )

    print("\n[3] Testing case-insensitive search...")

    results = manager.search_documents(
        filename.upper()
    )

    if not results:
        raise RuntimeError(
            "Case-insensitive search failed."
        )

    print("[OK] Case-insensitive search works.")

    print("\n[4] Testing empty query...")

    results = manager.search_documents("")

    if not results:
        raise RuntimeError(
            "Empty query should return all documents."
        )

    print(
        f"[OK] Empty query returned "
        f"{len(results)} document(s)."
    )

  
    print("DOCUMENT SEARCH TEST PASSED!")
   


if __name__ == "__main__":
    main()