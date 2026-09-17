from rag.document_manager import DocumentManager


def main():
    print("DocDev AI - Document Statistics Test")
    

    manager = DocumentManager(
        "vectorstore/duplicate_test"
    )

    print("\n[1] Reading statistics...")

    stats = manager.statistics()

    for key, value in stats.items():
        print(
            f"{key}: {value}"
        )

    print("\n[2] Validating statistics...")

    if stats["documents"] <= 0:
        raise RuntimeError(
            "Document count should be greater than zero."
        )

    if stats["registered_chunks"] <= 0:
        raise RuntimeError(
            "Registered chunks should be greater than zero."
        )

    if not stats["vector_store_exists"]:
        raise RuntimeError(
            "Vector store should exist."
        )

    if stats["vector_dimension"] != 384:
        raise RuntimeError(
            "Expected embedding dimension is 384."
        )

    if stats["vector_store_chunks"] <= 0:
        raise RuntimeError(
            "Vector store should contain chunks."
        )

    if stats["status"] != "Ready":
        raise RuntimeError(
            "Vector store status should be Ready."
        )

    print("[OK] Statistics are valid.")

    
    print("DOCUMENT STATISTICS TEST PASSED!")
    


if __name__ == "__main__":
    main()