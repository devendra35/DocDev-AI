from rag.document_registry import DocumentRegistry


def main():
    print("DocDev AI - Document Registry Test")
    print("=" * 60)

    file_path = "data/uploads/test.pdf"

    print("\n[1] Creating registry...")

    registry = DocumentRegistry(
        "vectorstore/test_documents.json"
    )

    print("[OK] Registry created.")

    print("\n[2] Calculating file hash...")

    file_hash = registry.calculate_hash(
        file_path
    )

    print(f"[OK] SHA-256: {file_hash}")

    if not file_hash:
        raise RuntimeError(
            "File hash was not generated."
        )

    print("\n[3] Checking registry...")

    already_exists = registry.contains_hash(
        file_hash
    )

    print(
        f"[OK] Already registered: "
        f"{already_exists}"
    )

    print("\n[4] Adding document...")

    registry.add(
        file_path=file_path,
        file_hash=file_hash,
        chunks=1,
        file_type=".pdf",
    )

    print("[OK] Document registered.")

    print("\n[5] Checking again...")

    if not registry.contains_hash(file_hash):
        raise RuntimeError(
            "Document was not found after registration."
        )

    print("[OK] Duplicate detection works.")

    print("\n[6] Registered documents:")

    for document in registry.list_documents():
        print(document)

    
    print("DOCUMENT REGISTRY TEST PASSED!")
   


if __name__ == "__main__":
    main()