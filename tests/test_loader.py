from pathlib import Path

from rag.loader import load_document


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


TEST_FILES = [
    UPLOAD_DIR / "test.txt",
    UPLOAD_DIR / "test.pdf",
    UPLOAD_DIR / "test.docx",
]


def test_document(file_path: Path) -> bool:
    print(f"\nTesting: {file_path.name}")
    

    documents = load_document(file_path)

    if not documents:
        print("[FAIL] No documents loaded")
        return False

    print(f"[OK] Loaded units: {len(documents)}")

    for document in documents:
        print(f"Source: {document.source}")
        print(f"Type: {document.file_type}")
        print(f"Page: {document.page}")
        print(f"Characters: {len(document.text)}")

        assert document.text.strip()
        assert document.source == file_path.name
        assert document.file_type == file_path.suffix.lower().replace(".", "")

    print("[OK] Metadata and text validation passed")

    return True


def main():
    print("DocDev AI - Multi-Format Document Loader Test")
   

    passed = 0

    for file_path in TEST_FILES:
        if not file_path.exists():
            print(f"[FAIL] Missing file: {file_path}")
            continue

        try:
            if test_document(file_path):
                passed += 1

        except Exception as exc:
            print(f"[FAIL] {exc}")

   
    print(f"Result: {passed}/{len(TEST_FILES)} tests passed")

    if passed == len(TEST_FILES):
        print("ALL DOCUMENT LOADER TESTS PASSED!")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()