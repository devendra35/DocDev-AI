from rag.ingestion import ingest_document


def main():
    print("DocDev AI - Document Ingestion Test")
    

    file_path = "data/uploads/test.pdf"

    print("\n[1] Ingesting document...")
    print(f"File: {file_path}")

    result = ingest_document(file_path)

    print("\n[OK] Document ingestion completed.")

   
    print("INGESTION RESULT")
    

    for key, value in result.items():
        print(f"{key}: {value}")

  
    print("DOCUMENT INGESTION TEST PASSED!")
    


if __name__ == "__main__":
    main()