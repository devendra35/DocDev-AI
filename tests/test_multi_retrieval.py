from rag.retriever import Retriever
from rag.vectorstore import FAISSVectorStore


def main():
    print("DocDev AI - Multi-Document Retrieval Test")
    

    print("\n[1] Loading vector store...")

    vector_store = FAISSVectorStore.load(
        "vectorstore"
    )

    print(
        f"[OK] Loaded {vector_store.size} chunks."
    )

    print("\n[2] Creating retriever...")

    retriever = Retriever(
        vector_store=vector_store,
        top_k=4,
    )

    print("[OK] Retriever created.")

    questions = [
        "What is DocDev AI?",
        "What technologies does the system use?",
        "What is retrieval augmented generation?",
    ]

    for question in questions:

        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)

        results = retriever.retrieve(question)

        if not results:
            print("[WARNING] No results found.")
            continue

        for index, result in enumerate(
            results,
            start=1,
        ):
            print(f"\nResult {index}")
            print(f"Source : {result.source}")
            print(f"Page   : {result.page}")
            print(f"Score  : {result.score:.4f}")
            print(f"Text   : {result.text}")

    
    print("MULTI-DOCUMENT RETRIEVAL TEST PASSED!")
   

if __name__ == "__main__":
    main()