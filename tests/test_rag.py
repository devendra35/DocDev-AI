from models.llm import GeminiLLM
from rag.retriever import Retriever
from rag.vectorstore import FAISSVectorStore
from rag.chain import RAGChain


def main():
    print("DocDev AI - Complete RAG Test")
    print("=" * 60)

   
    #  Load FAISS vector store
   

    print("\n[1] Loading vector store...")

    vector_store = FAISSVectorStore.load(
        "vectorstore"
    )

    print(
        f"[OK] Vector store loaded: "
        f"{vector_store.size} chunks"
    )

   
    # Create retriever
   

    print("\n[2] Creating retriever...")

    retriever = Retriever(
        vector_store=vector_store,
        top_k=4,
    )

    print("[OK] Retriever created.")

    
    #  Create Gemini LLM
   

    print("\n[3] Creating Gemini LLM...")

    llm = GeminiLLM()

    print("[OK] Gemini LLM created.")

   
    #  Create RAG chain
   

    print("\n[4] Creating RAG chain...")

    rag = RAGChain(
        retriever=retriever,
        llm=llm,
    )

    print("[OK] RAG chain created.")

   
    #  Ask a question
   

    question = "What is retrieval augmented generation?"

    print("\n[5] Asking question...")
    print(f"Question: {question}")

    response = rag.invoke(question)

   
    #  Display answer
   

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(response.answer)

   
    #  Display sources
   

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for index, source in enumerate(
        response.sources,
        start=1,
    ):
        print(
            f"{index}. "
            f"{source['source']} "
            f"| Page: {source['page']} "
            f"| Score: {source['score']:.4f}"
        )

  
    # Validation
  

    if not response.answer:
        raise RuntimeError(
            "RAG returned an empty answer."
        )

    if not response.retrieved_chunks:
        raise RuntimeError(
            "RAG returned no retrieved chunks."
        )

    print("\n" + "=" * 60)
    print("COMPLETE RAG PIPELINE TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()