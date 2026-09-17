from rag.context import build_context
from rag.prompts import build_rag_prompt
from rag.retriever import Retriever
from rag.vectorstore import FAISSVectorStore


def main():
    print("DocDev AI - RAG Prompt Diagnostic")
   


    #  Load vector store
    

    print("\n[1] Loading vector store...")

    vector_store = FAISSVectorStore.load(
        "vectorstore"
    )

    print(
        f"[OK] Vector store loaded: "
        f"{vector_store.size} chunks"
    )

   
    #  Create retriever
   

    print("\n[2] Creating retriever...")

    retriever = Retriever(
        vector_store=vector_store,
        top_k=4,
    )

    print("[OK] Retriever created.")


    #  Retrieve
   

    question = "What is retrieval augmented generation?"

    print("\n[3] Retrieving relevant chunks...")

    results = retriever.retrieve(
        question
    )

    print(
        f"[OK] Retrieved {len(results)} chunk(s)."
    )

    #  Build context
    

    print("\n[4] Building context...")

    context = build_context(
        results
    )

    print(
        f"[OK] Context characters: "
        f"{len(context)}"
    )

    #  Build prompt
   

    print("\n[5] Building RAG prompt...")

    prompt = build_rag_prompt(
        question=question,
        context=context,
    )

    print(
        f"[OK] Prompt characters: "
        f"{len(prompt)}"
    )

    print(
        f"[OK] Prompt words: "
        f"{len(prompt.split())}"
    )

  
    #  Display retrieved content
   

    print("\n" + "=" * 60)
    print("RETRIEVED CONTEXT")
    print("=" * 60)

    print(context)

    
    #  Display prompt
   

    print("\n" + "=" * 60)
    print("RAG PROMPT")
    print("=" * 60)

    print(prompt)

    
    #  Final validation


    if not results:
        raise RuntimeError(
            "No chunks were retrieved."
        )

    if not context:
        raise RuntimeError(
            "Context is empty."
        )

    if not prompt:
        raise RuntimeError(
            "Prompt is empty."
        )

    print("\n" + "=" * 60)
    print("RAG PROMPT DIAGNOSTIC PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()