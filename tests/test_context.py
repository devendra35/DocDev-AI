from pathlib import Path

from rag.chunker import chunk_documents
from rag.context import (
    build_context,
    build_source_list,
)
from rag.embeddings import embed_chunks
from rag.loader import load_document
from rag.prompts import build_rag_prompt
from rag.retriever import create_retriever
from rag.vectorstore import FAISSVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "uploads"
    / "test.pdf"
)

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "vectorstore"
    / "context_test"
)


def main():
    print("DocDev AI - RAG Context Test")
 

   
    #  Load document
    

    print("\n[1] Loading document...")

    documents = load_document(TEST_FILE)

    print(
        f"[OK] Loaded document units: "
        f"{len(documents)}"
    )

   
    #  Chunk document
   

    print("\n[2] Creating chunks...")

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        chunk_overlap=50,
    )

    print(
        f"[OK] Created chunks: "
        f"{len(chunks)}"
    )

   
    #  Create embeddings
  

    print("\n[3] Creating embeddings...")

    embeddings = embed_chunks(chunks)

    dimension = len(embeddings[0])

    print(
        f"[OK] Embedding dimension: "
        f"{dimension}"
    )

   
    #  Create FAISS store
    

    print("\n[4] Creating vector store...")

    store = FAISSVectorStore(
        dimension=dimension
    )

    store.add(
        chunks,
        embeddings,
    )

    store.save(VECTORSTORE_DIR)

    print(
        f"[OK] Stored vectors: "
        f"{store.size}"
    )

    
    #  Create retriever
    

    print("\n[5] Creating retriever...")

    loaded_store = FAISSVectorStore.load(
        VECTORSTORE_DIR
    )

    retriever = create_retriever(
        loaded_store,
        top_k=4,
    )

    print("[OK] Retriever ready.")

   
    #  Retrieve relevant chunks
    

    print("\n[6] Retrieving relevant context...")

    question = (
        "What technology does DocDev AI use "
        "for document search?"
    )

    results = retriever.retrieve(
        question
    )

    if not results:
        raise RuntimeError(
            "No retrieval results found."
        )

    print(
        f"[OK] Retrieved chunks: "
        f"{len(results)}"
    )

   
    #  Build context
    

    print("\n[7] Building RAG context...")

    context = build_context(results)

    if not context:
        raise RuntimeError(
            "Context generation failed."
        )

    print("[OK] Context created.")

    print("\n--- CONTEXT ---")
    print(context)

  
    #  Build source list
   

    print("\n[8] Building source list...")

    sources = build_source_list(results)

    if not sources:
        raise RuntimeError(
            "Source list is empty."
        )

    print("[OK] Sources created.")

    for source in sources:
        print(source)

    
    #  Build final RAG prompt
    

    print("\n[9] Building RAG prompt...")

    prompt = build_rag_prompt(
        question=question,
        context=context,
    )

    if not prompt:
        raise RuntimeError(
            "RAG prompt is empty."
        )

    print("[OK] RAG prompt created.")

    print("\n--- PROMPT PREVIEW ---")
    print(prompt[:2000])

  
    #  Validation
   

    assert question in prompt

    assert context in prompt

    assert "ONLY" in prompt

    assert "DOCUMENT CONTEXT" in prompt

    assert "USER QUESTION" in prompt

    assert "ANSWER" in prompt

    assert all(
        "source" in source
        for source in sources
    )

   
    print("RAG CONTEXT TEST PASSED!")
    


if __name__ == "__main__":
    main()