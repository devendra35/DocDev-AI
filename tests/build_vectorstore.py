from rag.loader import load_document
from rag.chunker import chunk_documents
from rag.embeddings import embed_chunks
from rag.vectorstore import FAISSVectorStore


def main():
    print("DocDev AI - Build Vector Store")
   

   
    # Load document
   

    print("\n[1] Loading document...")

    documents = load_document(
        "data/uploads/test.pdf"
    )

    print(
        f"[OK] Loaded {len(documents)} document unit(s)."
    )

  
    #  Create chunks
   

    print("\n[2] Creating chunks...")

    chunks = chunk_documents(
        documents
    )

    print(
        f"[OK] Created {len(chunks)} chunks."
    )

    #  Create embeddings
 

    print("\n[3] Creating embeddings...")

    embeddings = embed_chunks(
        chunks
    )

    print(
        f"[OK] Created {len(embeddings)} embeddings."
    )

    #  Create FAISS vector store
   

    print("\n[4] Creating FAISS vector store...")

    dimension = len(embeddings[0])

    vector_store = FAISSVectorStore(
        dimension=dimension
    )

    vector_store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(
        f"[OK] Added {vector_store.size} chunks."
    )

    #  Save vector store
  

    print("\n[5] Saving vector store...")

    vector_store.save(
        "vectorstore"
    )

    print("[OK] Vector store saved.")

   
    #  Final result
    

   
    print("VECTOR STORE BUILD PASSED!")
    


if __name__ == "__main__":
    main()