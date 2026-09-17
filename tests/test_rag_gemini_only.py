from models.llm import GeminiLLM
from rag.context import build_context
from rag.prompts import SYSTEM_PROMPT, build_rag_prompt
from rag.retriever import Retriever
from rag.vectorstore import FAISSVectorStore


def main():
    print("DocDev AI - RAG Gemini Diagnostic")
   

    print("\n[1] Loading vector store...")
    vector_store = FAISSVectorStore.load("vectorstore")
    print(f"[OK] Vector store loaded: {vector_store.size} chunks")

    print("\n[2] Creating retriever...")
    retriever = Retriever(vector_store=vector_store, top_k=4)
    print("[OK] Retriever created.")

    question = "What is retrieval augmented generation?"

    print("\n[3] Retrieving context...")
    results = retriever.retrieve(question)
    print(f"[OK] Retrieved {len(results)} chunk(s).")

    print("\n[4] Building context...")
    context = build_context(results)
    print(f"[OK] Context built: {len(context)} characters")

    print("\n[5] Building exact RAG prompt...")
    prompt = build_rag_prompt(
        question=question,
        context=context,
    )
    print(f"[OK] Prompt built: {len(prompt)} characters")

    print("\n[6] Creating Gemini LLM...")
    llm = GeminiLLM()
    print("[OK] Gemini LLM created.")

    print("\n[7] Sending exact RAG prompt to Gemini...")

    answer = llm.generate(
        prompt=prompt,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=1024,
    )

   
    print("GEMINI RESPONSE")
   
    print(answer)

    if not answer:
        raise RuntimeError("Gemini returned an empty response.")

   
    print("RAG GEMINI DIAGNOSTIC PASSED!")
    


if __name__ == "__main__":
    main()