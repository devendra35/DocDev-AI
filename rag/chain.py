from __future__ import annotations

from dataclasses import dataclass

from models.llm import GeminiLLM
from rag.context import build_context, build_source_list
from rag.prompts import SYSTEM_PROMPT, build_rag_prompt
from rag.retriever import RetrievedChunk, Retriever


@dataclass(frozen=True)
class RAGResponse:
    """
    Represents a complete RAG response.
    """

    answer: str
    sources: list[dict]
    retrieved_chunks: list[RetrievedChunk]


class RAGChain:
    """
    Complete retrieval-augmented generation pipeline.

    Flow:

    Question
        ↓
    Retriever
        ↓
    Relevant chunks
        ↓
    Context
        ↓
    Prompt
        ↓
    Gemini
        ↓
    Answer
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: GeminiLLM,
    ):
        self.retriever = retriever
        self.llm = llm

    def invoke(
        self,
        question: str,
    ) -> RAGResponse:
        """
        Answer a question using retrieved document context.
        """

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        # Retrieve relevant chunks
      

        results = self.retriever.retrieve(
            question
        )

        if not results:
            return RAGResponse(
                answer=(
                    "I could not find relevant "
                    "information in the provided "
                    "documents."
                ),
                sources=[],
                retrieved_chunks=[],
            )

    
        # Build document context
     

        context = build_context(
            results
        )

        # Build grounded prompt
       

        prompt = build_rag_prompt(
            question=question,
            context=context,
        )

   
        #  Generate answer
       

        answer = self.llm.generate(
            prompt=prompt,
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=1024,
        )

       
        # Build source information
       

        sources = build_source_list(
            results
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=results,
        )