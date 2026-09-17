from __future__ import annotations


SYSTEM_PROMPT = """
You are DocDev AI, a document intelligence assistant.

Your job is to answer questions using ONLY the
provided document context.

Rules:

1. Use the provided context as the primary source of truth.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain enough information,
   clearly say that the information is not available
   in the provided documents.
4. Do not use outside knowledge to fill missing information.
5. Keep answers clear, accurate, and concise.
6. When possible, mention the source and page that
   support the answer.
"""


def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build the final prompt sent to the LLM.
    """

    question = question.strip()
    context = context.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not context:
        raise ValueError(
            "Context cannot be empty."
        )

    return f"""
{SYSTEM_PROMPT}

DOCUMENT CONTEXT

{context}

USER QUESTION

{question}

ANSWER

Answer the user's question using only the
document context above.
""".strip()