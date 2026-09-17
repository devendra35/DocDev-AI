from __future__ import annotations

from pathlib import Path

from rag.loader import load_document


class OverviewError(Exception):
    """Raised when document overview generation fails."""


def generate_overview(
    file_path: str | Path,
) -> dict:
    """
    Generate basic document overview information.

    This function extracts factual metadata from the
    document without using the LLM.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise OverviewError(
            f"File not found: {file_path}"
        )

    try:
        documents = load_document(
            file_path
        )

        if not documents:
            raise OverviewError(
                "No content could be extracted."
            )

        full_text = "\n".join(
            document.text
            for document in documents
        )

        words = full_text.split()

        pages = {
            document.page
            for document in documents
            if document.page is not None
        }

        return {
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
            "documents": len(documents),
            "pages": len(pages),
            "characters": len(full_text),
            "words": len(words),
            "empty": not bool(full_text.strip()),
        }

    except OverviewError:
        raise

    except Exception as exc:
        raise OverviewError(
            f"Failed to generate overview: {exc}"
        ) from exc