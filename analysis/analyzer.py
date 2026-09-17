
from __future__ import annotations

import json
from pathlib import Path

from rag.loader import load_document
from models.llm import GeminiLLM


class AnalysisError(Exception):
    """Raised when document analysis fails."""


class DocumentAnalyzer:
    """Generate complete document intelligence using structured JSON."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
    ):
        self.llm = GeminiLLM(
            model_name=model_name
        )
        self._cache: dict[str, dict] = {}

    def _load_text(
        self,
        file_path: str | Path,
    ) -> str:

        file_path = Path(file_path)

        if not file_path.exists():
            raise AnalysisError(
                f"File not found: {file_path}"
            )

        try:
            documents = load_document(file_path)

        except Exception as exc:
            raise AnalysisError(
                f"Failed to load document: {exc}"
            ) from exc

        if not documents:
            raise AnalysisError(
                "No readable content found."
            )

        parts = []

        for document in documents:

            if document.page is not None:
                parts.append(
                    f"[Page {document.page}]\n"
                    f"{document.text}"
                )
            else:
                parts.append(document.text)

        text = "\n\n".join(parts)

        if not text.strip():
            raise AnalysisError(
                "Document contains no readable text."
            )

        return text

    def _parse_json(self, response: str) -> dict:

        response = response.strip()

        if not response:
            raise AnalysisError(
                "Gemini returned an empty response."
            )

        try:
            data = json.loads(response)

        except json.JSONDecodeError as exc:

            # Helpful debugging information
            preview = response[:500]

            raise AnalysisError(
                "Gemini returned invalid JSON.\n\n"
                f"Response preview:\n{preview}"
            ) from exc

        if not isinstance(data, dict):
            raise AnalysisError(
                "Gemini returned an invalid analysis object."
            )

        return data

    def analyze(
        self,
        file_path: str | Path,
    ) -> dict:

        file_path = Path(file_path)

        if not file_path.exists():
            raise AnalysisError(
                f"File not found: {file_path}"
            )

        # Cache based on file modification time
        cache_key = (
            f"{file_path.resolve()}:"
            f"{file_path.stat().st_mtime_ns}"
        )

        if cache_key in self._cache:
            return self._cache[cache_key]

        text = self._load_text(file_path)

        prompt = f"""
You are DocDev AI, a professional document
intelligence system.

Analyze ONLY the document provided below.

IMPORTANT RULES:

1. Use ONLY information contained in the document.
2. Never invent information.
3. Never use outside knowledge.
4. Keep every answer concise.
5. Preserve important facts, numbers, dates and names.
6. If a category has no information, return an empty array.
7. Page numbers may only come from [Page X] markers.
8. Return the requested structured JSON data.

DOCUMENT

{text}

Analyze the document and extract:

- A concise summary
- Important points
- Important concepts and explanations
- Main topics
- Important terms and definitions
- Important entities
- Useful questions and answers
- Links mentioned in the document
"""

        try:

            result = self.llm.generate_structured(
                prompt=prompt,
                temperature=0.1,
                max_output_tokens=5000,
            )

        except Exception as exc:

            raise AnalysisError(
                f"Gemini analysis failed: {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise AnalysisError(
                "Gemini returned an invalid analysis object."
            )

        defaults = {
            "summary": "",
            "important_points": [],
            "knowledge": [],
            "topics": [],
            "terms": [],
            "entities": [],
            "questions": [],
            "links": [],
        }

        for key, default in defaults.items():

            if key not in result:
                result[key] = default

        self._cache[cache_key] = result

        return result

    def summary(
        self,
        file_path: str | Path,
    ) -> str:

        return self.analyze(file_path)["summary"]

    def important_points(
        self,
        file_path: str | Path,
    ) -> list:

        return self.analyze(file_path)["important_points"]

    def knowledge(
        self,
        file_path: str | Path,
    ) -> list:

        return self.analyze(file_path)["knowledge"]

    def topics(
        self,
        file_path: str | Path,
    ) -> list:

        return self.analyze(file_path)["topics"]

    def terms(
        self,
        file_path: str | Path,
    ) -> list:

        return self.analyze(file_path)["terms"]

    def entities(
        self,
        file_path: str | Path,
    ) -> list:

        return self.analyze(file_path)["entities"]

    def questions(
        self,
        file_path: str | Path,
    ) -> list:

        return self.analyze(file_path)["questions"]

    def links(
        self,
        file_path: str | Path,
    ) -> list:

        return self.analyze(file_path)["links"]

