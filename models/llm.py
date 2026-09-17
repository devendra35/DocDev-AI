
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"


class LLMError(Exception):
    """Base exception for LLM errors."""


class GeminiLLM:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise LLMError(
                "GEMINI_API_KEY was not found. "
                "Check your .env file."
            )

        self.model_name = model_name

        self.client = genai.Client(
            api_key=api_key
        )

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> str:

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        try:

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            if system_instruction:
                config.system_instruction = system_instruction

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            if not response.text:
                raise LLMError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except Exception as exc:

            raise LLMError(
                f"Gemini generation failed: {exc}"
            ) from exc

    def generate_structured(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_output_tokens: int = 5000,
    ) -> dict:

        """
        Generate valid structured JSON using Gemini.
        """

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        response_schema = {
            "type": "object",

            "properties": {

                "summary": {
                    "type": "string"
                },

                "important_points": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },

                "knowledge": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "concept": {
                                "type": "string"
                            },
                            "explanation": {
                                "type": "string"
                            },
                        },
                        "required": [
                            "concept",
                            "explanation",
                        ],
                    },
                },

                "topics": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },

                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "term": {
                                "type": "string"
                            },
                            "definition": {
                                "type": "string"
                            },
                        },
                        "required": [
                            "term",
                            "definition",
                        ],
                    },
                },

                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string"
                            },
                            "type": {
                                "type": "string"
                            },
                        },
                        "required": [
                            "name",
                            "type",
                        ],
                    },
                },

                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string"
                            },
                            "answer": {
                                "type": "string"
                            },
                        },
                        "required": [
                            "question",
                            "answer",
                        ],
                    },
                },

                "links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string"
                            },
                            "context": {
                                "type": "string"
                            },
                        },
                        "required": [
                            "url",
                            "context",
                        ],
                    },
                },
            },

            "required": [
                "summary",
                "important_points",
                "knowledge",
                "topics",
                "terms",
                "entities",
                "questions",
                "links",
            ],
        }

        try:

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
                response_schema=response_schema,
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            if not response.text:
                raise LLMError(
                    "Gemini returned an empty structured response."
                )

            return json.loads(response.text)

        except Exception as exc:

            if isinstance(exc, LLMError):
                raise

            raise LLMError(
                f"Gemini structured generation failed: {exc}"
            ) from exc