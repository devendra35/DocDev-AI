from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving
    meaningful paragraphs and sentence structure.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove excessive spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()