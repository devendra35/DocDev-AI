from pathlib import Path

import pymupdf
from docx import Document


OUTPUT_DIR = Path("data/uploads")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TEST_TEXT = """
DocDev AI is an AI-powered document intelligence application.

It can analyze documents, extract important information, summarize content,
identify important topics, and answer questions using retrieval augmented generation.

The system uses document chunking, embeddings, FAISS vector search,
and a language model.

This document is used to test PDF and DOCX document loading.
"""


def create_pdf():
    pdf_path = OUTPUT_DIR / "test.pdf"

    pdf = pymupdf.open()

    page = pdf.new_page()

    page.insert_textbox(
        (50, 50, 550, 750),
        TEST_TEXT,
        fontsize=12,
    )

    pdf.save(pdf_path)
    pdf.close()

    print(f"[OK] Created {pdf_path}")


def create_docx():
    docx_path = OUTPUT_DIR / "test.docx"

    document = Document()

    for paragraph in TEST_TEXT.strip().split("\n\n"):
        document.add_paragraph(paragraph)

    document.save(docx_path)

    print(f"[OK] Created {docx_path}")


def main():
    print("Creating DocDev AI test documents...")
  
    create_pdf()
    create_docx()

    print("\nTest documents created successfully!")


if __name__ == "__main__":
    main()