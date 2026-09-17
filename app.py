from __future__ import annotations

from pathlib import Path

import streamlit as st

from analysis.analyzer import DocumentAnalyzer
from analysis.overview import generate_overview
from rag.document_manager import DocumentManager
from rag.ingestion import ingest_document
from rag.retriever import Retriever
from rag.vectorstore import FAISSVectorStore
from rag.context import build_context, build_source_list
from rag.prompts import build_rag_prompt
from models.llm import GeminiLLM


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "DocDev AI"
APP_TAGLINE = "Turn Documents Into Knowledge."

UPLOAD_DIR = Path("data/uploads")
VECTORSTORE_DIR = Path("vectorstore")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource
def get_analyzer():
    return DocumentAnalyzer()


@st.cache_resource
def get_llm():
    return GeminiLLM()


def get_manager():
    return DocumentManager(
        VECTORSTORE_DIR
    )


def get_selected_path():
    filename = st.session_state.selected_file

    if not filename:
        return None

    path = UPLOAD_DIR / filename

    if not path.exists():
        return None

    return path


def run_analysis(file_path):
    analyzer = get_analyzer()

    with st.spinner(
        "Analyzing document with AI..."
    ):
        result = analyzer.analyze(
            file_path
        )

    st.session_state.analysis = result


def format_sources(sources):
    if not sources:
        return "No sources found."

    lines = []

    for index, source in enumerate(
        sources,
        start=1,
    ):
        filename = source["source"]
        page = source["page"]
        score = source["score"]

        if page is not None:
            location = f"{filename}, Page {page}"
        else:
            location = filename

        lines.append(
            f"**[{index}] {location}** "
            f"(score: {score:.3f})"
        )

    return "\n\n".join(lines)


# ============================================================
# HEADER
# ============================================================

st.title(APP_TITLE)

st.markdown(
    f"### {APP_TAGLINE}"
)

st.caption(
    "AI-powered document intelligence and grounded knowledge assistant."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "docx",
            "txt",
        ],
        accept_multiple_files=True,
        help="Upload PDF, DOCX or TXT documents.",
    )

    if uploaded_files:

        st.subheader(
            "Selected files"
        )

        for uploaded_file in uploaded_files:

            st.write(
                f"📄 {uploaded_file.name}"
            )

        if st.button(
            "🚀 Process Documents",
            use_container_width=True,
        ):

            for uploaded_file in uploaded_files:

                destination = (
                    UPLOAD_DIR
                    / uploaded_file.name
                )

                with destination.open(
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

                try:

                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):

                        result = ingest_document(
                            destination,
                            VECTORSTORE_DIR,
                        )

                    if result["status"] == "skipped":

                        st.warning(
                            f"{uploaded_file.name} "
                            "already exists."
                        )

                    else:

                        st.success(
                            f"{uploaded_file.name} "
                            "processed successfully."
                        )

                except Exception as exc:

                    st.error(
                        f"Failed to process "
                        f"{uploaded_file.name}: {exc}"
                    )

            st.rerun()

    st.divider()

    st.header("📚 Document Library")

    manager = get_manager()

    documents = manager.list_documents()

    if not documents:

        st.info(
            "No documents available."
        )

    else:

        for document in documents:

            filename = document[
                "filename"
            ]

            if st.button(
                f"📄 {filename}",
                key=f"doc_{document['hash']}",
                use_container_width=True,
            ):

                st.session_state.selected_file = (
                    filename
                )

                st.session_state.analysis = None

                st.session_state.chat_messages = []

                st.rerun()

    st.divider()

    stats = manager.statistics()

    st.header("📊 Statistics")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Documents",
            stats["documents"],
        )

    with col2:

        st.metric(
            "Chunks",
            stats["vector_store_chunks"],
        )

    if stats["status"] == "Ready":

        st.success(
            "Vector store ready"
        )

    else:

        st.warning(
            "Vector store not ready"
        )


# ============================================================
# SELECTED DOCUMENT
# ============================================================

selected_path = get_selected_path()

if selected_path is None:

    st.info(
        "👈 Upload and process a document, "
        "then select it from the Document Library."
    )

    st.markdown(
        """
        ## What DocDev AI can do

        **📄 Document Intelligence**
        - Overview
        - Summary
        - Important information
        - Knowledge extraction
        - Topic detection
        - Term extraction
        - Entity extraction
        - Automatic Q&A
        - Links and references

        **💬 Grounded RAG Chat**
        - Ask questions about your documents
        - Retrieve relevant passages
        - Generate grounded answers
        - Show source and page information

        **🔎 Multi-document Knowledge**
        - Upload multiple documents
        - Search across documents
        - Compare information
        - Avoid duplicate documents
        """
    )

    st.stop()


# ============================================================
# DOCUMENT HEADER
# ============================================================

st.subheader(
    f"📄 {selected_path.name}"
)

overview = generate_overview(
    selected_path
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Words",
        overview["words"],
    )

with col2:

    st.metric(
        "Characters",
        overview["characters"],
    )

with col3:

    st.metric(
        "Pages",
        overview["pages"],
    )

with col4:

    st.metric(
        "Type",
        overview["file_type"].upper(),
    )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Overview",
        "Summary",
        "Important",
        "Knowledge",
        "Topics",
        "Terms",
        "Entities",
        "Questions",
        "Links",
        "💬 Chat",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.header(
        "Document Overview"
    )

    st.write(
        f"**Filename:** "
        f"{overview['filename']}"
    )

    st.write(
        f"**File type:** "
        f"{overview['file_type']}"
    )

    st.write(
        f"**Words:** "
        f"{overview['words']:,}"
    )

    st.write(
        f"**Characters:** "
        f"{overview['characters']:,}"
    )

    st.write(
        f"**Pages:** "
        f"{overview['pages']}"
    )


# ============================================================
# LOAD AI ANALYSIS
# ============================================================

def get_analysis():

    if st.session_state.analysis is None:

        run_analysis(
            selected_path
        )

    return st.session_state.analysis


# ============================================================
# SUMMARY
# ============================================================

with tabs[1]:

    st.header("Summary")

    analysis = get_analysis()

    st.write(
        analysis["summary"]
    )


# ============================================================
# IMPORTANT
# ============================================================

with tabs[2]:

    st.header(
        "Important Information"
    )

    analysis = get_analysis()

    for point in analysis[
        "important_points"
    ]:

        st.markdown(
            f"- {point}"
        )


# ============================================================
# KNOWLEDGE
# ============================================================

with tabs[3]:

    st.header(
        "Knowledge & Concepts"
    )

    analysis = get_analysis()

    for item in analysis[
        "knowledge"
    ]:

        with st.expander(
            item["concept"]
        ):

            st.write(
                item["explanation"]
            )


# ============================================================
# TOPICS
# ============================================================

with tabs[4]:

    st.header(
        "Topics"
    )

    analysis = get_analysis()

    for topic in analysis[
        "topics"
    ]:

        st.markdown(
            f"• **{topic}**"
        )


# ============================================================
# TERMS
# ============================================================

with tabs[5]:

    st.header(
        "Important Terms"
    )

    analysis = get_analysis()

    for item in analysis[
        "terms"
    ]:

        st.markdown(
            f"**{item['term']}**"
        )

        st.write(
            item["definition"]
        )

        st.divider()


# ============================================================
# ENTITIES
# ============================================================

with tabs[6]:

    st.header(
        "Entities"
    )

    analysis = get_analysis()

    if analysis["entities"]:

        for entity in analysis[
            "entities"
        ]:

            st.markdown(
                f"**{entity['name']}** "
                f"— {entity['type']}"
            )

    else:

        st.info(
            "No important entities found."
        )


# ============================================================
# QUESTIONS
# ============================================================

with tabs[7]:

    st.header(
        "Generated Questions & Answers"
    )

    analysis = get_analysis()

    for item in analysis[
        "questions"
    ]:

        with st.expander(
            item["question"]
        ):

            st.write(
                item["answer"]
            )


# ============================================================
# LINKS
# ============================================================

with tabs[8]:

    st.header(
        "Links & References"
    )

    analysis = get_analysis()

    if analysis["links"]:

        for item in analysis[
            "links"
        ]:

            st.markdown(
                f"🔗 {item['url']}"
            )

            if item.get(
                "context"
            ):

                st.caption(
                    item["context"]
                )

    else:

        st.info(
            "No links were found in this document."
        )


# ============================================================
# CHAT
# ============================================================

with tabs[9]:

    st.header(
        "💬 Ask Your Document"
    )

    st.caption(
        "Answers are generated from retrieved document context."
    )

    for message in st.session_state.chat_messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if message.get(
                "sources"
            ):

                with st.expander(
                    "📚 Sources"
                ):

                    st.markdown(
                        format_sources(
                            message["sources"]
                        )
                    )

    question = st.chat_input(
        "Ask a question about this document..."
    )

    if question:

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        try:

            vector_store = (
                FAISSVectorStore.load(
                    VECTORSTORE_DIR
                )
            )

            retriever = Retriever(
                vector_store,
                top_k=4,
            )

            results = retriever.retrieve(
                question
            )

            context = build_context(
                results
            )

            if not context:

                answer = (
                    "I could not find "
                    "relevant information "
                    "in the document."
                )

                sources = []

            else:

                prompt = build_rag_prompt(
                    question,
                    context,
                )

                llm = get_llm()

                answer = llm.generate(
                    prompt,
                    temperature=0.2,
                    max_output_tokens=1024,
                )

                sources = (
                    build_source_list(
                        results
                    )
                )

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

            st.rerun()

        except Exception as exc:

            st.error(
                f"Unable to answer question: {exc}"
            )