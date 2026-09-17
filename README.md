#  DocDev AI



### Turn Documents Into Knowledge.



DocDev AI is an **AI-powered document intelligence and knowledge assistant** that transforms documents into structured, searchable, and interactive knowledge.



Instead of simply allowing users to chat with a document, DocDev AI analyzes uploaded documents, extracts important information, identifies concepts and topics, generates questions and answers, and provides **grounded RAG-based conversations with document sources**.







##  Features



###  Document Intelligence



Upload and analyze documents such as:



* PDF

* DOCX

* TXT



DocDev AI automatically extracts and analyzes the content.



###  AI-Powered Analysis



For each document, DocDev AI can generate:



*  Document Summary

*  Important Information

*  Knowledge \& Concepts

*  Topics

*  Important Terms \& Definitions

*  Entities

*  Automatically Generated Questions \& Answers

*  Links \& References



###  Grounded RAG Chat



Ask questions about your documents using Retrieval-Augmented Generation.



The system:



```text

Question

&#x20;  ↓

Query Embedding

&#x20;  ↓

FAISS Vector Search

&#x20;  ↓

Relevant Document Chunks

&#x20;  ↓

Context Construction

&#x20;  ↓

Gemini

&#x20;  ↓

Grounded Answer

&#x20;  ↓

Sources

```



The assistant is instructed to answer using the retrieved document context and avoid inventing information.



###  Multi-Document Knowledge



Upload multiple documents and build a shared knowledge base.



DocDev AI supports:



* Multiple document ingestion

* Cross-document retrieval

* Duplicate document detection

* Document management

* Vector-store rebuilding

* Source tracking



###  Semantic Search



Documents are converted into vector embeddings using:



`sentence-transformers/all-MiniLM-L6-v2`



FAISS is then used for efficient similarity search.



### Hallucination Control



DocDev AI is designed to reduce unsupported answers by:



* Restricting RAG answers to retrieved document context

* Using source metadata
* Displaying retrieved sources

* Instructing the LLM not to invent information

* Using structured JSON output for document analysis







# 🏗️ Architecture



```text

&#x20;                   ┌─────────────────────┐

&#x20;                   │       User          │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │    Streamlit UI     │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;               ┌──────────────┴──────────────┐

&#x20;               │                             │

&#x20;               ▼                             ▼

&#x20;      ┌─────────────────┐           ┌─────────────────┐

&#x20;      │ Document Upload │           │   RAG Chat      │

&#x20;      └────────┬────────┘           └────────┬────────┘

&#x20;               │                             │

&#x20;               ▼                             ▼

&#x20;      ┌─────────────────┐           ┌─────────────────┐

&#x20;      │ Document Loader │           │ Query Embedding │

&#x20;      └────────┬────────┘           └────────┬────────┘

&#x20;               │                             │

&#x20;               ▼                             ▼

&#x20;      ┌─────────────────┐           ┌─────────────────┐

&#x20;      │ Cleaning        │           │  FAISS Search   │

&#x20;      │ \& Chunking      │           └────────┬────────┘

&#x20;      └────────┬────────┘                    │

&#x20;               │                             ▼

&#x20;               ▼                    ┌─────────────────┐

&#x20;      ┌─────────────────┐           │ Relevant Chunks │

&#x20;      │    Embeddings   │           └────────┬────────┘

&#x20;      └────────┬────────┘                    │

&#x20;               │                             ▼

&#x20;               ▼                    ┌─────────────────┐

&#x20;      ┌─────────────────┐           │ Context Builder │

&#x20;      │      FAISS      │           └────────┬────────┘

&#x20;      │  Vector Store   │                    │

&#x20;      └─────────────────┘                    ▼

&#x20;                                    ┌─────────────────┐

&#x20;                                    │ Gemini 2.5 Flash│

&#x20;                                    └────────┬────────┘

&#x20;                                             │

&#x20;                                             ▼

&#x20;                                    ┌─────────────────┐

&#x20;                                    │ Grounded Answer │

&#x20;                                    │ + Sources       │

&#x20;                                    └─────────────────┘

```



\---



\# 🧠 Document Analysis Pipeline



DocDev AI uses a separate document-intelligence pipeline for extracting structured knowledge.



```text

Document

&#x20;  ↓

Text Extraction

&#x20;  ↓

Document Cleaning

&#x20;  ↓

Gemini Analysis

&#x20;  ↓

Structured JSON

&#x20;  ↓

┌─────────────────────────────┐

│ Summary                     │

│ Important Information       │

│ Knowledge \& Concepts        │

│ Topics                      │

│ Terms                       │

│ Entities                    │

│ Questions \& Answers         │

│ Links                       │

└─────────────────────────────┘

```



Gemini's structured JSON output is used to make the analysis more reliable and easier for the application to process.







# 🛠️ Technology Stack



## Frontend / UI



* Streamlit



## AI / LLM



* Google Gemini

* Gemini 2.5 Flash

* Google GenAI SDK



## RAG



\* LangChain ecosystem

* Retrieval-Augmented Generation

* FAISS

* Sentence Transformers



## Embeddings



```text

sentence-transformers/all-MiniLM-L6-v2

```



## Document Processing



* PyMuPDF

* python-docx

* Python text processing



## Configuration



* python-dotenv

* Pydantic Settings



\## Language



\* Python 3.12+



\---



\# 📁 Project Structure



```text

DocDev-AI/

│

├── app.py

├── config.py

├── requirements.txt

├── README.md

├── .env

├── .env.example

├── .gitignore

│

├── data/

│   └── uploads/

│

├── vectorstore/

│

├── rag/

│   ├── \_\_init\_\_.py

│   ├── loader.py

│   ├── cleaner.py

│   ├── chunker.py

│   ├── embeddings.py

│   ├── vectorstore.py

│   ├── retriever.py

│   ├── prompts.py

│   ├── context.py

│   ├── chain.py

│   ├── memory.py

│   ├── ingestion.py

│   ├── document\_registry.py

│   └── document\_manager.py

│

├── analysis/

│   ├── \_\_init\_\_.py

│   ├── overview.py

│   └── analyzer.py

│

├── models/

│   ├── \_\_init\_\_.py

│   └── llm.py

│

├── utils/

│   ├── \_\_init\_\_.py

│   ├── file\_utils.py

│   └── text\_utils.py

│

└── tests/

&#x20;   ├── \_\_init\_\_.py

&#x20;   ├── test\_loader.py

&#x20;   ├── test\_chunker.py

&#x20;   ├── test\_retriever.py

&#x20;   ├── test\_rag.py

&#x20;   ├── test\_overview.py

&#x20;   └── test\_full\_analysis.py

```



\---



\# ⚙️ Installation



\## 1. Clone the Repository



```bash

git clone https://github.com/devendra35/DocDev-AI.git

cd DocDev-AI

```



\## 2. Create a Virtual Environment



\### Windows



```powershell

python -m venv .venv

```



Activate it:



```powershell

.venv\\Scripts\\Activate.ps1

```



\### Linux / macOS



```bash

python3 -m venv .venv

source .venv/bin/activate

```



\---



\# 📦 Install Dependencies



```bash

pip install -r requirements.txt

```



If you are installing manually:



```bash

pip install langchain langchain-community langchain-text-splitters

pip install pymupdf python-docx

pip install sentence-transformers faiss-cpu

pip install streamlit python-dotenv pydantic-settings

pip install -U google-genai

```



\---



\# 🔑 Environment Variables



Create a `.env` file in the project root.



```env

GEMINI\_API\_KEY=your\_gemini\_api\_key



CHUNK\_SIZE=800

CHUNK\_OVERLAP=150

RETRIEVAL\_TOP\_K=4



EMBEDDING\_MODEL\_NAME=sentence-transformers/all-MiniLM-L6-v2



MAX\_FILE\_SIZE\_MB=50

```



\### Important



Never commit your `.env` file.



The project already includes `.env` in `.gitignore`.



\---



\# ▶️ Run the Application



Start DocDev AI with:



```bash

python -m streamlit run app.py

```



The application will open in your browser.



Default local address:



```text

http://localhost:8501

```



\---



\# 📖 How to Use



\### 1. Upload a Document



From the sidebar, upload:



```text

PDF

DOCX

TXT

```



\### 2. Process the Document



Click:



```text

🚀 Process Documents

```



DocDev AI will:



1\. Validate the document

2\. Extract text

3\. Clean the text

4\. Split it into chunks

5\. Generate embeddings

6\. Store embeddings in FAISS

7\. Register the document

8\. Make it available for retrieval



\### 3. Select a Document



Choose a document from the \*\*Document Library\*\*.



\### 4. Explore Document Intelligence



Use the tabs:



```text

Overview

Summary

Important

Knowledge

Topics

Terms

Entities

Questions

Links

Chat

```



\### 5. Ask Questions



Open:



```text

💬 Chat

```



Ask questions about the uploaded documents.



For example:



```text

What is this document about?

```



```text

What technologies are mentioned?

```



```text

What are the main features?

```



```text

What are the important concepts?

```



The system retrieves relevant document chunks before generating an answer.



\---



\# 🔎 RAG Configuration



Default configuration:



| Setting           | Value            |

| ----------------- | ---------------- |

| Chunk Size        | 800              |

| Chunk Overlap     | 150              |

| Retrieval Top-K   | 4                |

| Embedding Model   | all-MiniLM-L6-v2 |

| Vector Database   | FAISS            |

| LLM               | Gemini 2.5 Flash |

| Maximum File Size | 50 MB            |



\---



\# 🧪 Testing



The project contains tests for major components.



Run:



```bash

python -m pytest

```



Tests cover areas including:



\* Document loading

\* Text chunking

\* Embeddings

\* Retrieval

\* RAG pipeline

\* Document overview

\* Complete document analysis



\---



\# 🛡️ Security



DocDev AI follows several basic security practices:



\* API keys stored in `.env`

\* `.env` excluded from Git

\* Uploaded files excluded from Git

\* Vector-store files excluded from Git

\* File type validation

\* File size validation

\* Duplicate document detection



Never expose your Gemini API key publicly.



\---



\# 🔮 Future Improvements



Planned improvements include:



\### 📄 More File Formats



\* CSV

\* XLSX

\* PPTX

\* Markdown

\* HTML



\### 🖼️ Advanced Document Understanding



\* OCR

\* Tables

\* Charts

\* Diagrams

\* Images



\### 🤖 AI Features



\* Document comparison

\* Cross-document reasoning

\* Better citation tracking

\* Automatic document classification

\* Advanced semantic search

\* Follow-up question generation

\* Knowledge graphs



\### 🎙️ Multimodal Features



\* Audio summaries

\* Text-to-speech

\* Voice-based document questions



\### 📤 Export



\* PDF reports

\* Markdown

\* DOCX

\* JSON

\* CSV



\---



\# 🎯 Project Goals



DocDev AI aims to move beyond the traditional:



> "Chat with your PDF"



concept.



The goal is to build a complete \*\*document intelligence platform\*\* that can transform unstructured documents into:



```text

Documents

&#x20;   ↓

Information

&#x20;   ↓

Knowledge

&#x20;   ↓

Search

&#x20;   ↓

Questions \& Answers

&#x20;   ↓

Grounded AI Conversations

```



\---



\# 👨‍💻 Author



\*\*Devendra Khanal\*\*



BSc. CSIT Student



Nepal



GitHub:

https://github.com/devendra35



\---



\# 📄 License



This project is intended for educational and development purposes.



Add an appropriate open-source license to the repository if you plan to distribute or reuse the project publicly.



\---



\## ⭐ If you find this project useful



Give the repository a ⭐ on GitHub and feel free to explore, improve, and contribute to the project.



