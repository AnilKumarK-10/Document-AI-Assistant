**Document AI Assistant**

🔎 **Overview**

The Document AI Assistant is a lightweight app built with Streamlit that helps you interact with long PDFs using the Retrieval-Augmented Generation (RAG) framework.
**Retrieve** → Extract & embed document chunks with MiniLM and store in FAISS
**Augment** → Fetch the most relevant chunks when you ask a question
**Generate** → The LLM (BART) answers or summarizes only using retrieved context, ensuring responses are grounded in your PDF with page citations
This avoids hallucinations and gives accurate, page-aware insights from your documents.

 **About the Project**

Problem: Long documents (contracts, reports, manuals) are slow to process.
Solution: Automate summarization and Q&A of the document using RAG + LLM.

**📂 Project Structure**

document-ai-assistant/

├─ embeddings.py # MiniLM embeddings

├─ summarizer.py # BART summarization

├─ pdf_loader.py # PDF text extraction

├─ preprocessor.py # Clean & chunk text

├─ vector_store.py # FAISS vector index

├─ qa.py # Retriever + Q&A

├─ llm.py # LLM wrapper (answers with context)

├─ streamlit_app.py # Streamlit UI

├─ requirements.txt # Dependencies

└─ .env # API keys (not committed)



**Models**

Embeddings → all-MiniLM-L6-v2 (512-token input, 384-dim output)
Summarizer/Generator → facebook/bart-large-cnn


🔄 **Workflow**

PDF Loader (pdf_loader.py) → Extracts text.
Preprocessor → Cleans & chunks into ≤512 tokens.
Embeddings (embeddings.py) → Creates 384-dim vectors with MiniLM.
Vector Store (vector_store.py) → Saves/retrieves chunks using FAISS.
Retriever (qa.py) → Selects most relevant chunks.
LLM (llm.py) → Uses retrieved context to answer with citations or summarize via BART.
Streamlit App → Simple UI to upload PDFs, ask questions, or get summaries.


🛠 **Tech Stack**

Frontend: Streamlit

RAG Components: SentenceTransformers (MiniLM) + FAISS + HuggingFace (BART)

Embeddings: all-MiniLM-L6-v2

LLM / Summarizer: facebook/bart-large-cnn

PDF Parsing: PyPDF2

Python: 3.13


**Acknowledgments**

HuggingFace (MiniLM & BART models)

Streamlit for UI

FAISS for semantic vector search

PyPDF2 for PDF parsing
