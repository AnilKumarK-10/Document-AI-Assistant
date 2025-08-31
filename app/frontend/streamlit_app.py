# app/frontend/streamlit_app.py

import os, sys
from pathlib import Path
from typing import List

import streamlit as st


os.environ["TOKENIZERS_PARALLELISM"] = "false"


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


from backend.pdf_loader import load_contract
from backend.preprocessor import split_into_chunks
from backend.embeddings import embed_texts
from backend.vector_store import VectorStore
from backend.llm import ask_llm, summarize_llm

# Streamlit page setup
st.set_page_config(page_title="📄 PDF Summarizer & Q&A", layout="wide")
st.title("📄 PDF Summarizer & Q&A")


MODEL = "gpt-5 nano" 
TOP_K = 5  # how many chunks to fetch for context

# Upload PDF & extract text
uploaded = st.file_uploader("Upload a PDF (≥5 pages)", type=["pdf"])
if not uploaded:
    st.info("Please upload a PDF to get started.")
    st.stop()

if "full_text" not in st.session_state:
    st.session_state.full_text = load_contract(uploaded.read())

if not st.session_state.full_text:
    st.error("❌ Failed to extract any text from this PDF.")
    st.stop()

#  Build or reuse FAISS vector store
if "vs" not in st.session_state:
    chunks = split_into_chunks(st.session_state.full_text, chunk_size=1000, overlap=200)
    embeddings = embed_texts(chunks)
    vs = VectorStore(dim=len(embeddings[0]))
    vs.add(texts=chunks, embeddings=embeddings)
    st.session_state.vs = vs


# Summary section
def _on_summarize():
    text = st.session_state.full_text
    snippet = text[:3000] + ("\n\n…(truncated)…" if len(text) > 3000 else "")
    try:
        # GPT-4 uses max_tokens, not max_completion_tokens
        st.session_state.summary = summarize_llm(
            text=snippet, model=MODEL, max_tokens=300
        )
        st.session_state.summary_error = None
    except Exception as e:
        import traceback

        st.session_state.summary_error = traceback.format_exc()


st.subheader("🔍 Summary")
st.button("📝 Generate Summary", on_click=_on_summarize)

if err := st.session_state.get("summary_error"):
    st.error("❌ Summary error:\n" + err)
elif summary := st.session_state.get("summary"):
    st.markdown("> " + summary)
else:
    st.info("Click “Generate Summary” to get started.")

st.markdown("---")

#  Q&A section
st.subheader("❓ Ask a Question")
question = st.text_input("Enter your question about the PDF:", key="qa_input")

if st.button("🔎 Get Answer") and question:
    # retrieve context
    q_emb = embed_texts([question])[0]
    top_chunks = st.session_state.vs.query(emb=q_emb, top_k=TOP_K)
    context = "\n\n---\n\n".join(top_chunks)

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer using only the above context."
    )
    try:
        answer = ask_llm(
            messages=[{"role": "user", "content": prompt}], model=MODEL, max_tokens=200
        )
        st.session_state.qa_error = None
        st.session_state.answer = answer
    except Exception as e:
        import traceback

        st.session_state.qa_error = traceback.format_exc()

if err := st.session_state.get("qa_error"):
    st.error("❌ Q&A error:\n" + err)
elif ans := st.session_state.get("answer"):
    st.markdown("**Answer:**  " + ans)
