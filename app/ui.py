"""Streamlit chat and document upload interface."""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
if not API_URL.startswith(("http://", "https://")):
    API_URL = f"https://{API_URL}"

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="📚")
st.title("Enterprise RAG Assistant")
st.caption("Ask questions grounded in your indexed documents.")

with st.sidebar:
    st.header("Add documents")
    uploads = st.file_uploader(
        "PDF, Markdown, or DOCX",
        type=["pdf", "md", "markdown", "docx"],
        accept_multiple_files=True,
    )
    if st.button("Ingest documents", disabled=not uploads):
        response = requests.post(
            f"{API_URL}/ingest",
            files=[
                ("files", (upload.name, upload.getvalue(), upload.type))
                for upload in uploads or []
            ],
            timeout=300,
        )
        response.raise_for_status()
        st.success(f"Indexed {response.json()['chunks_indexed']} chunks.")

query = st.chat_input("Ask about your documents")
if query:
    with st.spinner("Searching documents and preparing an answer..."):
        response = requests.post(f"{API_URL}/chat", json={"query": query}, timeout=180)
        if response.ok:
            result = response.json()
            st.markdown(result["answer"])
            for citation in result["citations"]:
                label = f"{citation['source']} — chunk {citation['chunk_index']}"
                if citation["page_number"]:
                    label += f", page {citation['page_number']}"
                with st.expander(label):
                    st.metric("Rerank score", f"{citation['score']:.3f}")
            if result["cached"]:
                st.caption("Served from the semantic cache.")
        else:
            st.error(f"Request failed ({response.status_code}): {response.text}")
