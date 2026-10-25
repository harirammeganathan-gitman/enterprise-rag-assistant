"""Streamlit chat and document upload interface."""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
API_ACCESS_KEY = os.environ.get("API_ACCESS_KEY")
if not API_URL.startswith(("http://", "https://")):
    API_URL = f"https://{API_URL}"


def post_to_api(path: str, **kwargs: object) -> requests.Response | None:
    try:
        headers = {}
        if API_ACCESS_KEY:
            headers["X-API-Key"] = API_ACCESS_KEY
        return requests.post(f"{API_URL}{path}", headers=headers, **kwargs)
    except requests.ConnectionError:
        st.error(
            "Cannot connect to the API. Start it in another terminal with "
            "`python -m uvicorn app.main:app --reload`."
        )
    except requests.Timeout:
        st.error("The API request timed out. Please try again.")
    except requests.RequestException as error:
        st.error(f"API request failed: {error}")
    return None


st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="📚")
st.title("Enterprise RAG Assistant")
st.caption("Ask questions grounded in your indexed documents.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Add documents")
    if st.button("Clear chat"):
        st.session_state.chat_history = []
    uploads = st.file_uploader(
        "PDF, Markdown, or DOCX",
        type=["pdf", "md", "markdown", "docx"],
        accept_multiple_files=True,
    )
    if st.button("Ingest documents", disabled=not uploads):
        response = post_to_api(
            "/ingest",
            files=[
                ("files", (upload.name, upload.getvalue(), upload.type))
                for upload in uploads or []
            ],
            timeout=300,
        )
        if response is not None:
            if response.ok:
                st.success(f"Indexed {response.json()['chunks_indexed']} chunks.")
            else:
                st.error(f"Request failed ({response.status_code}): {response.text}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask about your documents")
if query:
    user_message = {"role": "user", "content": query}
    st.session_state.chat_history.append(user_message)
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and preparing an answer..."):
            response = post_to_api("/chat", json={"query": query}, timeout=180)
            if response is not None and response.ok:
                answer = response.json()["answer"]
                st.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
            elif response is not None:
                st.error(f"Request failed ({response.status_code}): {response.text}")
