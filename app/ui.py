"""Streamlit chat and document upload interface."""

import os
import time

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
API_ACCESS_KEY = os.environ.get("API_ACCESS_KEY")
MAX_SESSION_MESSAGES = 40
MAX_API_HISTORY_MESSAGES = 8
if not API_URL.startswith(("http://", "https://")):
    API_URL = f"https://{API_URL}"

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    :root {
        --ink: #f2f5ff;
        --muted: #aab6d3;
        --teal: #59e1d2;
        --violet: #a78bfa;
        --line: rgba(174, 194, 242, .16);
    }
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        color: var(--ink);
        background-color: #080d1b !important;
        background:
            radial-gradient(
                ellipse at 12% 0%, rgba(52, 76, 145, .28), transparent 36rem
            ),
            radial-gradient(
                ellipse at 90% 16%, rgba(73, 45, 131, .24), transparent 34rem
            ),
            linear-gradient(
                145deg, #080d1b 0%, #0b1224 48%, #101329 100%
            );
    }
    [data-testid="stHeader"] { background: rgba(8, 13, 27, .42); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101a31 0%, #101326 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 2.8rem;
        padding-bottom: 4rem;
        background: transparent !important;
    }
    h1, h2, h3 {
        color: var(--ink) !important;
        letter-spacing: -.035em;
        line-height: 1.18 !important;
    }
    h1 {
        font-size: clamp(2.15rem, 4vw, 3.25rem) !important;
        font-weight: 760 !important;
    }
    h2 {
        font-size: clamp(1.45rem, 2.4vw, 2rem) !important;
        font-weight: 700 !important;
    }
    h3 { font-size: 1.15rem !important; font-weight: 680 !important; }
    p, label, [data-testid="stMetricLabel"] {
        font-size: 1rem;
        line-height: 1.65;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(
            140deg, rgba(30, 43, 76, .9), rgba(20, 27, 52, .86)
        );
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        min-height: 112px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, .18);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .9rem; }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.75rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        background: linear-gradient(
            145deg, rgba(25, 36, 67, .75), rgba(17, 23, 45, .7)
        );
        border-radius: 20px !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(15, 22, 43, .65);
        margin: .7rem 0;
    }
    [data-testid="stExpander"] summary {
        color: var(--ink);
        font-size: 1.03rem;
        font-weight: 650;
        padding: .35rem .4rem;
    }
    [data-testid="stTextInput"] input, [data-testid="stFileUploader"] section {
        border-radius: 12px;
    }
    [data-testid="stTextInput"] input {
        min-height: 2.9rem;
        background: rgba(8, 14, 29, .72);
        border-color: rgba(174, 194, 242, .24);
        color: var(--ink);
        font-size: 1rem;
    }
    .stButton > button,
    [data-testid="stFormSubmitButton"] button {
        border-radius: 12px;
        min-height: 2.75rem;
        border: 1px solid rgba(89, 225, 210, .38);
        background: linear-gradient(
            110deg, rgba(42, 157, 156, .2), rgba(124, 91, 214, .22)
        );
        color: var(--ink);
        font-weight: 650;
        transition:
            border-color .18s ease,
            transform .18s ease,
            box-shadow .18s ease;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
        border-color: var(--teal);
        box-shadow: 0 0 22px rgba(89, 225, 210, .16);
        transform: translateY(-1px);
        color: white;
    }
    [data-testid="stChatMessage"] {
        color: #f4f7ff !important;
        border: 1px solid rgba(153, 179, 230, .3);
        border-radius: 18px;
        background: #121b31;
        padding: 1rem 1.15rem;
        margin: .75rem 0;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {
        color: #f4f7ff !important;
        font-size: 1.05rem;
        line-height: 1.75;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {
        color: #75eee0 !important;
        text-decoration: underline;
        text-underline-offset: .18em;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {
        color: #f4f7ff !important;
        background: #080f20 !important;
    }
    :focus-visible {
        outline: 3px solid #75eee0 !important;
        outline-offset: 3px;
    }
    [data-testid="stChatInput"] textarea {
        border: 1px solid rgba(89, 225, 210, .35);
        border-radius: 16px;
        background: rgba(14, 21, 41, .94);
        color: var(--ink);
        font-size: 1rem;
    }
    [data-testid="stChatInput"] {
        background: transparent !important;
    }
    [data-testid="stChatInput"] > div {
        background: rgba(14, 21, 41, .94) !important;
        border: 1px solid rgba(89, 225, 210, .35);
        border-radius: 16px;
    }
    [data-testid="stChatInputSubmitButton"] {
        background: rgba(89, 225, 210, .14) !important;
        color: var(--ink) !important;
        border: 1px solid rgba(89, 225, 210, .3) !important;
        border-radius: 10px !important;
    }
    [data-testid="stBottom"] > div {
        background: rgba(8, 13, 27, .96) !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 14px;
    }
    [data-testid="stAlert"] { border-radius: 14px; }
    .eyebrow {
        color: var(--teal);
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .18em;
        text-transform: uppercase;
        margin: 0 0 .45rem;
    }
    .lede {
        color: var(--muted);
        font-size: 1.08rem;
        max-width: 760px;
        margin-top: -.25rem;
        margin-bottom: 1.8rem;
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1.5rem 1rem 3rem; }
        [data-testid="stMetric"] { min-height: 92px; padding: .75rem; }
        [data-testid="stMetricValue"] { font-size: 1.4rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def post_to_api(
    path: str,
    *,
    admin_token: str | None = None,
    **kwargs: object,
) -> requests.Response | None:
    try:
        headers = {}
        if API_ACCESS_KEY:
            headers["X-API-Key"] = API_ACCESS_KEY
        if admin_token:
            headers["Authorization"] = f"Bearer {admin_token}"
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


def get_from_api(
    path: str, *, admin_token: str, **kwargs: object
) -> requests.Response | None:
    try:
        headers = {"Authorization": f"Bearer {admin_token}"}
        if API_ACCESS_KEY:
            headers["X-API-Key"] = API_ACCESS_KEY
        return requests.get(f"{API_URL}{path}", headers=headers, **kwargs)
    except requests.ConnectionError:
        st.error("Cannot connect to the API. Check that the API service is running.")
    except requests.Timeout:
        st.error("The API request timed out. Please try again.")
    except requests.RequestException as error:
        st.error(f"API request failed: {error}")
    return None


def _clear_admin_session() -> None:
    st.session_state["admin_token"] = None
    st.session_state["admin_expires_at"] = 0.0
    st.session_state["admin_mode"] = False
    st.session_state["admin_chunks"] = []
    st.session_state["admin_chunk_offset"] = None
    st.session_state["admin_chunks_loaded"] = False


def _authenticate_admin() -> None:
    username = st.session_state.get("admin_username_input", "")
    password = st.session_state.get("admin_password_input", "")
    response = post_to_api(
        "/admin/login",
        json={"username": username, "password": password},
        timeout=15,
    )
    st.session_state["admin_password_input"] = ""
    if response is None:
        st.session_state["admin_login_error"] = (
            "Unable to reach the authentication service."
        )
    elif response.status_code == 401:
        st.session_state["admin_login_error"] = "Invalid username or password."
    elif not response.ok:
        st.session_state["admin_login_error"] = (
            f"Admin login failed ({response.status_code}): {response.text}"
        )
    else:
        result = response.json()
        st.session_state["admin_token"] = result["access_token"]
        st.session_state["admin_expires_at"] = time.time() + result["expires_in"]
        st.session_state["admin_mode"] = True
        st.session_state["admin_login_open"] = False
        st.session_state["admin_login_error"] = ""


def _load_admin_chunks(admin_token: str) -> None:
    response = get_from_api(
        "/admin/chunks",
        admin_token=admin_token,
        params={
            "limit": 50,
            "offset": st.session_state.get("admin_chunk_offset"),
        },
        timeout=30,
    )
    if response is None:
        return
    if response.status_code == 401:
        _clear_admin_session()
        st.warning("Your admin session expired. Please log in again.")
        st.rerun()
    if not response.ok:
        st.error(f"Could not load chunk details ({response.status_code}).")
        return
    result = response.json()
    st.session_state["admin_chunks"].extend(result["chunks"])
    st.session_state["admin_chunk_offset"] = result["next_page_offset"]
    st.session_state["admin_chunks_loaded"] = True


def _render_admin_portal(admin_token: str) -> None:
    st.markdown(
        '<p class="eyebrow">CONTROL ROOM · ADMIN</p>',
        unsafe_allow_html=True,
    )
    st.header("Admin portal")
    st.markdown(
        '<p class="lede">A clear view of your knowledge base and '
        "connected services.</p>",
        unsafe_allow_html=True,
    )

    if st.button("Refresh admin status"):
        st.session_state["admin_chunks"] = []
        st.session_state["admin_chunk_offset"] = None
        st.session_state["admin_chunks_loaded"] = False

    status_response = get_from_api("/admin/status", admin_token=admin_token, timeout=20)
    config_response = get_from_api("/admin/config", admin_token=admin_token, timeout=20)
    if any(
        response is not None and response.status_code == 401
        for response in (status_response, config_response)
    ):
        _clear_admin_session()
        st.warning("Your admin session expired. Please log in again.")
        st.rerun()

    if status_response is not None and status_response.ok:
        status = status_response.json()
        st.subheader("Vector database")
        first, second, third = st.columns(3)
        first.metric("Collection", status["collection_name"])
        second.metric("Indexed chunks", status["total_chunks_ingested"])
        third.metric(
            "Collection status",
            "Contains documents" if status["has_documents"] else "Empty",
        )
    elif status_response is not None:
        st.error(f"Could not load Qdrant status ({status_response.status_code}).")

    if config_response is not None and config_response.ok:
        config = config_response.json()
        with st.expander("Connected services & model setup", expanded=False):
            left, right = st.columns(2)
            with left:
                st.markdown("#### Vector database")
                st.write(f"**Host:** {config['qdrant_host']}")
                st.write(f"**Collection:** {config['qdrant_collection']}")
                st.write(
                    "**Access key:** "
                    + (
                        "Configured"
                        if config["qdrant_api_key_configured"]
                        else "Not configured"
                    )
                )
            with right:
                st.markdown("#### Language model")
                st.write(f"**Host:** {config['llm_host']}")
                st.write(f"**Model:** {config['llm_model']}")
                st.write(
                    "**Access key:** "
                    + (
                        "Configured"
                        if config["llm_api_key_configured"]
                        else "Not configured"
                    )
                )
            st.divider()
            st.write(f"**Embedding model:** {config['embedding_model']}")
            st.write(f"**Reranker model:** {config['reranker_model']}")
            st.write(
                "**API access key:** "
                + (
                    "Configured"
                    if config["api_access_key_configured"]
                    else "Not configured"
                )
            )
    elif config_response is not None:
        st.error(
            f"Could not load service configuration ({config_response.status_code})."
        )

    with st.expander("Ingest documents", expanded=True):
        st.caption("Add PDF, Markdown, or DOCX files to the active collection.")
        uploads = st.file_uploader(
            "Choose documents",
            type=["pdf", "md", "markdown", "docx"],
            accept_multiple_files=True,
            key="admin_uploads",
        )
        if st.button("Ingest selected documents", disabled=not uploads):
            response = post_to_api(
                "/admin/ingest",
                admin_token=admin_token,
                files=[
                    ("files", (upload.name, upload.getvalue(), upload.type))
                    for upload in uploads or []
                ],
                timeout=300,
            )
            if response is not None and response.ok:
                st.success(
                    f"Indexed {response.json()['chunks_indexed']} chunks "
                    f"from {response.json()['files_processed']} files."
                )
                st.session_state["admin_chunks"] = []
                st.session_state["admin_chunk_offset"] = None
                st.session_state["admin_chunks_loaded"] = False
            elif response is not None:
                st.error(f"Ingestion failed ({response.status_code}): {response.text}")

    with st.expander("Indexed chunk details", expanded=False):
        st.caption(
            "Document names and chunk metadata only; file contents are not shown."
        )
        if not st.session_state["admin_chunks_loaded"]:
            _load_admin_chunks(admin_token)
        if st.session_state["admin_chunks"]:
            st.dataframe(st.session_state["admin_chunks"], use_container_width=True)
        else:
            st.info("No indexed chunk details are available yet.")
        if st.session_state["admin_chunk_offset"]:
            if st.button("Load more chunk details"):
                _load_admin_chunks(admin_token)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for key, default in (
    ("admin_token", None),
    ("admin_expires_at", 0.0),
    ("admin_mode", False),
    ("admin_login_open", False),
    ("admin_login_error", ""),
    ("admin_chunks", []),
    ("admin_chunk_offset", None),
    ("admin_chunks_loaded", False),
):
    if key not in st.session_state:
        st.session_state[key] = default

if (
    st.session_state["admin_token"]
    and time.time() >= st.session_state["admin_expires_at"]
):
    _clear_admin_session()

with st.sidebar:
    st.title("Enterprise RAG")
    if st.session_state["admin_token"]:
        if st.button("Back to chat"):
            st.session_state["admin_mode"] = False
            st.rerun()
        if st.button("Admin logout"):
            _clear_admin_session()
            st.rerun()
    elif st.button("Admin"):
        st.session_state["admin_login_open"] = not st.session_state["admin_login_open"]

    if not st.session_state["admin_token"] and st.session_state["admin_login_open"]:
        with st.form("admin_login_form"):
            st.text_input("Admin username", key="admin_username_input")
            st.text_input(
                "Admin password",
                type="password",
                key="admin_password_input",
            )
            st.form_submit_button("Log in", on_click=_authenticate_admin)
        if st.session_state["admin_login_error"]:
            st.error(st.session_state["admin_login_error"])

    if not st.session_state["admin_mode"] and st.button("Clear chat"):
        st.session_state.chat_history = []

if st.session_state["admin_mode"] and st.session_state["admin_token"]:
    _render_admin_portal(st.session_state["admin_token"])
else:
    st.markdown(
        '<p class="eyebrow">YOUR KNOWLEDGE, IN CONVERSATION</p>', unsafe_allow_html=True
    )
    st.title("Enterprise RAG Assistant")
    st.markdown(
        '<p class="lede">Ask a question and get a grounded answer from your '
        "indexed documents.</p>",
        unsafe_allow_html=True,
    )
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask about your documents")
    if query:
        history = st.session_state.chat_history[-MAX_API_HISTORY_MESSAGES:]
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and preparing an answer..."):
                response = post_to_api(
                    "/chat",
                    json={"query": query, "history": history},
                    timeout=180,
                )
                if response is not None and response.ok:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.chat_history.extend(
                        [
                            {"role": "user", "content": query},
                            {"role": "assistant", "content": answer},
                        ]
                    )
                    st.session_state.chat_history = st.session_state.chat_history[
                        -MAX_SESSION_MESSAGES:
                    ]
                elif response is not None:
                    st.error(
                        f"Request failed ({response.status_code}): {response.text}"
                    )
