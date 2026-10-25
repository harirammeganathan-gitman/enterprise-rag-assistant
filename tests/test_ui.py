from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def test_chat_history_persists_for_session_and_can_be_cleared() -> None:
    app = AppTest.from_file("app/ui.py").run()
    response = SimpleNamespace(ok=True, json=lambda: {"answer": "Hello!"})

    with patch("requests.post", return_value=response) as api_call:
        app.chat_input[0].set_value("hi").run()
        app.chat_input[0].set_value("hello").run()

    assert app.session_state["chat_history"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered_markdown = [element.value for element in app.markdown]
    assert all(
        value in rendered_markdown
        for value in [
            "hi",
            "Hello!",
            "hello",
            "Hello!",
        ]
    )
    assert api_call.call_args_list[0].kwargs["json"]["history"] == []
    assert api_call.call_args_list[1].kwargs["json"]["history"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
    ]

    next(button for button in app.button if button.label == "Clear chat").click().run()

    assert app.session_state["chat_history"] == []


def test_admin_controls_are_hidden_until_authenticated() -> None:
    app = AppTest.from_file("app/ui.py").run()

    assert [button.label for button in app.button] == ["Admin", "Clear chat"]
    assert len(app.get("file_uploader")) == 0

    next(button for button in app.button if button.label == "Admin").click().run()

    assert {item.label for item in app.text_input} == {
        "Admin username",
        "Admin password",
    }
    assert len(app.get("file_uploader")) == 0
    assert "Admin portal" not in [item.value for item in app.header]


def test_valid_admin_login_opens_portal_without_exposing_secrets() -> None:
    app = AppTest.from_file("app/ui.py").run()
    next(button for button in app.button if button.label == "Admin").click().run()
    app.text_input[0].set_value("admin")
    app.text_input[1].set_value("private-password")

    login_response = SimpleNamespace(
        ok=True,
        status_code=200,
        json=lambda: {"access_token": "signed-admin-token", "expires_in": 1800},
        text="",
    )
    status_response = SimpleNamespace(
        ok=True,
        status_code=200,
        json=lambda: {
            "collection_name": "documents_local",
            "total_chunks_ingested": 0,
            "has_documents": False,
        },
        text="",
    )
    config_response = SimpleNamespace(
        ok=True,
        status_code=200,
        json=lambda: {
            "qdrant_host": "cluster.example",
            "qdrant_collection": "documents_local",
            "qdrant_api_key_configured": True,
            "llm_host": "llm.example",
            "llm_model": "test-model",
            "llm_api_key_configured": True,
            "embedding_model": "test-embedding",
            "reranker_model": "test-reranker",
            "api_access_key_configured": True,
        },
        text="",
    )
    chunks_response = SimpleNamespace(
        ok=True,
        status_code=200,
        json=lambda: {"chunks": [], "next_page_offset": None},
        text="",
    )

    def fake_get(url, **kwargs):
        if url.endswith("/admin/status"):
            return status_response
        if url.endswith("/admin/config"):
            return config_response
        if url.endswith("/admin/chunks"):
            return chunks_response
        raise AssertionError(f"Unexpected API request: {url}")

    with (
        patch("requests.post", return_value=login_response) as api_login,
        patch("requests.get", side_effect=fake_get),
    ):
        next(button for button in app.button if button.label == "Log in").click().run()

    assert app.session_state["admin_mode"] is True
    assert app.session_state["admin_token"] == "signed-admin-token"
    assert app.session_state["admin_password_input"] == ""
    assert "private-password" not in repr(app.session_state["chat_history"])
    assert "Admin portal" in [item.value for item in app.header]
    assert "Ingest selected documents" in [item.label for item in app.button]
    assert api_login.call_args.kwargs["json"] == {
        "username": "admin",
        "password": "private-password",
    }
    assert "Authorization" not in api_login.call_args.kwargs.get("headers", {})
