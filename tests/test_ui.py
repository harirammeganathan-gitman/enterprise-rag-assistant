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
    assert [element.value for element in app.markdown] == [
        "hi",
        "Hello!",
        "hello",
        "Hello!",
    ]
    assert api_call.call_args_list[0].kwargs["json"]["history"] == []
    assert api_call.call_args_list[1].kwargs["json"]["history"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
    ]

    app.button[0].click().run()

    assert app.session_state["chat_history"] == []
