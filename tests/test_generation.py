from types import SimpleNamespace

from src.generation import answer as answer_module


def test_generation_uses_configured_local_model_without_cloud_key(
    monkeypatch,
) -> None:
    settings = SimpleNamespace(
        local_model_base_url="http://localhost:11434/v1",
        local_model_name="llama3.2",
        local_model_api_key="ollama",
    )
    calls = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        @staticmethod
        def create_completion(**kwargs):
            calls["completion"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Answer [1]"))]
            )

    monkeypatch.setattr(answer_module, "get_settings", lambda: settings)
    monkeypatch.setattr(answer_module, "OpenAI", FakeOpenAI)

    answer = answer_module.generate_answer("Question?", ["Local context"])

    assert answer == "Answer [1]"
    assert calls["client"] == {
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
    }
    assert calls["completion"]["model"] == "llama3.2"
    assert "Local context" in calls["completion"]["messages"][1]["content"]
    assert calls["client"]["api_key"] == "ollama"
