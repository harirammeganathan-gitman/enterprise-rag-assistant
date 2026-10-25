from src.config import Settings


def test_cloud_llm_environment_variables_configure_generation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "cloud-model")
    monkeypatch.setenv("LLM_API_KEY", "cloud-secret")
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LOCAL_MODEL_NAME", "llama3.2")
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "ollama")

    settings = Settings(_env_file=None)

    assert settings.local_model_base_url == "https://llm.example/v1"
    assert settings.local_model_name == "cloud-model"
    assert settings.local_model_api_key == "cloud-secret"


def test_legacy_local_model_environment_variables_remain_supported(
    monkeypatch,
) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LOCAL_MODEL_NAME", "llama3.2")
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "ollama")

    settings = Settings(_env_file=None)

    assert settings.local_model_base_url == "http://localhost:11434/v1"
    assert settings.local_model_name == "llama3.2"
    assert settings.local_model_api_key == "ollama"
