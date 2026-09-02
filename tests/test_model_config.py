from src.rag import model_config


def test_model_config_defaults(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("RAG_MAX_CONTEXT_CHARS", raising=False)

    assert model_config.get_llm_model_name() == "llama-3.3-70b-versatile"
    assert model_config.get_llm_temperature() == 0.2
    assert model_config.get_embedding_model_name() == "sentence-transformers/all-MiniLM-L6-v2"
    assert model_config.get_groq_api_key() == ""
    assert model_config.get_groq_timeout_seconds() == 180
    assert model_config.get_max_completion_tokens() == 4096
    assert model_config.get_max_context_chars() == 12000


def test_model_config_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-embed")
    monkeypatch.setenv("GROQ_API_KEY", "custom-key")
    monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
    monkeypatch.setenv("RAG_MAX_CONTEXT_CHARS", "8000")

    assert model_config.get_llm_model_name() == "custom-model"
    assert model_config.get_llm_temperature() == 0.7
    assert model_config.get_embedding_model_name() == "custom-embed"
    assert model_config.get_groq_api_key() == "custom-key"
    assert model_config.get_groq_timeout_seconds() == 300
    assert model_config.get_max_completion_tokens() == 2048
    assert model_config.get_max_context_chars() == 8000
