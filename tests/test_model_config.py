import importlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _reload_module():
    from src.rag import model_config

    return importlib.reload(model_config)


def test_defaults_use_groq_and_local_embeddings(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_TIMEOUT_SECONDS", raising=False)

    module = _reload_module()

    assert module.get_llm_model_name() == "llama-3.3-70b-versatile"
    assert module.get_embedding_model_name() == "sentence-transformers/all-MiniLM-L6-v2"
    assert module.get_groq_api_key() == ""
    assert module.get_groq_timeout_seconds() == 180


def test_env_overrides_model_selection(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "240")

    module = _reload_module()

    assert module.get_llm_model_name() == "llama-3.1-8b-instant"
    assert module.get_embedding_model_name() == "sentence-transformers/all-mpnet-base-v2"
    assert module.get_groq_api_key() == "gsk_test_key"
    assert module.get_groq_timeout_seconds() == 240


def test_numeric_defaults_and_overrides(monkeypatch):
    for name in (
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
        "RAG_MAX_CONTEXT_CHARS",
    ):
        monkeypatch.delenv(name, raising=False)

    module = _reload_module()

    assert module.get_llm_temperature() == 0.2
    assert module.get_max_completion_tokens() == 4096
    assert module.get_max_context_chars() == 12000

    monkeypatch.setenv("LLM_TEMPERATURE", "0.75")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
    monkeypatch.setenv("RAG_MAX_CONTEXT_CHARS", "6000")
    module = _reload_module()

    assert module.get_llm_temperature() == 0.75
    assert module.get_max_completion_tokens() == 2048
    assert module.get_max_context_chars() == 6000
