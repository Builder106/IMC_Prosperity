from unittest.mock import MagicMock, patch

import pytest
from src.rag.groq_llm import run_groq_chat


def _fake_completion(text):
    completion = MagicMock()
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    completion.choices = [choice]
    return completion


def test_run_groq_chat_returns_message_text():
    with patch("src.rag.groq_llm.Groq") as mock_groq:
        client = mock_groq.return_value
        client.chat.completions.create.return_value = _fake_completion("response text\n")

        result = run_groq_chat(
            "hello",
            model="llama-3.3-70b-versatile",
            api_key="test-key",
            temperature=0.2,
            timeout_seconds=5,
        )

    assert result == "response text"
    mock_groq.assert_called_once_with(api_key="test-key", timeout=5)
    client.chat.completions.create.assert_called_once_with(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
    )


def test_run_groq_chat_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        run_groq_chat("hello", model="llama-3.3-70b-versatile", api_key=None)
