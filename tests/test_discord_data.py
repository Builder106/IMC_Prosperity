import builtins
import importlib
import json

from src.rag.discord_data import load_discord_exports


def test_discord_document_fallback_when_langchain_is_unavailable(monkeypatch):
    import src.rag.discord_data as discord_data

    original_import = builtins.__import__

    def import_without_langchain(name, *args, **kwargs):
        if name == "langchain_core.documents":
            raise ImportError("langchain is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_langchain)
    importlib.reload(discord_data)

    try:
        document = discord_data.Document(page_content="message", metadata={"type": "discord"})
        assert document.page_content == "message"
        assert document.metadata == {"type": "discord"}
    finally:
        monkeypatch.undo()
        importlib.reload(discord_data)


def test_load_discord_exports_reads_messages(tmp_path):
    discord_dir = tmp_path / "discord"
    discord_dir.mkdir()
    export_path = discord_dir / "round-2-chat.json"
    export_path.write_text(
        json.dumps(
            {
                "guild": {"name": "IMC Prosperity"},
                "channel": {"name": "round-2-chat"},
                "messages": [
                    {
                        "id": "1",
                        "timestamp": "2026-04-17T12:00:00.0000000+00:00",
                        "content": "Kelp spread looks tighter today.",
                        "author": {"name": "alice"},
                        "type": "Default",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    docs = load_discord_exports(discord_dir, chunk_size=500)

    assert len(docs) == 1
    assert "Kelp spread looks tighter today." in docs[0].page_content
    assert docs[0].metadata["type"] == "discord"
    assert docs[0].metadata["channel"] == "round-2-chat"
    assert docs[0].metadata["guild"] == "IMC Prosperity"


def test_load_discord_exports_filters_non_default_and_empty(tmp_path):
    discord_dir = tmp_path / "discord"
    discord_dir.mkdir()
    export_path = discord_dir / "announcements.json"
    export_path.write_text(
        json.dumps(
            {
                "guild": {"name": "IMC Prosperity"},
                "channel": {"name": "announcements"},
                "messages": [
                    {
                        "id": "1",
                        "timestamp": "2026-04-17T12:00:00.0000000+00:00",
                        "content": "",
                        "author": {"name": "system"},
                        "type": "Default",
                    },
                    {
                        "id": "2",
                        "timestamp": "2026-04-17T12:01:00.0000000+00:00",
                        "content": "Pinned a message",
                        "author": {"name": "system"},
                        "type": "ChannelPinnedMessage",
                    },
                    {
                        "id": "3",
                        "timestamp": "2026-04-17T12:02:00.0000000+00:00",
                        "content": "Round 2 starts tomorrow.",
                        "author": {"name": "mod"},
                        "type": "Default",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    docs = load_discord_exports(discord_dir, chunk_size=500)

    assert len(docs) == 1
    assert "Round 2 starts tomorrow." in docs[0].page_content
    assert "Pinned a message" not in docs[0].page_content


def test_load_discord_exports_chunks_large_threads(tmp_path):
    discord_dir = tmp_path / "discord"
    discord_dir.mkdir()
    export_path = discord_dir / "strategy.json"
    export_path.write_text(
        json.dumps(
            {
                "guild": {"name": "IMC Prosperity"},
                "channel": {"name": "strategy"},
                "messages": [
                    {
                        "id": "1",
                        "timestamp": "2026-04-17T12:00:00.0000000+00:00",
                        "content": "A" * 1400,
                        "author": {"name": "alice"},
                        "type": "Default",
                    },
                    {
                        "id": "2",
                        "timestamp": "2026-04-17T12:01:00.0000000+00:00",
                        "content": "B" * 1400,
                        "author": {"name": "bob"},
                        "type": "Default",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    docs = load_discord_exports(discord_dir, chunk_size=1000)

    assert len(docs) >= 2
    assert all(len(doc.page_content) <= 1200 for doc in docs)


def test_load_discord_exports_handles_missing_directory_and_bad_json(tmp_path):
    assert load_discord_exports(tmp_path / "missing") == []

    discord_dir = tmp_path / "discord"
    discord_dir.mkdir()
    (discord_dir / "broken.json").write_text("{not json", encoding="utf-8")
    (discord_dir / "empty.json").write_text(json.dumps({"messages": []}), encoding="utf-8")

    assert load_discord_exports(discord_dir) == []


def test_load_discord_exports_uses_fallback_metadata_and_message_defaults(tmp_path):
    discord_dir = tmp_path / "discord"
    discord_dir.mkdir()
    (discord_dir / "fallback.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"content": "  useful note  ", "author": {}, "type": "Default"},
                ]
            }
        ),
        encoding="utf-8",
    )

    docs = load_discord_exports(discord_dir)

    assert len(docs) == 1
    assert docs[0].metadata["channel"] == "fallback"
    assert docs[0].metadata["guild"] == ""
    assert docs[0].metadata["thread"] == ""
    assert docs[0].metadata["message_count"] == 1
    assert "unknown: useful note" in docs[0].page_content


def test_load_discord_exports_skips_non_default_and_missing_content(tmp_path):
    discord_dir = tmp_path / "discord"
    discord_dir.mkdir()
    (discord_dir / "filtered.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"type": "ChannelPinnedMessage", "content": "ignore"},
                    {"type": "Default", "content": "   "},
                    {"type": "Default", "content": "keep"},
                ]
            }
        ),
        encoding="utf-8",
    )

    docs = load_discord_exports(discord_dir)

    assert len(docs) == 1
    assert "keep" in docs[0].page_content
    assert "ignore" not in docs[0].page_content
