import json
from dataclasses import dataclass
from pathlib import Path

try:
    from langchain_core.documents import Document  # type: ignore
except ImportError:
    @dataclass
    class Document:
        page_content: str
        metadata: dict


def _normalize_message(message):
    message_type = message.get("type", "Default")
    content = (message.get("content") or "").strip()
    if message_type != "Default":
        return None
    if not content:
        return None
    author = message.get("author") or {}
    return {
        "id": str(message.get("id", "")),
        "timestamp": message.get("timestamp", ""),
        "author": author.get("name", "unknown"),
        "content": content,
    }


def _chunk_lines(lines, chunk_size):
    prepared_lines = []
    for line in lines:
        if len(line) <= chunk_size:
            prepared_lines.append(line)
            continue
        start = 0
        while start < len(line):
            prepared_lines.append(line[start:start + chunk_size])
            start += chunk_size

    chunks = []
    current_lines = []
    current_len = 0
    for line in prepared_lines:
        line_len = len(line) + 1
        if current_lines and current_len + line_len > chunk_size:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = line_len
        else:
            current_lines.append(line)
            current_len += line_len
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


def load_discord_exports(discord_dir, chunk_size=1200):
    discord_path = Path(discord_dir)
    if not discord_path.exists():
        return []

    documents = []
    for export_file in sorted(discord_path.glob("*.json")):
        try:
            data = json.loads(export_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        guild_name = ((data.get("guild") or {}).get("name")) or ""
        channel_name = ((data.get("channel") or {}).get("name")) or export_file.stem
        thread_name = ((data.get("thread") or {}).get("name")) or ""
        messages = data.get("messages") or []

        formatted_lines = []
        normalized_messages = []
        for message in messages:
            normalized = _normalize_message(message)
            if not normalized:
                continue
            normalized_messages.append(normalized)
            formatted_lines.append(
                f"[{normalized['timestamp']}] {normalized['author']}: {normalized['content']}"
            )

        if not formatted_lines:
            continue

        chunks = _chunk_lines(formatted_lines, chunk_size)
        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": str(export_file),
                        "type": "discord",
                        "guild": guild_name,
                        "channel": channel_name,
                        "thread": thread_name,
                        "chunk_index": idx,
                        "message_count": len(normalized_messages),
                    },
                )
            )

    return documents
