import json
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from src.rag.build_rag_system import (
    create_combined_retriever,
    create_rag_chain,
    create_vector_stores,
    extract_code_blocks,
    extract_title,
    process_discord_data,
    process_list_items,
    process_notion_wiki_data,
    process_trading_data,
)


def test_process_list_items():
    items = [{"content": "item 1", "level": 0}, {"content": "nested item", "level": 1}]
    bulleted = process_list_items(items, style="bulleted")
    assert "- item 1" in bulleted
    assert "  - nested item" in bulleted

    numbered = process_list_items(items, style="numbered")
    assert "1. item 1" in numbered
    assert "  1. nested item" in numbered


def test_extract_title():
    data_list = [{"type": "h1", "content": "My Heading Title"}]
    assert extract_title(data_list) == "My Heading Title"

    data_dict = {"title": "Dict Title"}
    assert extract_title(data_dict) == "Dict Title"

    assert extract_title([]) == ""


def test_extract_code_blocks():
    docs = [
        Document(
            page_content="Some intro text\n```python\nprint('hello')\n```\nMore text",
            metadata={"source": "doc1"},
        ),
        Document(page_content="No code here", metadata={"source": "doc2"}),
    ]
    code_docs = extract_code_blocks(docs)
    assert len(code_docs) == 1
    assert code_docs[0].page_content == "print('hello')"
    assert code_docs[0].metadata["language"] == "python"
    assert code_docs[0].metadata["content_type"] == "code_block"


def test_process_notion_wiki_data(tmp_path):
    wiki_dir = tmp_path / "prosperity_wiki"
    cat_dir = wiki_dir / "about_prosperity"
    cat_dir.mkdir(parents=True)

    # Markdown file
    md_file = wiki_dir / "index.md"
    md_file.write_text("# Welcome to Prosperity Wiki\nSome content", encoding="utf-8")

    # JSON file with list
    json_data = [
        {"type": "h1", "content": "Introduction"},
        {"type": "p", "content": "Paragraph 1"},
        {"type": "list", "style": "bulleted", "items": [{"content": "Point 1"}]},
        {"type": "code", "language": "python", "code": "x = 10"},
    ]
    json_file = cat_dir / "intro.json"
    json_file.write_text(json.dumps(json_data), encoding="utf-8")

    docs = process_notion_wiki_data(wiki_dir)
    assert len(docs) >= 2
    assert any("Welcome to Prosperity Wiki" in d.page_content for d in docs)
    assert any("Introduction" in d.page_content for d in docs)


def test_process_discord_data(tmp_path):
    discord_dir = tmp_path / "discord"
    discord_dir.mkdir(parents=True)
    json_data = {
        "guild": {"name": "IMC Guild"},
        "channel": {"name": "general"},
        "messages": [
            {
                "type": "Default",
                "id": "123",
                "timestamp": "2026-04-01T12:00:00",
                "author": {"name": "TraderJoe"},
                "content": "Resin prices are surging!",
            }
        ],
    }
    (discord_dir / "general.json").write_text(json.dumps(json_data), encoding="utf-8")

    docs = process_discord_data(discord_dir)
    assert len(docs) == 1
    assert "TraderJoe: Resin prices are surging!" in docs[0].page_content


def test_process_trading_data():
    with (
        patch("src.rag.build_rag_system.discover_rounds") as mock_discover,
        patch("src.rag.build_rag_system.process_round_data") as mock_process,
    ):
        mock_discover.return_value = ["round_1"]
        mock_process.return_value = [
            {"content": "Trading data for RESIN", "metadata": {"product": "RESIN", "day": 1}}
        ]

        docs = process_trading_data()
        assert len(docs) == 1
        assert docs[0].page_content == "Trading data for RESIN"


def test_create_vector_stores_and_retrievers():
    with (
        patch("src.rag.build_rag_system.HuggingFaceEmbeddings"),
        patch("src.rag.build_rag_system.Chroma") as mock_chroma,
        patch("src.rag.build_rag_system.EnsembleRetriever") as mock_ensemble,
    ):
        mock_vs = MagicMock()
        mock_chroma.from_documents.return_value = mock_vs
        mock_ensemble.return_value = MagicMock()

        notion_docs = [
            Document(page_content="Notion doc ```python\nx = 1\n```", metadata={"source": "notion"})
        ]
        trading_docs = [Document(page_content="Trading doc", metadata={"source": "trading"})]

        notion_vs, trading_vs, code_vs = create_vector_stores(notion_docs, trading_docs)
        assert notion_vs is not None
        assert trading_vs is not None
        assert code_vs is not None

        retriever = create_combined_retriever(notion_vs, trading_vs, code_vs)
        assert retriever is not None

        chain = create_rag_chain(retriever)
        assert chain is not None


def test_create_combined_retriever_empty():
    assert create_combined_retriever(None, None, None) is None
