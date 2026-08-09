
from src.rag.build_rag_system import process_notion_wiki_data


def test_process_notion_wiki_data_reads_markdown(tmp_path):
    wiki_dir = tmp_path / "prosperity_wiki"
    about_dir = wiki_dir / "about_prosperity"
    about_dir.mkdir(parents=True)
    faq_md = about_dir / "faq.md"
    faq_md.write_text("# FAQ\n\nThis is a markdown FAQ file.", encoding="utf-8")

    docs = process_notion_wiki_data(wiki_dir=wiki_dir)

    assert len(docs) == 1
    assert docs[0].page_content.startswith("# FAQ")
    assert "This is a markdown FAQ file." in docs[0].page_content
    assert docs[0].metadata["category"] == "about_prosperity"
    assert docs[0].metadata["title"] == "FAQ"
    assert docs[0].metadata["source"] == str(faq_md)
