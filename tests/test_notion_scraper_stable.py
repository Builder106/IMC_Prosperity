import json

from bs4 import BeautifulSoup
from playwright.sync_api import Error as PlaywrightError
from src.utils.notion_scraper import notion_scraper_stable as scraper


class FakePage:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def goto(self, url, timeout=None, wait_until=None):
        self.calls.append(
            {
                "url": url,
                "timeout": timeout,
                "wait_until": wait_until,
            }
        )
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def wait_for_timeout(self, _):
        return None


def test_safe_goto_succeeds_first_try():
    page = FakePage([None])
    result = scraper.safe_goto(page, "https://example.com")
    assert result is True
    assert len(page.calls) == 1


def test_safe_goto_retries_then_succeeds():
    page = FakePage([PlaywrightError("timeout"), None])
    result = scraper.safe_goto(page, "https://example.com", retries=1)
    assert result is True
    assert len(page.calls) == 2


def test_safe_goto_fails_after_all_retries():
    page = FakePage(
        [PlaywrightError("timeout"), PlaywrightError("timeout"), PlaywrightError("timeout")]
    )
    result = scraper.safe_goto(page, "https://example.com", retries=2)
    assert result is False
    assert len(page.calls) == 3


def test_load_code_file_mapping(tmp_path):
    mapping_file = tmp_path / "code_map.md"
    mapping_file.write_text(
        "# Header\n| ID | Description |\n|---|---|\n| code_1 | My Strategy Script |\n",
        encoding="utf-8",
    )
    mapping = scraper.load_code_file_mapping(str(mapping_file))
    assert mapping.get("code_1") == "My Strategy Script.py"


def test_save_json(tmp_path):
    data = [{"title": "Test"}]
    scraper.save_json(data, str(tmp_path), "test.json")
    saved_file = tmp_path / "test.json"
    assert saved_file.exists()
    assert json.loads(saved_file.read_text(encoding="utf-8")) == data


def test_determine_category():
    assert scraper.determine_category("Round 1 Overview", []) == "rounds"
    assert scraper.determine_category("Glossary", []) == "e-learning_center"
    assert scraper.determine_category("About IMC", []) == "about_prosperity"


def test_process_code_content():
    # Python code indentation handling
    py_code = """Python Copy
def trade():
if True:
x = 1
return x
"""
    processed = scraper.process_code_content(py_code, "python")
    assert "def trade():" in processed
    assert "    if True:" in processed
    assert "        x = 1" in processed

    # Non-python code
    non_py = "SELECT * FROM trades;"
    assert scraper.process_code_content(non_py, "sql") == "SELECT * FROM trades;"
    assert scraper.process_code_content("", "python") == ""


def test_extract_content_and_elements():
    html = """
    <html>
      <head><title>Test Page</title></head>
      <body>
        <div class="notion-page-block">
          <h1>Main Title</h1>
        </div>
        <div class="notion-page-content">
          <h2>Section Title</h2>
          <div class="notion-text-block">This is a paragraph.</div>
        </div>
      </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    blocks = scraper.extract_content(soup, "Test Page")
    assert len(blocks) > 0
    assert any(b.get("type") == "h1" for b in blocks)
    assert any(b.get("type") == "h2" for b in blocks)
    assert any(b.get("type") == "p" for b in blocks)
