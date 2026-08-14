from src.utils.notion_scraper import notion_scraper_stable as scraper
from playwright.sync_api import Error as PlaywrightError


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
    page = FakePage([PlaywrightError("timeout"), PlaywrightError("timeout"), PlaywrightError("timeout")])
    result = scraper.safe_goto(page, "https://example.com", retries=2)
    assert result is False
    assert len(page.calls) == 3
