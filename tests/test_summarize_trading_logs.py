from src.utils.summarize_trading_logs import (
    extract_representative_examples,
    format_summary_for_llm,
    summarize_trading_logs,
)


def test_summarize_trading_logs(tmp_path):
    sample_log = """
{"sandboxLog": "", "lambdaLog": "Processing RAINFOREST_RESIN - Current position: 10\nRAINFOREST_RESIN: Collecting price history (5/20)\nRAINFOREST_RESIN: Placed 2 orders\n--- Orders Generated: [Order('RAINFOREST_RESIN', 100, 1)]", "timestamp": 1000}
{"sandboxLog": "", "lambdaLog": "Processing RAINFOREST_RESIN - Current position: 20\nProcessing SQUID_INK - Current position: -5\nSQUID_INK: Placed 3 orders\nSQUID_INK [('SQUID_INK', 100, 1)]", "timestamp": 2000}
"""
    log_file = tmp_path / "trade.log"
    log_file.write_text(sample_log, encoding="utf-8")

    summary_text = summarize_trading_logs(str(log_file))

    assert "RAINFOREST_RESIN" in summary_text
    assert "SQUID_INK" in summary_text
    assert "Position Trends" in summary_text
    assert "Price History Collection" in summary_text
    assert "Order Patterns" in summary_text
    assert "Representative Examples" in summary_text


def test_format_summary_for_llm_branches():
    summary_data = {
        "position_trends": {
            "UP_PROD": [(0, 10), (100, 20)],
            "DOWN_PROD": [(0, 20), (100, 10)],
            "FLAT_PROD": [(0, 10), (100, 10)],
        },
        "price_history_collection": {
            "UP_PROD": (10, 10),
        },
        "order_patterns": {
            "UP_PROD_standard": 5,
        },
        "special_order_handling": {
            "DOWN_PROD_position_management": 3,
        },
        "representative_examples": {
            "standard_orders": "Order example text",
        },
    }

    result = format_summary_for_llm(summary_data)
    assert "UP_PROD: 10 → 20 (↑ 10)" in result
    assert "DOWN_PROD: 20 → 10 (↓ 10)" in result
    assert "FLAT_PROD: 10 → 10 (→ 0)" in result


def test_extract_representative_examples_none():
    examples = extract_representative_examples("nothing to see here")
    assert examples == {}
