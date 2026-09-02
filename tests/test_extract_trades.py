import csv

from src.utils.extract_trades import extract_trade_history_to_csv


def test_extract_trade_history_to_csv_json_block(tmp_path):
    sample_log = """
Trade History:
[
  {
    "timestamp": 1000,
    "buyer": "BUYER_1",
    "seller": "SELLER_1",
    "symbol": "RAINFOREST_RESIN",
    "currency": "SEASHELLS",
    "price": 10000,
    "quantity": 2
  }
]
"""
    log_file = tmp_path / "trade.log"
    log_file.write_text(sample_log, encoding="utf-8")
    out_csv = tmp_path / "trades.csv"

    extract_trade_history_to_csv(str(log_file), str(out_csv))
    assert out_csv.exists()

    with open(out_csv, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0] == [
            "timestamp",
            "buyer",
            "seller",
            "symbol",
            "currency",
            "price",
            "quantity",
        ]
        assert rows[1] == [
            "1000",
            "BUYER_1",
            "SELLER_1",
            "RAINFOREST_RESIN",
            "SEASHELLS",
            "10000",
            "2",
        ]


def test_extract_trade_history_to_csv_empty_trade_data(tmp_path):
    sample_log = "Trade History:\n[]\n"
    log_file = tmp_path / "empty_trades.log"
    log_file.write_text(sample_log, encoding="utf-8")
    out_csv = tmp_path / "empty_trades.csv"

    extract_trade_history_to_csv(str(log_file), str(out_csv))
    assert out_csv.exists()


def test_extract_trade_history_to_csv_invalid_json(tmp_path):
    sample_log = "Trade History:\n[invalid json]\n"
    log_file = tmp_path / "invalid.log"
    log_file.write_text(sample_log, encoding="utf-8")
    out_csv = tmp_path / "invalid.csv"

    extract_trade_history_to_csv(str(log_file), str(out_csv))
    assert not out_csv.exists()


def test_extract_trade_history_to_csv_missing_file(tmp_path):
    out_csv = tmp_path / "missing.csv"
    extract_trade_history_to_csv(str(tmp_path / "missing.log"), str(out_csv))
    assert not out_csv.exists()
