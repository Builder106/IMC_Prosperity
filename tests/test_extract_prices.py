import csv
from src.utils.extract_prices import extract_price_data_to_csv


def test_extract_price_data_to_csv(tmp_path):
    sample_log = """Activities log:
day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
1;1000;RAINFOREST_RESIN;100;10;;;;;102;10;;;;;101.5;0.0
1;2000;RAINFOREST_RESIN;101;10;;;;;103;10;;;;;102.5;10.0
Trade History:
[{"timestamp": 1000, "buyer": "SUBMISSION", "seller": "", "symbol": "RAINFOREST_RESIN", "currency": "SEASHELLS", "price": 100, "quantity": 1}]
"""
    log_file = tmp_path / "trade.log"
    log_file.write_text(sample_log, encoding="utf-8")

    out_csv = tmp_path / "prices.csv"
    success = extract_price_data_to_csv(str(log_file), str(out_csv))
    assert success is True
    assert out_csv.exists()

    with open(out_csv, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)
        assert len(rows) == 3
        assert rows[0][0] == "day"
        assert rows[1][2] == "RAINFOREST_RESIN"


def test_extract_price_data_to_csv_no_activity_log(tmp_path):
    log_file = tmp_path / "empty.log"
    log_file.write_text("No activities here", encoding="utf-8")
    out_csv = tmp_path / "out.csv"
    success = extract_price_data_to_csv(str(log_file), str(out_csv))
    assert success is False


def test_extract_price_data_to_csv_os_error(tmp_path):
    success = extract_price_data_to_csv(str(tmp_path / "non_existent.log"), str(tmp_path / "out.csv"))
    assert success is False
