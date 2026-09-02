from unittest.mock import patch

import pandas as pd
from src.rag.process_raw_trading_data import (
    calculate_trade_metrics,
    calculate_trading_metrics,
    discover_rounds,
    get_basket_info,
    main,
    process_round_data,
    process_trading_csv,
)


def test_get_basket_info():
    assert get_basket_info("data/round_2/picnicbasket1_data.csv") == "picnicbasket1"
    assert get_basket_info("data/round_1/resin.csv") is None


def test_calculate_trading_metrics():
    df = pd.DataFrame(
        {
            "mid_price": [100.0, 102.0, 101.0],
            "bid_price_1": [99.0, 101.0, 100.0],
            "ask_price_1": [101.0, 103.0, 102.0],
            "bid_volume_1": [10, 10, 10],
            "ask_volume_1": [10, 10, 10],
        }
    )
    metrics = calculate_trading_metrics(df)
    assert "volatility" in metrics
    assert "price_momentum" in metrics
    assert "volume_weighted_price" in metrics
    assert "avg_spread" in metrics
    assert metrics["market_depth"] == 1


def test_calculate_trade_metrics():
    df = pd.DataFrame(
        {
            "price": [100.0, 105.0, 102.0],
            "quantity": [5, 10, 5],
            "timestamp": [1000, 2000, 3000],
            "buyer": ["Alice", "Bob", "Alice"],
            "seller": ["Charlie", "Charlie", "David"],
        }
    )
    metrics = calculate_trade_metrics(df)
    assert metrics["total_volume"] == 20
    assert metrics["vwap"] == (100 * 5 + 105 * 10 + 102 * 5) / 20
    assert metrics["time_span"] == 2000
    assert metrics["unique_buyers"] == 2
    assert metrics["unique_sellers"] == 2
    assert metrics["price_change"] == 2.0


def test_process_trading_csv_market_data(tmp_path):
    csv_content = """day;timestamp;product;bid_price_1;bid_volume_1;ask_price_1;ask_volume_1;mid_price
1;1000;RAINFOREST_RESIN;99;10;101;10;100
1;2000;RAINFOREST_RESIN;100;10;102;10;101
"""
    csv_file = tmp_path / "round_1_prices_day_1.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    out_dir = tmp_path / "processed"
    docs = process_trading_csv(csv_file, out_dir)
    assert len(docs) == 1
    assert docs[0]["metadata"]["product"] == "RAINFOREST_RESIN"
    assert docs[0]["metadata"]["day"] == 1
    assert "Trading data for RAINFOREST_RESIN" in docs[0]["content"]


def test_process_trading_csv_trade_data(tmp_path):
    csv_content = """timestamp;buyer;seller;symbol;currency;price;quantity
1000;Alice;Bob;KELP;SEASHELLS;50;10
2000;Alice;Charlie;KELP;SEASHELLS;52;5
"""
    csv_file = tmp_path / "trades_round_1_day_1.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    out_dir = tmp_path / "processed"
    docs = process_trading_csv(csv_file, out_dir)
    assert len(docs) == 1
    assert docs[0]["metadata"]["product"] == "KELP"
    assert docs[0]["metadata"]["file_type"] == "trade"


def test_process_trading_csv_unrecognized_format(tmp_path):
    csv_file = tmp_path / "unknown.csv"
    csv_file.write_text("foo;bar\n1;2\n", encoding="utf-8")
    docs = process_trading_csv(csv_file, tmp_path / "out")
    assert docs == []


def test_discover_rounds_and_process_round_data(tmp_path):
    trading_dir = tmp_path / "trading_data"
    round1_dir = trading_dir / "round_1"
    raw_dir = round1_dir / "round_1_raw_trading_data"
    raw_dir.mkdir(parents=True)

    csv_content = """day;timestamp;product;bid_price_1;bid_volume_1;ask_price_1;ask_volume_1;mid_price
1;1000;RESIN;99;10;101;10;100
"""
    (raw_dir / "prices_day_1.csv").write_text(csv_content, encoding="utf-8")

    rounds = discover_rounds(str(trading_dir))
    assert rounds == ["round_1"]

    docs = process_round_data("round_1", str(trading_dir))
    assert len(docs) == 1
    assert (round1_dir / "round_1_processed_trading_data" / "all_round_1_trading_data.json").exists()


def test_main_cli(tmp_path):
    trading_dir = tmp_path / "trading_data"
    round1_dir = trading_dir / "round_1"
    raw_dir = round1_dir / "round_1_raw_trading_data"
    raw_dir.mkdir(parents=True)

    csv_content = """day;timestamp;product;bid_price_1;bid_volume_1;ask_price_1;ask_volume_1;mid_price
1;1000;RESIN;99;10;101;10;100
"""
    (raw_dir / "prices_day_1.csv").write_text(csv_content, encoding="utf-8")

    with patch("sys.argv", ["process_raw_trading_data", "--data_dir", str(trading_dir), "--rounds", "all"]):
        main()
