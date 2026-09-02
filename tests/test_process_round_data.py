import numpy as np
import pandas as pd
from src.utils.process_round_trading_data_alpha import (
    calculate_metrics as alpha_metrics,
)
from src.utils.process_round_trading_data_stable_prices_trades import (
    calculate_basic_metrics,
    calculate_hurst_exponent,
    calculate_metrics,
    calculate_order_book_metrics,
    calculate_price_impact_metrics,
    calculate_spread_metrics,
    calculate_statistical_metrics,
    calculate_time_based_metrics,
    process_prices_and_trades,
    process_round_csv,
)


def test_calculate_hurst_exponent():
    # Constant or short series returns nan
    assert np.isnan(calculate_hurst_exponent([1, 2, 3]))

    # Long random walk series
    np.random.seed(42)
    ts = np.cumsum(np.random.randn(100)) + 100
    h = calculate_hurst_exponent(ts)
    assert not np.isnan(h)


def test_calculate_stable_metrics():
    df = pd.DataFrame(
        {
            "mid_price": [100.0, 102.0, 101.0, 103.0, 102.0],
            "price": [100.0, 102.0, 101.0, 103.0, 102.0],
            "quantity": [10, 20, 15, 30, 25],
            "bid_price_1": [99.0, 101.0, 100.0, 102.0, 101.0],
            "ask_price_1": [101.0, 103.0, 102.0, 104.0, 103.0],
            "bid_volume_1": [10, 10, 10, 10, 10],
            "ask_volume_1": [10, 10, 10, 10, 10],
            "timestamp": [1000, 2000, 3000, 4000, 5000],
        }
    )

    basic = calculate_basic_metrics(df)
    assert "volatility" in basic
    assert "total_volume" in basic

    spread = calculate_spread_metrics(df)
    assert "avg_spread" in spread
    assert "avg_relative_spread" in spread

    ob = calculate_order_book_metrics(df)
    assert "avg_total_bid_volume" in ob

    impact = calculate_price_impact_metrics(df)
    assert isinstance(impact, dict)

    time_m = calculate_time_based_metrics(df)
    assert "time_span" in time_m

    stat_m = calculate_statistical_metrics(df)
    assert "returns_mean" in stat_m

    all_m = calculate_metrics(df)
    assert "volatility" in all_m


def test_process_prices_and_trades(tmp_path):
    prices_csv = tmp_path / "prices.csv"
    prices_csv.write_text(
        "day;timestamp;product;bid_price_1;bid_volume_1;ask_price_1;ask_volume_1;mid_price\n"
        "1;1000;RESIN;99;10;101;10;100\n"
        "1;2000;RESIN;100;10;102;10;101\n",
        encoding="utf-8",
    )

    trades_csv = tmp_path / "trades.csv"
    trades_csv.write_text(
        "timestamp;buyer;seller;symbol;currency;price;quantity\n1000;B;S;RESIN;SEASHELLS;100;5\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    docs = process_prices_and_trades(str(prices_csv), str(trades_csv), str(out_dir))
    assert len(docs) > 0
    assert (out_dir / "all_merged_trading_data.json").exists()


def test_process_round_csv(tmp_path):
    csv_file = tmp_path / "single_day_1.csv"
    csv_file.write_text(
        "day;timestamp;product;bid_price_1;bid_volume_1;ask_price_1;ask_volume_1;mid_price\n"
        "1;1000;RESIN;99;10;101;10;100\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out_single"
    docs = process_round_csv(str(csv_file), str(out_dir))
    assert len(docs) > 0


def test_alpha_metrics():
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "mid_price": [100.0 + np.sin(i / 5.0) * 5.0 for i in range(150)],
            "bid_price_1": [99.0 + np.sin(i / 5.0) * 5.0 for i in range(150)],
            "ask_price_1": [101.0 + np.sin(i / 5.0) * 5.0 for i in range(150)],
            "bid_volume_1": [10] * 150,
            "ask_volume_1": [10] * 150,
            "timestamp": [i * 100 for i in range(150)],
        }
    )
    metrics = alpha_metrics(df)
    assert "volatility" in metrics
