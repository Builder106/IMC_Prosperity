import json
import sys
from pathlib import Path

# Add src/algorithms/round_1 to sys.path so 'import datamodel' succeeds as expected by IMC Prosperity submission environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "algorithms" / "round_1"))

from datamodel import (  # type: ignore
    Observation,
    OrderDepth,
    TradingState,
)
from src.algorithms.round_2.trader import ASH, PEPPER, POSITION_LIMIT, Trader


def make_state(
    order_depths: dict[str, OrderDepth],
    position: dict[str, int] | None = None,
    trader_data: str = "",
) -> TradingState:
    return TradingState(
        traderData=trader_data,
        timestamp=0,
        listings={},
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position=position or {},
        observations=Observation({}, {}),
    )


def test_round2_trader_run():
    trader = Trader()

    depth_ash = OrderDepth()
    depth_ash.buy_orders = {9998: 5}
    depth_ash.sell_orders = {10002: -5}

    depth_pep = OrderDepth()
    depth_pep.buy_orders = {12998: 5}
    depth_pep.sell_orders = {13002: -5}

    state = TradingState(
        traderData=json.dumps({"fair_values": {ASH: 10000.0, PEPPER: 13000.0}}),
        timestamp=100,
        listings={},
        order_depths={
            ASH: depth_ash,
            PEPPER: depth_pep,
        },
        own_trades={},
        market_trades={},
        position={
            ASH: 0,
            PEPPER: 0,
        },
        observations=Observation({}, {}),
    )

    orders, conversions, _ = trader.run(state)
    assert ASH in orders
    assert PEPPER in orders
    assert conversions == 0
    assert trader.bid() == 1500


def test_round2_trader_edge_cases():
    trader = Trader()

    empty_depth = OrderDepth()
    empty_depth.buy_orders = {}
    empty_depth.sell_orders = {}

    buy_only_depth = OrderDepth()
    buy_only_depth.buy_orders = {100: 5}
    buy_only_depth.sell_orders = {}

    sell_only_depth = OrderDepth()
    sell_only_depth.buy_orders = {}
    sell_only_depth.sell_orders = {100: -5}

    assert trader._microprice(empty_depth) is None
    assert trader._microprice(buy_only_depth) == 100.0
    assert trader._microprice(sell_only_depth) == 100.0

    state = TradingState(
        traderData="invalid json",
        timestamp=100,
        listings={},
        order_depths={
            ASH: empty_depth,
            PEPPER: empty_depth,
        },
        own_trades={},
        market_trades={},
        position={
            ASH: 80,
            PEPPER: -80,
        },
        observations=Observation({}, {}),
    )

    orders, _, _ = trader.run(state)
    assert orders[ASH] == []
    assert orders[PEPPER] == []


def test_round2_first_tick_uses_observed_fair_value():
    depth = OrderDepth()
    depth.buy_orders = {10994: 10}
    depth.sell_orders = {11006: -10}

    orders, _, trader_data = Trader().run(make_state({PEPPER: depth}))

    fair_values = json.loads(trader_data)["fair_values"]
    assert fair_values[PEPPER] == 11000.0
    assert any(order.price >= 11003 for order in orders[PEPPER] if order.quantity < 0)


def test_round2_buy_orders_stay_within_position_limit():
    depth = OrderDepth()
    depth.buy_orders = {9999: 5}
    depth.sell_orders = {9994: -30}
    starting_position = POSITION_LIMIT[ASH] - 2

    orders, _, _ = Trader().run(make_state({ASH: depth}, position={ASH: starting_position}))

    buy_quantity = sum(order.quantity for order in orders[ASH] if order.quantity > 0)
    assert buy_quantity <= POSITION_LIMIT[ASH] - starting_position
