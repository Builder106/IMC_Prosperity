import json
import sys
from pathlib import Path

# Add src/algorithms/round_1 to sys.path so 'import datamodel' succeeds as expected by IMC Prosperity submission environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "algorithms" / "round_1"))

from datamodel import (  # type: ignore
    ConversionObservation,
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Trade,
    TradingState,
)
from src.algorithms.round_1.trader import ASH, PEPPER, POSITION_LIMIT, Trader


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


def test_datamodel_classes():
    order = Order(ASH, 100, 5)
    assert str(order) == f"({ASH}, 100, 5)"
    assert repr(order) == f"({ASH}, 100, 5)"

    trade = Trade(ASH, 100, 5, "buyer", "seller", 1000)
    assert str(trade) == f"({ASH}, buyer << seller, 100, 5, 1000)"
    assert repr(trade) == f"({ASH}, buyer << seller, 100, 5, 1000)"

    listing = Listing(ASH, "Ash", "SEASHELLS")
    assert listing.symbol == ASH

    obs = Observation({}, {})
    assert obs.plainValueObservations == {}

    conv_obs = ConversionObservation(1.0, 1.1, 0.1, 0.2, 0.05, 0.01, 0.0)
    assert conv_obs.bidPrice == 1.0

    state = TradingState(
        traderData="",
        timestamp=1000,
        listings={ASH: listing},
        order_depths={ASH: OrderDepth()},
        own_trades={},
        market_trades={},
        position={},
        observations=obs,
    )
    json_str = state.toJSON()
    assert '"timestamp": 1000' in json_str

    encoded = json.dumps(order, cls=ProsperityEncoder)
    assert f'"symbol": "{ASH}"' in encoded


def test_trader_round1_full_run():
    trader = Trader()

    depth_ash = OrderDepth()
    depth_ash.buy_orders = {9998: 10, 9995: 5}
    depth_ash.sell_orders = {10002: -10, 10005: -5}

    depth_pep = OrderDepth()
    depth_pep.buy_orders = {12998: 10}
    depth_pep.sell_orders = {13002: -10}

    state = TradingState(
        traderData="",
        timestamp=100,
        listings={},
        order_depths={ASH: depth_ash, PEPPER: depth_pep},
        own_trades={},
        market_trades={},
        position={ASH: 0, PEPPER: 0},
        observations=Observation({}, {}),
    )

    orders, conversions, trader_data = trader.run(state)
    assert ASH in orders
    assert PEPPER in orders
    assert conversions == 0
    assert "fair_values" in trader_data


def test_trader_round1_edge_cases():
    trader = Trader()

    empty_depth = OrderDepth()
    empty_depth.buy_orders = {}
    empty_depth.sell_orders = {}

    # When position is at limit (+80 for ASH, -80 for PEPPER), trader can still make orders to reduce position
    state = TradingState(
        traderData="invalid json",
        timestamp=100,
        listings={},
        order_depths={ASH: empty_depth, PEPPER: empty_depth},
        own_trades={},
        market_trades={},
        position={ASH: 80, PEPPER: -80},
        observations=Observation({}, {}),
    )

    orders, _, _ = trader.run(state)
    assert ASH in orders
    assert PEPPER in orders
    assert orders[ASH]
    assert all(order.quantity < 0 for order in orders[ASH])
    assert orders[PEPPER]
    assert all(order.quantity > 0 for order in orders[PEPPER])

    buy_only_depth = OrderDepth()
    buy_only_depth.buy_orders = {10000: 5}
    buy_only_depth.sell_orders = {}
    assert trader._microprice(buy_only_depth, 0.0) == 10000.0

    sell_only_depth = OrderDepth()
    sell_only_depth.buy_orders = {}
    sell_only_depth.sell_orders = {10000: -5}
    assert trader._microprice(sell_only_depth, 0.0) == 10000.0
    assert trader._microprice(empty_depth, 0.0) == 0.0


def test_trader_round1_take_orders():
    trader = Trader()
    depth = OrderDepth()
    # Mispriced orders that trader should take for ASH (fair is around 10000)
    depth.buy_orders = {10005: 10}  # Bid is higher than fair value 10000 -> sell into it
    depth.sell_orders = {9995: -10}  # Ask is lower than fair value 10000 -> buy it

    state = TradingState(
        traderData='{"fair_values": {"ASH_COATED_OSMIUM": 10000.0}}',
        timestamp=100,
        listings={},
        order_depths={ASH: depth},
        own_trades={},
        market_trades={},
        position={ASH: 0},
        observations=Observation({}, {}),
    )

    orders, _, _ = trader.run(state)
    ash_orders = orders[ASH]
    assert any(o.price == 9995 and o.quantity > 0 for o in ash_orders)
    assert any(o.price == 10005 and o.quantity < 0 for o in ash_orders)


def test_round1_buy_orders_stay_within_position_limit():
    depth = OrderDepth()
    depth.buy_orders = {9999: 5}
    depth.sell_orders = {9995: -30}
    starting_position = POSITION_LIMIT[ASH] - 5

    orders, _, _ = Trader().run(make_state({ASH: depth}, position={ASH: starting_position}))

    buy_quantity = sum(order.quantity for order in orders[ASH] if order.quantity > 0)
    assert buy_quantity <= POSITION_LIMIT[ASH] - starting_position


def test_round1_long_inventory_lowers_sell_quote():
    depth = OrderDepth()
    depth.buy_orders = {12998: 10}
    depth.sell_orders = {13002: -10}
    trader_data = json.dumps({"fair_values": {PEPPER: 13000.0}})

    flat_orders, _, _ = Trader().run(make_state({PEPPER: depth}, trader_data=trader_data))
    long_orders, _, _ = Trader().run(
        make_state({PEPPER: depth}, position={PEPPER: 70}, trader_data=trader_data)
    )

    flat_sell_price = min(order.price for order in flat_orders[PEPPER] if order.quantity < 0)
    long_sell_price = min(order.price for order in long_orders[PEPPER] if order.quantity < 0)
    assert long_sell_price < flat_sell_price
