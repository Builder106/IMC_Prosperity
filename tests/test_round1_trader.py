import importlib.util
import sys
import types
from pathlib import Path

ROUND1_DIR = Path(__file__).resolve().parents[1] / "src" / "algorithms" / "round_1"


def _load_module(module_name: str, file_name: str):
    module_path = ROUND1_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if "jsonpickle" not in sys.modules:
    jsonpickle_stub = types.ModuleType("jsonpickle")
    setattr(jsonpickle_stub, "encode", lambda value: str(value))
    sys.modules["jsonpickle"] = jsonpickle_stub

datamodel = _load_module("round1_datamodel", "datamodel.py")
sys.modules["datamodel"] = datamodel
trader_module = _load_module("round1_trader", "trader.py")

OrderDepth = datamodel.OrderDepth
Observation = datamodel.Observation
TradingState = datamodel.TradingState
Trader = trader_module.Trader
ASH = trader_module.ASH
PEPPER = trader_module.PEPPER


def make_state(order_depths, position=None, trader_data=""):
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


def test_buys_underpriced_level():
    depth = OrderDepth()
    depth.buy_orders = {9998: 5}
    depth.sell_orders = {9996: -7}
    state = make_state({ASH: depth})
    trader = Trader()

    result, _, trader_data = trader.run(state)

    orders = result[ASH]
    assert any(order.quantity > 0 for order in orders)
    assert trader_data


def test_respects_position_limit_for_buys():
    depth = OrderDepth()
    depth.buy_orders = {9999: 5}
    depth.sell_orders = {9995: -30}
    state = make_state({ASH: depth}, position={ASH: 75})
    trader = Trader()

    result, _, _ = trader.run(state)

    buy_qty = sum(order.quantity for order in result[ASH] if order.quantity > 0)
    assert buy_qty <= 5


def test_skew_encourages_selling_when_long():
    depth = OrderDepth()
    depth.buy_orders = {13000: 10}
    depth.sell_orders = {13002: -10}
    state = make_state({PEPPER: depth}, position={PEPPER: 70})
    trader = Trader()

    result, _, _ = trader.run(state)

    assert any(order.quantity < 0 for order in result[PEPPER])
