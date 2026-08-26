import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUND1_DIR = ROOT / "src" / "algorithms" / "round_1"
ROUND2_DIR = ROOT / "src" / "algorithms" / "round_2"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if "jsonpickle" not in sys.modules:
    jsonpickle_stub = types.ModuleType("jsonpickle")
    setattr(jsonpickle_stub, "encode", lambda value: str(value))
    sys.modules["jsonpickle"] = jsonpickle_stub

datamodel = _load_module("round2_datamodel_base", ROUND1_DIR / "datamodel.py")
sys.modules["datamodel"] = datamodel
trader_module = _load_module("round2_trader_module", ROUND2_DIR / "trader.py")

OrderDepth = datamodel.OrderDepth
Observation = datamodel.Observation
TradingState = datamodel.TradingState
Trader = trader_module.Trader
ASH = trader_module.ASH
PEPPER = trader_module.PEPPER
MARKET_ACCESS_FEE_BID = trader_module.MARKET_ACCESS_FEE_BID


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


def test_has_round2_bid_method():
    trader = Trader()
    assert trader.bid() == MARKET_ACCESS_FEE_BID


def test_first_tick_uses_observed_fair_without_bias():
    depth = OrderDepth()
    depth.buy_orders = {10994: 10}
    depth.sell_orders = {11006: -10}
    state = make_state({PEPPER: depth})
    trader = Trader()

    result, _, _ = trader.run(state)

    assert any(order.price >= 11003 for order in result[PEPPER] if order.quantity < 0)


def test_respects_position_limit_for_buys():
    depth = OrderDepth()
    depth.buy_orders = {9999: 5}
    depth.sell_orders = {9994: -30}
    state = make_state({ASH: depth}, position={ASH: 78})
    trader = Trader()

    result, _, _ = trader.run(state)

    buy_qty = sum(order.quantity for order in result[ASH] if order.quantity > 0)
    assert buy_qty <= 2
