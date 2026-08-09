import json

from datamodel import Order, OrderDepth, TradingState

ASH = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

DEFAULT_FAIR_VALUE: dict[str, int] = {ASH: 10000, PEPPER: 13000}
POSITION_LIMIT: dict[str, int] = {ASH: 80, PEPPER: 80}
TAKE_EDGE: dict[str, int] = {ASH: 1, PEPPER: 2}
MAKE_EDGE: dict[str, int] = {ASH: 2, PEPPER: 3}
SKEW_COEFF: dict[str, float] = {ASH: 0.02, PEPPER: 0.03}
MAX_TAKE_QTY: dict[str, int] = {ASH: 12, PEPPER: 8}
MAX_MAKE_QTY: dict[str, int] = {ASH: 10, PEPPER: 8}
EWMA_ALPHA = 0.3


class Trader:
    def _load_fair_values(self, trader_data: str) -> dict[str, float]:
        if not trader_data:
            return {}
        try:
            parsed = json.loads(trader_data)
        except json.JSONDecodeError:
            return {}
        fairs = parsed.get("fair_values", {})
        result: dict[str, float] = {}
        for product, value in fairs.items():
            if product in POSITION_LIMIT:
                result[product] = float(value)
        return result

    def _microprice(self, order_depth: OrderDepth, fallback: float) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            bid_volume = max(order_depth.buy_orders[best_bid], 1)
            ask_volume = max(-order_depth.sell_orders[best_ask], 1)
            mid = (best_bid + best_ask) / 2
            micro = (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
            return 0.4 * mid + 0.6 * micro
        if order_depth.buy_orders:
            return float(max(order_depth.buy_orders))
        if order_depth.sell_orders:
            return float(min(order_depth.sell_orders))
        return fallback

    def _fair_value(self, product: str, order_depth: OrderDepth, prev_fair: float) -> float:
        observed = self._microprice(order_depth, prev_fair)
        return EWMA_ALPHA * observed + (1 - EWMA_ALPHA) * prev_fair

    def _take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: float,
        position: int,
    ) -> tuple[list[Order], int]:
        orders: list[Order] = []
        limit = POSITION_LIMIT[product]
        skew = position * SKEW_COEFF[product]
        buy_trigger = fair - TAKE_EDGE[product] - skew
        sell_trigger = fair + TAKE_EDGE[product] - skew
        buy_capacity = max(0, limit - position)
        sell_capacity = max(0, limit + position)

        for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
            available = -ask_qty
            if available <= 0:
                continue
            if ask_price > buy_trigger or buy_capacity <= 0:
                break
            qty = min(available, buy_capacity, MAX_TAKE_QTY[product])
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty
                position += qty

        for bid_price, bid_qty in sorted(order_depth.buy_orders.items(), reverse=True):
            available = bid_qty
            if available <= 0:
                continue
            if bid_price < sell_trigger or sell_capacity <= 0:
                break
            qty = min(available, sell_capacity, MAX_TAKE_QTY[product])
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty
                position -= qty

        return orders, position

    def _make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: float,
        position: int,
    ) -> list[Order]:
        orders: list[Order] = []
        limit = POSITION_LIMIT[product]
        skew = position * SKEW_COEFF[product]
        bid_quote = int(fair - MAKE_EDGE[product] - skew)
        ask_quote = int(fair + MAKE_EDGE[product] - skew)

        if order_depth.buy_orders:
            bid_quote = min(bid_quote, max(order_depth.buy_orders) + 1)
        if order_depth.sell_orders:
            ask_quote = max(ask_quote, min(order_depth.sell_orders) - 1)

        buy_capacity = max(0, limit - position)
        sell_capacity = max(0, limit + position)

        if buy_capacity > 0 and (not order_depth.sell_orders or bid_quote < min(order_depth.sell_orders)):
            orders.append(Order(product, bid_quote, min(buy_capacity, MAX_MAKE_QTY[product])))
        if sell_capacity > 0 and (not order_depth.buy_orders or ask_quote > max(order_depth.buy_orders)):
            orders.append(Order(product, ask_quote, -min(sell_capacity, MAX_MAKE_QTY[product])))

        return orders

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        previous_fairs = self._load_fair_values(state.traderData)
        next_fairs: dict[str, float] = {}

        for product in (ASH, PEPPER):
            if product not in state.order_depths:
                continue

            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)
            prev_fair = previous_fairs.get(product, float(DEFAULT_FAIR_VALUE[product]))
            fair = self._fair_value(product, order_depth, prev_fair)
            next_fairs[product] = fair
            take_orders, updated_position = self._take_orders(product, order_depth, fair, position)
            make_orders = self._make_orders(product, order_depth, fair, updated_position)
            orders = take_orders + make_orders

            result[product] = orders

        trader_data = json.dumps({"fair_values": next_fairs}, separators=(",", ":"))
        return result, 0, trader_data
