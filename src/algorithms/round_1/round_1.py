from collections import deque

import numpy as np
from datamodel import Order, OrderDepth, TradingState

RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"

RESIN_FAIR_VALUE = 10000
RESIN_SPREAD_CAPTURE = 2
RESIN_ORDER_VOLUME = 5
RESIN_POS_LIMIT = 50

SQUID_SMA_WINDOW = 20
SQUID_DEVIATION_THRESHOLD = 10
SQUID_ORDER_VOLUME = 3
SQUID_POS_LIMIT = 50


class Trader:
    def __init__(self):
        self.squid_ink_mid_price_history = deque(maxlen=SQUID_SMA_WINDOW)
        print("Trader initialized.")

    def calculate_mid_price(self, order_depth: OrderDepth) -> float | None:
        """Calculates the mid-price from order depth."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        print(f"\n--- Trader Run - Timestamp {state.timestamp} ---")
        result = {}

        if RAINFOREST_RESIN in state.order_depths:
            product = RAINFOREST_RESIN
            order_depth = state.order_depths[product]
            orders: list[Order] = []
            current_position = state.position.get(product, 0)

            buy_price = RESIN_FAIR_VALUE - RESIN_SPREAD_CAPTURE
            sell_price = RESIN_FAIR_VALUE + RESIN_SPREAD_CAPTURE

            print(
                f"  {product}: Fair Value={RESIN_FAIR_VALUE}, Target Buy={buy_price}, Target Sell={sell_price}, Position={current_position}"
            )

            if current_position < RESIN_POS_LIMIT:
                buy_volume = min(RESIN_ORDER_VOLUME, RESIN_POS_LIMIT - current_position)
                if buy_volume > 0:
                    orders.append(Order(product, buy_price, buy_volume))
                    print(f"    Placing BUY order: {buy_volume} units at {buy_price}")

            if current_position > -RESIN_POS_LIMIT:
                sell_volume = min(RESIN_ORDER_VOLUME, RESIN_POS_LIMIT + current_position)
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume))
                    print(f"    Placing SELL order: {sell_volume} units at {sell_price}")

            result[product] = orders
        else:
            print(f"  {RAINFOREST_RESIN}: No order depth data available.")

        if KELP in state.order_depths:
            print(
                f"  {KELP}: Insufficient information provided for a trading strategy. No orders placed."
            )

        if SQUID_INK in state.order_depths:
            product = SQUID_INK
            order_depth = state.order_depths[product]
            orders: list[Order] = []
            current_position = state.position.get(product, 0)

            mid_price = self.calculate_mid_price(order_depth)

            if mid_price is not None:
                self.squid_ink_mid_price_history.append(mid_price)
                print(f"  {product}: Mid Price={mid_price:.2f}, Position={current_position}")

                if len(self.squid_ink_mid_price_history) == SQUID_SMA_WINDOW:
                    sma = np.mean(self.squid_ink_mid_price_history)
                    deviation = mid_price - sma
                    print(f"    SMA({SQUID_SMA_WINDOW})={sma:.2f}, Deviation={deviation:.2f}")

                    if deviation < -SQUID_DEVIATION_THRESHOLD:
                        if current_position < SQUID_POS_LIMIT:
                            buy_volume = min(SQUID_ORDER_VOLUME, SQUID_POS_LIMIT - current_position)
                            if buy_volume > 0:
                                buy_order_price = round(sma - SQUID_DEVIATION_THRESHOLD)
                                orders.append(Order(product, buy_order_price, buy_volume))
                                print(
                                    f"    Mean Reversion BUY Signal: Price ({mid_price:.2f}) < SMA ({sma:.2f}) - Threshold ({SQUID_DEVIATION_THRESHOLD}). Placing BUY: {buy_volume} at {buy_order_price}"
                                )

                    elif (
                        deviation > SQUID_DEVIATION_THRESHOLD
                        and current_position > -SQUID_POS_LIMIT
                    ):
                        sell_volume = min(SQUID_ORDER_VOLUME, SQUID_POS_LIMIT + current_position)
                        if sell_volume > 0:
                            sell_order_price = round(sma + SQUID_DEVIATION_THRESHOLD)
                            orders.append(Order(product, sell_order_price, -sell_volume))
                            print(
                                f"    Mean Reversion SELL Signal: Price ({mid_price:.2f}) > SMA ({sma:.2f}) + Threshold ({SQUID_DEVIATION_THRESHOLD}). Placing SELL: {sell_volume} at {sell_order_price}"
                            )

                else:
                    print(
                        f"    Collecting price history for SMA calculation ({len(self.squid_ink_mid_price_history)}/{SQUID_SMA_WINDOW})."
                    )

            else:
                print(f"  {product}: Could not calculate mid-price (likely thin order book).")

            result[product] = orders
        else:
            print(f"  {SQUID_INK}: No order depth data available.")

        print(
            f"--- Orders Generated: {[(k, [(o.symbol, o.price, o.quantity) for o in v]) for k, v in result.items()]} ---"
        )
        traderData = ""
        conversions = 0
        return result, conversions, traderData
