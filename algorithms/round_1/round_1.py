import json
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order
from collections import deque
import numpy as np

# Define product names as constants
RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"

# --- Strategy Parameters ---

# Rainforest Resin: Stable around 10000
RESIN_FAIR_VALUE = 10000
RESIN_SPREAD_CAPTURE = 2 # Place orders this far from fair value (buy below, sell above)
RESIN_ORDER_VOLUME = 5   # Example order volume
RESIN_POS_LIMIT = 50     # Placeholder position limit

# Squid Ink: Mean Reversion
SQUID_SMA_WINDOW = 20    # Look at the last 20 mid-prices for the average (needs tuning)
SQUID_DEVIATION_THRESHOLD = 10 # Trigger trade if price deviates by this much from SMA (needs tuning)
SQUID_ORDER_VOLUME = 3   # Example order volume
SQUID_POS_LIMIT = 50     # Placeholder position limit

# Kelp: Insufficient information to form a strategy based *only* on provided text.

class Trader:

    def __init__(self):
        # History for Squid Ink Simple Moving Average (SMA)
        self.squid_ink_mid_price_history = deque(maxlen=SQUID_SMA_WINDOW)
        print("Trader initialized.")

    def calculate_mid_price(self, order_depth: OrderDepth) -> float | None:
        """Calculates the mid-price from order depth."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None # Not enough liquidity to determine mid-price reliably
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Takes TradingState and returns orders dictionary.
        Only trades based on retrieved information.
        """
        print(f"\n--- Trader Run - Timestamp {state.timestamp} ---")
        result = {} # Orders to place for each product

        # --- 1. Rainforest Resin Strategy ---
        # Insight: Stable around 10000. Low volatility. Aim to capture spread.
        # Strategy: Place buy orders slightly below 10000 and sell orders slightly above 10000.
        if RAINFOREST_RESIN in state.order_depths:
            product = RAINFOREST_RESIN
            order_depth = state.order_depths[product]
            orders: list[Order] = []
            current_position = state.position.get(product, 0)

            # Define buy and sell prices around the perceived fair value
            buy_price = RESIN_FAIR_VALUE - RESIN_SPREAD_CAPTURE
            sell_price = RESIN_FAIR_VALUE + RESIN_SPREAD_CAPTURE

            print(f"  {product}: Fair Value={RESIN_FAIR_VALUE}, Target Buy={buy_price}, Target Sell={sell_price}, Position={current_position}")

            # Place buy order if below position limit
            if current_position < RESIN_POS_LIMIT:
                buy_volume = min(RESIN_ORDER_VOLUME, RESIN_POS_LIMIT - current_position) # Don't exceed limit
                if buy_volume > 0:
                    orders.append(Order(product, buy_price, buy_volume))
                    print(f"    Placing BUY order: {buy_volume} units at {buy_price}")

            # Place sell order if above negative position limit
            if current_position > -RESIN_POS_LIMIT:
                sell_volume = min(RESIN_ORDER_VOLUME, RESIN_POS_LIMIT + current_position) # Don't exceed limit (volume is positive)
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume)) # Sell order uses negative quantity
                    print(f"    Placing SELL order: {sell_volume} units at {sell_price}")

            result[product] = orders
        else:
             print(f"  {RAINFOREST_RESIN}: No order depth data available.")


        # --- 2. Kelp Strategy ---
        # Insight: Goes up and down, volatile. No specific pattern or quantitative data provided.
        # Strategy: Based *only* on the provided text, there isn't enough information
        #           to formulate a specific trading strategy (e.g., trend following vs. mean reversion).
        #           Therefore, we will not trade Kelp with this algorithm.
        if KELP in state.order_depths:
             print(f"  {KELP}: Insufficient information provided for a trading strategy. No orders placed.")
        # No orders generated for KELP


        # --- 3. Squid Ink Strategy ---
        # Insight: Volatile, large swings, tendency to revert short-term swings.
        # Strategy: Mean Reversion. Buy when price drops significantly below recent average,
        #           sell when price rises significantly above recent average.
        if SQUID_INK in state.order_depths:
            product = SQUID_INK
            order_depth = state.order_depths[product]
            orders: list[Order] = []
            current_position = state.position.get(product, 0)

            mid_price = self.calculate_mid_price(order_depth)

            if mid_price is not None:
                self.squid_ink_mid_price_history.append(mid_price)
                print(f"  {product}: Mid Price={mid_price:.2f}, Position={current_position}")

                # Need enough history to calculate SMA
                if len(self.squid_ink_mid_price_history) == SQUID_SMA_WINDOW:
                    sma = np.mean(self.squid_ink_mid_price_history)
                    deviation = mid_price - sma
                    print(f"    SMA({SQUID_SMA_WINDOW})={sma:.2f}, Deviation={deviation:.2f}")

                    # Buy Signal: Price significantly below SMA
                    if deviation < -SQUID_DEVIATION_THRESHOLD:
                        # Buy only if below position limit
                        if current_position < SQUID_POS_LIMIT:
                            buy_volume = min(SQUID_ORDER_VOLUME, SQUID_POS_LIMIT - current_position)
                            if buy_volume > 0:
                                # Place buy order slightly above mid-price to increase fill chance, or at calculated threshold
                                buy_order_price = int(round(sma - SQUID_DEVIATION_THRESHOLD)) # Or potentially mid_price + 1
                                orders.append(Order(product, buy_order_price, buy_volume))
                                print(f"    Mean Reversion BUY Signal: Price ({mid_price:.2f}) < SMA ({sma:.2f}) - Threshold ({SQUID_DEVIATION_THRESHOLD}). Placing BUY: {buy_volume} at {buy_order_price}")

                    # Sell Signal: Price significantly above SMA
                    elif deviation > SQUID_DEVIATION_THRESHOLD:
                         # Sell only if above negative position limit
                        if current_position > -SQUID_POS_LIMIT:
                            sell_volume = min(SQUID_ORDER_VOLUME, SQUID_POS_LIMIT + current_position)
                            if sell_volume > 0:
                                # Place sell order slightly below mid-price, or at calculated threshold
                                sell_order_price = int(round(sma + SQUID_DEVIATION_THRESHOLD)) # Or potentially mid_price - 1
                                orders.append(Order(product, sell_order_price, -sell_volume))
                                print(f"    Mean Reversion SELL Signal: Price ({mid_price:.2f}) > SMA ({sma:.2f}) + Threshold ({SQUID_DEVIATION_THRESHOLD}). Placing SELL: {sell_volume} at {sell_order_price}")

                    # Optional: Logic to close positions when price reverts closer to SMA
                    # (Could add logic here to place smaller orders in the opposite direction
                    # if holding a position and the price is between +/- threshold around SMA)

                else:
                    print(f"    Collecting price history for SMA calculation ({len(self.squid_ink_mid_price_history)}/{SQUID_SMA_WINDOW}).")

            else:
                 print(f"  {product}: Could not calculate mid-price (likely thin order book).")

            result[product] = orders
        else:
            print(f"  {SQUID_INK}: No order depth data available.")


        # --- Return orders ---
        print(f"--- Orders Generated: {[(k, [(o.symbol, o.price, o.quantity) for o in v]) for k, v in result.items()]} ---")        # Trader data can be used to store state if needed, but SMA history is kept in self
        traderData = ""
        conversions = 0
        return result, conversions, traderData