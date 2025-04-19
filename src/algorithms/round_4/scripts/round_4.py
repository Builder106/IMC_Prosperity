import math
import jsonpickle
from datamodel import Listing, Order, OrderDepth, Trade, TradingState, Symbol, Observation, ProsperityEncoder
from typing import Any, Optional, Dict, List
from collections import deque, defaultdict

# Position limits per product
POSITION_LIMITS = {
    'RAINFOREST_RESIN': 50, 'KELP': 50, 'SQUID_INK': 50, 'CROISSANTS': 250,
    'JAMS': 350, 'DJEMBES': 60, 'PICNIC_BASKET1': 60, 'PICNIC_BASKET2': 100,
    'VOLCANIC_ROCK': 400, 'VOLCANIC_ROCK_VOUCHER_9500': 200,
    'VOLCANIC_ROCK_VOUCHER_9750': 200, 'VOLCANIC_ROCK_VOUCHER_10000': 200,
    'VOLCANIC_ROCK_VOUCHER_10250': 200, 'VOLCANIC_ROCK_VOUCHER_10500': 200,
    'MAGNIFICENT_MACARONS': 75,
}

# Static parameters (used as fallbacks or initial values)
FAIR_VALUES = {
    'RAINFOREST_RESIN': 10000, 'CROISSANTS': 4266, 'JAMS': 6532,
    'DJEMBES': 13455, 'PICNIC_BASKET1': 58847, 'PICNIC_BASKET2': 30098,
    'VOLCANIC_ROCK': 10157
}
PARAMS = {
    'RAINFOREST_RESIN': {'volatility': 2.97, 'spread_capture': 5.5, 'mean_reversion_strength': 0.4, 'sma_window': 12, 'position_scale': 1.5},
    'CROISSANTS': {'volatility': 1.95, 'spread_capture': 1.0, 'mean_reversion_strength': 0.4, 'sma_window': 10, 'position_scale': 0.8},
    'JAMS': {'volatility': 5.89, 'spread_capture': 1.5, 'mean_reversion_strength': 0.5, 'sma_window': 12, 'position_scale': 1.5},
    'DJEMBES': {'volatility': 26.92, 'spread_capture': 1.5, 'mean_reversion_strength': 0.4, 'sma_window': 15, 'position_scale': 0.6},
    'PICNIC_BASKET1': {'volatility': 48.59, 'spread_capture': 5.5, 'mean_reversion_strength': 0.2, 'sma_window': 10, 'position_scale': 0.5},
    'PICNIC_BASKET2': {'volatility': 12.57, 'spread_capture': 3.5, 'mean_reversion_strength': 0.3, 'sma_window': 10, 'position_scale': 0.4},
    'VOLCANIC_ROCK': {'volatility': 95, 'spread_capture': 5.0, 'mean_reversion_strength': 0.3, 'sma_window': 15, 'position_scale': 1.0},
}
BASKET1_COMPONENTS = {'DJEMBES': 1, 'JAMS': 3, 'CROISSANTS': 6}
BASKET2_COMPONENTS = {'JAMS': 2, 'CROISSANTS': 4}

class RollingStats:
    def __init__(self, window=20):
        self.window = window
        self.values = deque(maxlen=window)
    def update(self, value):
        self.values.append(value)
    def mean(self):
        return sum(self.values) / len(self.values) if self.values else None
    def std(self):
        if len(self.values) < 2: return None
        m = self.mean()
        if m is None: return None
        variance = sum((x-m)**2 for x in self.values) / len(self.values)
        return math.sqrt(variance) if variance >= 0 else None

class Logger:
    # Simplified Logger
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        try:
            log_line = sep.join(map(str, objects)) + end
            if len(self.logs) + len(log_line) <= self.max_log_length - 100: # Keep buffer
                 self.logs += log_line
        except Exception: pass # Ignore logging errors

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        try:
            log_output = {
                "timestamp": state.timestamp,
                "traderData": self.truncate(trader_data, 1000),
                "logs": self.truncate(self.logs, 2000),
            }
            print(jsonpickle.encode(log_output, unpicklable=False))
        except Exception as e:
            print(f"CRITICAL ERROR DURING LOGGING: {e}")
        finally:
             self.logs = ""

    def to_json(self, value: Any) -> str:
        return jsonpickle.encode(value, unpicklable=False)

    def truncate(self, value: Any, max_length: int) -> str:
        s_value = str(value) # Convert to string first
        if len(s_value) <= max_length: return s_value
        return s_value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        self.rolling = defaultdict(lambda: RollingStats(20))
        self.price_history = defaultdict(lambda: deque(maxlen=40))  # Longer history for advanced signals
        self.spread_history = defaultdict(lambda: deque(maxlen=20))
        self.max_single_trade = 10
        self.last_mid_prices = {}
        logger.print("Trader initialized with advanced strategies.")

    def get_mid(self, order_depth: OrderDepth) -> Optional[float]:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2.0
        return None

    def calculate_order_imbalance(self, order_depth: OrderDepth) -> float:
        buy_vol = sum(order_depth.buy_orders.values())
        sell_vol = -sum(order_depth.sell_orders.values())
        total = buy_vol + sell_vol
        return (buy_vol - sell_vol) / total if total > 0 else 0

    def determine_trend(self, prices: deque, lookback: int = 5) -> float:
        if len(prices) < lookback:
            return 0
        x = list(range(lookback))
        y = list(prices)[-lookback:]
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0
        return slope / (y_mean if y_mean != 0 else 1)

    def get_trade_signal(self, product: Symbol, mid: float, params: dict) -> float:
        # Mean reversion signal
        sma_window = params['sma_window']
        if len(self.price_history[product]) < sma_window:
            return 0
        sma = sum(list(self.price_history[product])[-sma_window:]) / sma_window
        deviation = mid - sma
        normalized = deviation / (params['volatility'] if params['volatility'] else 1)
        signal = -math.tanh(normalized * params['mean_reversion_strength'])
        # Trend adjustment
        trend = self.determine_trend(self.price_history[product], lookback=5)
        signal += 0.3 * trend
        # Order book imbalance
        # (imbalance is added in market_make_and_mean_revert)
        return max(-1, min(1, signal))

    def calculate_position_size(self, product: Symbol, signal: float, pos: int, params: dict) -> int:
        max_pos = POSITION_LIMITS[product]
        scale = params.get('position_scale', 1.0)
        # Reduce size as we approach position limits
        headroom = max_pos - abs(pos)
        base_size = int(max(1, min(self.max_single_trade, headroom * abs(signal) * scale)))
        return int(base_size * (1 if signal > 0 else -1))

    def market_make_and_mean_revert(self, product: Symbol, order_depth: OrderDepth, pos: int, state: TradingState) -> List[Order]:
        params = self.get_dynamic_params(product, order_depth)
        mid = self.get_mid(order_depth)
        # --- Ensure price history is always updated ---
        if mid is not None:
            self.price_history[product].append(mid)
        orders = []
        if mid is None:
            return orders
        # Market making: place passive orders near fair value
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        spread_capture = params['spread_capture']
        fair = params['fair']
        # --- Loosen entry conditions ---
        if pos < POSITION_LIMITS[product] and best_ask is not None:
            buy_price = math.floor(fair - spread_capture / 2)
            # Allow buy if best_ask <= buy_price (was <)
            if best_ask <= buy_price:
                buy_vol = min(-order_depth.sell_orders[best_ask], POSITION_LIMITS[product] - pos, self.max_single_trade)
                if buy_vol > 0:
                    orders.append(Order(product, best_ask, buy_vol))
        if pos > -POSITION_LIMITS[product] and best_bid is not None:
            sell_price = math.ceil(fair + spread_capture / 2)
            # Allow sell if best_bid >= sell_price (was >)
            if best_bid >= sell_price:
                sell_vol = min(order_depth.buy_orders[best_bid], pos + POSITION_LIMITS[product], self.max_single_trade)
                if sell_vol > 0:
                    orders.append(Order(product, best_bid, -sell_vol))
        # Advanced mean reversion/momentum
        signal = self.get_trade_signal(product, mid, params)
        if abs(signal) > 0.05:  # Loosened from 0.1
            pos_size = self.calculate_position_size(product, signal, pos, params)
            if pos_size > 0 and pos < POSITION_LIMITS[product]:
                price = int(mid - params['spread_capture'] * 0.5)
                orders.append(Order(product, price, min(pos_size, POSITION_LIMITS[product] - pos)))
            elif pos_size < 0 and pos > -POSITION_LIMITS[product]:
                price = int(mid + params['spread_capture'] * 0.5)
                orders.append(Order(product, price, max(pos_size, -POSITION_LIMITS[product] - pos)))
        # Order book imbalance adjustment
        imbalance = self.calculate_order_imbalance(order_depth)
        if abs(imbalance) > 0.25:  # Loosened from 0.3
            # If strong imbalance, take more aggressive action
            if imbalance > 0.25 and pos < POSITION_LIMITS[product] and best_ask is not None:
                buy_vol = min(-order_depth.sell_orders[best_ask], POSITION_LIMITS[product] - pos, self.max_single_trade)
                if buy_vol > 0:
                    orders.append(Order(product, best_ask, buy_vol))
            elif imbalance < -0.25 and pos > -POSITION_LIMITS[product] and best_bid is not None:
                sell_vol = min(order_depth.buy_orders[best_bid], pos + POSITION_LIMITS[product], self.max_single_trade)
                if sell_vol > 0:
                    orders.append(Order(product, best_bid, -sell_vol))
        return orders

    def basket_arbitrage(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        orders = defaultdict(list)
        # Basket 1
        if all(p in state.order_depths for p in ['PICNIC_BASKET1', 'JAMS', 'CROISSANTS', 'DJEMBES']):
            basket = 'PICNIC_BASKET1'
            comps = BASKET1_COMPONENTS
            basket_depth = state.order_depths[basket]
            try:
                comp_asks = {c: min(state.order_depths[c].sell_orders.keys()) for c in comps if state.order_depths[c].sell_orders}
                comp_bids = {c: max(state.order_depths[c].buy_orders.keys()) for c in comps if state.order_depths[c].buy_orders}
                if len(comp_asks) != len(comps) or len(comp_bids) != len(comps): raise ValueError("Missing component prices")

                basket_bid = max(basket_depth.buy_orders.keys()) if basket_depth.buy_orders else None
                basket_ask = min(basket_depth.sell_orders.keys()) if basket_depth.sell_orders else None

                basket_fair_ask = sum(comp_asks[c] * q for c, q in comps.items())
                basket_fair_bid = sum(comp_bids[c] * q for c, q in comps.items())
                threshold = 100

                pos = state.position.get(basket, 0)

                if basket_bid and basket_bid > basket_fair_ask + threshold and pos < POSITION_LIMITS[basket]:
                    vol = min(POSITION_LIMITS[basket] - pos, 5)
                    if vol > 0:
                        orders[basket].append(Order(basket, basket_bid, -vol))
                        for c, q in comps.items(): orders[c].append(Order(c, comp_asks[c], q * vol))

                if basket_ask and basket_ask < basket_fair_bid - threshold and pos > -POSITION_LIMITS[basket]:
                    vol = min(pos + POSITION_LIMITS[basket], 5)
                    if vol > 0:
                        orders[basket].append(Order(basket, basket_ask, vol))
                        for c, q in comps.items(): orders[c].append(Order(c, comp_bids[c], -q * vol))
            except Exception as e: logger.print(f"Error in Basket 1 arbitrage: {e}")
        # Basket 2
        if all(p in state.order_depths for p in ['PICNIC_BASKET2', 'JAMS', 'CROISSANTS']):
            basket = 'PICNIC_BASKET2'
            comps = BASKET2_COMPONENTS
            basket_depth = state.order_depths[basket]
            try:
                comp_asks = {c: min(state.order_depths[c].sell_orders.keys()) for c in comps if state.order_depths[c].sell_orders}
                comp_bids = {c: max(state.order_depths[c].buy_orders.keys()) for c in comps if state.order_depths[c].buy_orders}
                if len(comp_asks) != len(comps) or len(comp_bids) != len(comps): raise ValueError("Missing component prices")

                basket_bid = max(basket_depth.buy_orders.keys()) if basket_depth.buy_orders else None
                basket_ask = min(basket_depth.sell_orders.keys()) if basket_depth.sell_orders else None

                basket_fair_ask = sum(comp_asks[c] * q for c, q in comps.items())
                basket_fair_bid = sum(comp_bids[c] * q for c, q in comps.items())
                threshold = 80

                pos = state.position.get(basket, 0)

                if basket_bid and basket_bid > basket_fair_ask + threshold and pos < POSITION_LIMITS[basket]:
                    vol = min(POSITION_LIMITS[basket] - pos, 5)
                    if vol > 0:
                        orders[basket].append(Order(basket, basket_bid, -vol))
                        for c, q in comps.items(): orders[c].append(Order(c, comp_asks[c], q * vol))

                if basket_ask and basket_ask < basket_fair_bid - threshold and pos > -POSITION_LIMITS[basket]:
                    vol = min(pos + POSITION_LIMITS[basket], 5)
                    if vol > 0:
                        orders[basket].append(Order(basket, basket_ask, vol))
                        for c, q in comps.items(): orders[c].append(Order(c, comp_bids[c], -q * vol))
            except Exception as e: logger.print(f"Error in Basket 2 arbitrage: {e}")
        return orders

    def run(self, state: TradingState):
        """Main entry point for the trader's logic."""
        # print("RUN METHOD ENTERED") # Keep for debugging if needed
        logger.print(f"Timestamp: {state.timestamp}")
        result = {}
        conversions = 0
        trader_data = "SELF_CONTAINED_V1" # Indicate strategy version
        positions = state.position

        # Define product groups
        core_products = ['RAINFOREST_RESIN', 'CROISSANTS', 'JAMS', 'DJEMBES', 'VOLCANIC_ROCK'] # VR moved here
        simple_mr_products = ['KELP', 'SQUID_INK']
        voucher_products = [p for p in POSITION_LIMITS if p.startswith('VOLCANIC_ROCK_VOUCHER_')]
        other_special_products = ['MAGNIFICENT_MACARONS'] # VR removed

        try:
            # --- Update Rolling Stats ---
            for product, trades in state.market_trades.items():
                # Only update rolling stats for products we use them for
                if product in core_products + simple_mr_products: # VR included via core_products
                    for trade in trades:
                        self.rolling[product].update(trade.price)

            # --- Apply Strategies ---
            for product, order_depth in state.order_depths.items():
                pos = positions.get(product, 0)
                orders = []

                if product in core_products:
                    orders = self.market_make_and_mean_revert(product, order_depth, pos, state)
                elif product in simple_mr_products:
                    mean = self.rolling[product].mean()
                    std = self.rolling[product].std()
                    if mean is not None:
                        threshold = 0
                        if product == 'KELP': threshold = 3
                        elif product == 'SQUID_INK': threshold = (std * 2) if std is not None and std > 0 else 7 # Reverted threshold

                        # Buy below mean - threshold
                        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                        if best_ask and best_ask < mean - threshold and pos < POSITION_LIMITS[product]:
                            buy_vol = min(-order_depth.sell_orders[best_ask], POSITION_LIMITS[product] - pos, self.max_single_trade)
                            if buy_vol > 0: orders.append(Order(product, best_ask, buy_vol))
                        # Sell above mean + threshold
                        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                        if best_bid and best_bid > mean + threshold and pos > -POSITION_LIMITS[product]:
                            sell_vol = min(order_depth.buy_orders[best_bid], pos + POSITION_LIMITS[product], self.max_single_trade)
                            if sell_vol > 0: orders.append(Order(product, best_bid, -sell_vol))

                elif product in voucher_products:
                    # --- VOLCANIC_ROCK_VOUCHERS ---
                    try:
                        vr_rolling_mean = self.rolling['VOLCANIC_ROCK'].mean()
                        vr_rolling_std = self.rolling['VOLCANIC_ROCK'].std()
                        # Use static fair value as fallback if rolling mean is not available yet
                        underlying = vr_rolling_mean if vr_rolling_mean is not None else FAIR_VALUES['VOLCANIC_ROCK']
                        # Use static volatility as fallback if rolling std is not available yet
                        vol_estimate = vr_rolling_std if vr_rolling_std is not None else PARAMS['VOLCANIC_ROCK']['volatility']

                        strike = int(product.split('_')[-1])
                        day_index = state.timestamp // 1000000
                        round_day_index = day_index + 3 # Round 4 starts on day 3 (0-indexed)
                        days_remaining = 7 - round_day_index
                        TTE = max(0.01, days_remaining) # Use days

                        intrinsic = max(0, underlying - strike)
                        time_value = vol_estimate * 0.1 * math.sqrt(TTE) # Reverted multiplier
                        fair_theoretical = intrinsic + time_value

                        # Use theoretical fair value as primary reference
                        fair = fair_theoretical
                        # logger.print(f"{product}: Underlying={underlying:.2f}, Strike={strike}, TTE={TTE:.2f}, Fair={fair:.2f}")

                        # Simple fixed spread MM around theoretical fair value
                        spread = 4 # Fixed spread
                        buy_price = math.floor(fair - spread / 2)
                        sell_price = math.ceil(fair + spread / 2)

                        # MM logic
                        if pos < POSITION_LIMITS[product]:
                            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                            # Buy only if best ask is below our calculated buy price
                            if best_ask and best_ask < buy_price:
                                buy_vol = min(-order_depth.sell_orders[best_ask], POSITION_LIMITS[product] - pos, self.max_single_trade)
                                if buy_vol > 0: orders.append(Order(product, best_ask, buy_vol))
                        if pos > -POSITION_LIMITS[product]:
                            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                            # Sell only if best bid is above our calculated sell price
                            if best_bid and best_bid > sell_price:
                                sell_vol = min(order_depth.buy_orders[best_bid], pos + POSITION_LIMITS[product], self.max_single_trade)
                                if sell_vol > 0: orders.append(Order(product, best_bid, -sell_vol))
                    except Exception as voucher_e:
                        logger.print(f"Error processing voucher {product}: {voucher_e}")

                elif product == 'MAGNIFICENT_MACARONS':
                    # --- MAGNIFICENT_MACARONS ---
                    obs = state.observations.conversionObservations.get(product)
                    sunlight = obs.sunlightIndex if obs and hasattr(obs, 'sunlightIndex') else 54
                    fair = 650 if sunlight < self.sunlight_csi else 610
                    spread = 5
                    buy_price = fair - spread
                    sell_price = fair + spread

                    if pos < POSITION_LIMITS[product]:
                        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                        if best_ask and best_ask < buy_price:
                            buy_vol = min(-order_depth.sell_orders[best_ask], POSITION_LIMITS[product] - pos, 10, self.max_single_trade)
                            if buy_vol > 0: orders.append(Order(product, best_ask, buy_vol))
                    if pos > -POSITION_LIMITS[product]:
                        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                        if best_bid and best_bid > sell_price:
                            sell_vol = min(order_depth.buy_orders[best_bid], pos + POSITION_LIMITS[product], 10, self.max_single_trade)
                            if sell_vol > 0: orders.append(Order(product, best_bid, -sell_vol))

                # Add generated orders to the result dictionary
                if orders:
                    result[product] = orders

            # --- Basket Arbitrage ---
            basket_orders = self.basket_arbitrage(state)
            for k, v in basket_orders.items():
                if k in result: result[k].extend(v)
                else: result[k] = v

        except Exception as e:
            logger.print(f"ERROR in run method: {e}")

        # --- Final Flush ---
        try:
            logger.flush(state, result, conversions, trader_data)
        except Exception as flush_e:
             print(f"CRITICAL ERROR: logger.flush failed: {flush_e}")
             print(f"Accumulated logs before flush error: {logger.logs}")

        return result, conversions, trader_data
