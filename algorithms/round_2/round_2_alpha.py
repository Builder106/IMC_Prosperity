import json
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order
from collections import deque
import numpy as np
import math

# Define product names as constants
RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"
CROISSANTS = "CROISSANTS"
JAMS = "JAMS" 
DJEMBES = "DJEMBES"
PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"

# Position limits for each product
POSITION_LIMITS = {
    RAINFOREST_RESIN: 50,
    KELP: 50,
    SQUID_INK: 50,
    CROISSANTS: 250,
    JAMS: 350,
    DJEMBES: 60,
    PICNIC_BASKET1: 60,
    PICNIC_BASKET2: 100
}

# Product fair values (derived from data analysis)
FAIR_VALUES = {
    RAINFOREST_RESIN: 10000,  # Still seems accurate
    KELP: 2035,               # Consider adjusting based on day-to-day performance
    SQUID_INK: 1890,          # Slight adjustment from 1887
    CROISSANTS: 4272,         # Appears accurate
    JAMS: 6530,               # Adjust slightly down from 6534
    DJEMBES: 13390,           # Adjust slightly from 13393
    PICNIC_BASKET1: 58630,    # Adjust from 58644
    PICNIC_BASKET2: 30260     # Adjust from 30251
}

# Volatility and strategy parameters (derived from data analysis)
PARAMS = {
    RAINFOREST_RESIN: {
        "volatility": 3.24,
        "spread_capture": 3,     # Higher than volatility to account for wide spreads
        "mean_reversion_strength": 0.3,  # Increase from 0.2
        "sma_window": 15,               # Reduce from 20 for faster response
        "position_scale": 1.2    # Scale for position sizing
    },
    KELP: {
        "volatility": 1.17,
        "spread_capture": 1.5,
        "mean_reversion_strength": 0.5,  # Stronger mean reversion due to lower volatility
        "sma_window": 15,
        "position_scale": 0.8    # Lower position scale due to underperforming
    },
    SQUID_INK: {
        "volatility": 15.46,
        "spread_capture": 2.5,
        "mean_reversion_strength": 1.2,  # Increase from 1.0
        "sma_window": 8,                # Faster response
        "position_scale": 1.0
    },
    CROISSANTS: {
        "volatility": 2.92,
        "spread_capture": 2.0,
        "mean_reversion_strength": 0.4,
        "sma_window": 15,
        "position_scale": 1.0
    },
    JAMS: {
        "volatility": 7.58,
        "spread_capture": 2.5,
        "mean_reversion_strength": 0.5,
        "sma_window": 15,
        "position_scale": 1.2
    },
    DJEMBES: {
        "volatility": 16.57,
        "spread_capture": 3.0,
        "mean_reversion_strength": 0.6,
        "sma_window": 20, 
        "position_scale": 0.7    # Lower position scale due to high volatility
    },
    PICNIC_BASKET1: {
        "volatility": 33.62,
        "spread_capture": 7.5,   # Based on avg_spread from data
        "mean_reversion_strength": 0.3,
        "sma_window": 15,
        "position_scale": 0.7    # Lower position scale due to very high volatility
    },
    PICNIC_BASKET2: {
        "volatility": 18.92,
        "spread_capture": 5.0,   # Based on avg_spread from data
        "mean_reversion_strength": 0.5,  # Increase from 0.4
        "sma_window": 15,
        "position_scale": 0.6    # Lower position scale due to high volatility
    }
}

# Conversion parameters for baskets
BASKET1_COMPONENTS = {
    DJEMBES: 1,
    JAMS: 3,
    CROISSANTS: 6
}

BASKET2_COMPONENTS = {
    JAMS: 2,
    CROISSANTS: 4
}

class Trader:
    def __init__(self):
        # Store price history for all products
        self.price_history = {
            product: deque(maxlen=max([p["sma_window"] for p in PARAMS.values()]))
            for product in POSITION_LIMITS.keys()
        }
        
        # Store mid prices from last iteration
        self.last_mid_prices = {}
        
        # Store position history for basket conversion analysis
        self.previous_position = {}
        
        print("Trader initialized with strategies for all products.")
        
    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate the mid price from the order book."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0
        
    def calculate_vwap(self, order_depth: OrderDepth, side: str) -> float:
        """Calculate volume-weighted average price for a side (buy or sell)."""
        if side == "buy":
            if not order_depth.buy_orders:
                return None
            total_volume = 0
            weighted_sum = 0
            for price, volume in order_depth.buy_orders.items():
                weighted_sum += price * volume
                total_volume += volume
        else:  # sell
            if not order_depth.sell_orders:
                return None
            total_volume = 0
            weighted_sum = 0
            for price, volume in order_depth.sell_orders.items():
                weighted_sum += price * abs(volume)
                total_volume += abs(volume)
                
        return weighted_sum / total_volume if total_volume > 0 else None

    def calculate_order_imbalance(self, order_depth: OrderDepth) -> float:
        """Calculate order imbalance from the order book."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
            
        total_buy_volume = sum(order_depth.buy_orders.values())
        total_sell_volume = sum(abs(volume) for volume in order_depth.sell_orders.values())
        
        if total_buy_volume + total_sell_volume == 0:
            return 0
            
        return (total_buy_volume - total_sell_volume) / (total_buy_volume + total_sell_volume)

    def determine_general_trend(self, prices_deque):
        """Determine the general trend from recent price history."""
        if len(prices_deque) < 5:
            return 0
            
        # Take the last 5 prices
        recent_prices = list(prices_deque)[-5:]
        
        # Simple linear regression to get trend
        x = list(range(len(recent_prices)))
        y = recent_prices
        
        if len(x) < 2:
            return 0
            
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Normalize the slope
        return slope / (y_mean if y_mean != 0 else 1)

    def get_trade_signal(self, product: str, mid_price: float) -> float:
        """
        Generate a trade signal (-1 to 1) for mean reversion strategy.
        Negative means sell, positive means buy.
        """
        if len(self.price_history[product]) < PARAMS[product]["sma_window"]:
            return 0
        
        sma = sum(self.price_history[product]) / len(self.price_history[product])
        deviation = mid_price - sma
        normalized_deviation = deviation / PARAMS[product]["volatility"]
        
        # Tanh gives a nice curve between -1 and 1
        signal = -math.tanh(normalized_deviation * PARAMS[product]["mean_reversion_strength"])
        
        # Add trend detection to moderate mean reversion
        trend = self.determine_general_trend(self.price_history[product])
        trend_adjustment = 0.3 * trend  # Adjust signal based on trend direction
        
        final_signal = signal + trend_adjustment
        
        # Ensure signal remains between -1 and 1
        return max(-1, min(1, final_signal))
        
    def calculate_position_size(self, product: str, signal_strength: float, current_position: int) -> int:
        """Calculate the appropriate position size based on signal strength and current position."""
        max_position = POSITION_LIMITS[product]
        position_scale = PARAMS[product]["position_scale"]
        
        # How much of our limit we want to use
        target_position = int(max_position * signal_strength * position_scale)
        
        # Adjust based on current position to avoid crossing limits
        position_adjustment = target_position - current_position
        
        # Scale down adjustment if it would exceed limits
        if current_position + position_adjustment > max_position:
            position_adjustment = max_position - current_position
        elif current_position + position_adjustment < -max_position:
            position_adjustment = -max_position - current_position
            
        return position_adjustment
        
    def market_make_orders(self, product: str, order_depth: OrderDepth, current_position: int):
        """Generate market making orders for a product."""
        orders = []
        params = PARAMS[product]
        fair_value = FAIR_VALUES[product]
        
        # Skip if insufficient data in the order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread = best_ask - best_bid
        
        # More dynamic bid/ask adjustment based on product performance
        if product in [RAINFOREST_RESIN, JAMS, PICNIC_BASKET1]:
            # More aggressive for highly profitable products
            our_bid = best_bid + 1 if best_bid < fair_value - params["spread_capture"]/2 else best_bid
            our_ask = best_ask - 1 if best_ask > fair_value + params["spread_capture"]/2 else best_ask
        else:
            # Standard approach for other products
            our_bid = best_bid + 1 if best_bid < fair_value - params["spread_capture"] else best_bid
            our_ask = best_ask - 1 if best_ask > fair_value + params["spread_capture"] else best_ask
        
        # Adjust for current position - be more aggressive on the side that reduces position
        if abs(current_position) > POSITION_LIMITS[product] * 0.7:
            if current_position > 0:  # We need to sell
                our_ask = min(our_ask, best_bid + 1)  # Make our ask more competitive
            else:  # We need to buy
                our_bid = max(our_bid, best_ask - 1)  # Make our bid more competitive
        
        # Calculate volumes based on order imbalance
        imbalance = self.calculate_order_imbalance(order_depth)
        base_volume = int(POSITION_LIMITS[product] * 0.1)  # Base 10% of position limit
        
        # Adjust volumes based on imbalance
        bid_volume = int(base_volume * (1 + imbalance * 0.5))
        ask_volume = int(base_volume * (1 - imbalance * 0.5))
        
        # Ensure we don't exceed position limits
        bid_volume = min(bid_volume, POSITION_LIMITS[product] - current_position)
        ask_volume = min(ask_volume, POSITION_LIMITS[product] + current_position)
        
        # Only place orders if volumes are positive
        if bid_volume > 0:
            orders.append(Order(product, our_bid, bid_volume))
        if ask_volume > 0:
            orders.append(Order(product, our_ask, -ask_volume))
            
        return orders
        
    def mean_reversion_orders(self, product: str, mid_price: float, current_position: int):
        """Generate mean reversion orders for a product."""
        orders = []
        
        # Get trade signal (-1 to 1)
        signal = self.get_trade_signal(product, mid_price)
        
        if abs(signal) < 0.1:  # Signal too weak
            return orders
            
        # Calculate position size
        position_adjustment = self.calculate_position_size(product, signal, current_position)
        
        if position_adjustment == 0:
            return orders
            
        # Determine price based on direction
        params = PARAMS[product]
        fair_value = mid_price  # Use current mid price as the fair value reference
        
        if position_adjustment > 0:  # Buy order
            # Place limit order slightly above mid price to increase fill probability
            price = int(fair_value - params["spread_capture"] * 0.5)
            orders.append(Order(product, price, position_adjustment))
        else:  # Sell order
            price = int(fair_value + params["spread_capture"] * 0.5)
            orders.append(Order(product, price, position_adjustment))
            
        return orders
        
    def evaluate_basket_conversion(self, state: TradingState) -> int:
        """
        Evaluate whether to convert between baskets and individual components.
        Returns number of conversions to perform (positive for creating baskets,
        negative for breaking baskets).
        """
        conversions = 0
        
        # Only proceed if we have all required products
        required_products = [JAMS, CROISSANTS, DJEMBES, PICNIC_BASKET1, PICNIC_BASKET2]
        all_products_available = all(product in state.order_depths for product in required_products)
        
        if not all_products_available:
            return 0
            
        # Get current mid prices
        mid_prices = {}
        for product in required_products:
            mid_price = self.calculate_mid_price(state.order_depths[product])
            if mid_price is None:
                return 0  # Missing price data
            mid_prices[product] = mid_price
            
        current_positions = {
            product: state.position.get(product, 0) for product in required_products
        }
        
        # Calculate basket arbitrage opportunities
        # Basket 1: 3 JAMS + 6 CROISSANTS + 1 DJEMBE
        basket1_cost = 3 * mid_prices[JAMS] + 6 * mid_prices[CROISSANTS] + 1 * mid_prices[DJEMBES]
        basket1_arb = mid_prices[PICNIC_BASKET1] - basket1_cost
        
        # Basket 2: 2 JAMS + 4 CROISSANTS  
        basket2_cost = 2 * mid_prices[JAMS] + 4 * mid_prices[CROISSANTS]
        basket2_arb = mid_prices[PICNIC_BASKET2] - basket2_cost
        
        # Conversion thresholds - based on typical spreads from data
        basket1_threshold = mid_prices[PICNIC_BASKET1] * 0.002  # Reduce from 0.0025 to capture more opportunities
        basket2_threshold = mid_prices[PICNIC_BASKET2] * 0.003  # Increase from 0.0025 for more conservative approach
        
        # Decision logic for Basket 1
        if basket1_arb > basket1_threshold:  # Create baskets
            # Check position limits
            max_conversions_jams = current_positions[JAMS] // 3
            max_conversions_croissants = current_positions[CROISSANTS] // 6
            max_conversions_djembes = current_positions[DJEMBES] // 1
            max_basket1_create = min(max_conversions_jams, max_conversions_croissants, max_conversions_djembes)
            
            # Check position limit for PICNIC_BASKET1
            basket1_headroom = POSITION_LIMITS[PICNIC_BASKET1] - current_positions[PICNIC_BASKET1]
            max_basket1_create = min(max_basket1_create, basket1_headroom)
            
            if max_basket1_create > 0:
                conversions += max_basket1_create
                
        elif basket1_arb < -basket1_threshold:  # Break baskets
            # Check how many baskets we have
            max_basket1_break = current_positions[PICNIC_BASKET1]
            
            # Check position limits for components
            jams_headroom = (POSITION_LIMITS[JAMS] - current_positions[JAMS]) // 3
            croissants_headroom = (POSITION_LIMITS[CROISSANTS] - current_positions[CROISSANTS]) // 6
            djembes_headroom = (POSITION_LIMITS[DJEMBES] - current_positions[DJEMBES]) // 1
            max_basket1_break = min(max_basket1_break, jams_headroom, croissants_headroom, djembes_headroom)
            
            if max_basket1_break > 0:
                conversions -= max_basket1_break
                
        # Decision logic for Basket 2
        if basket2_arb > basket2_threshold:  # Create baskets
            # Check position limits
            max_conversions_jams = current_positions[JAMS] // 2
            max_conversions_croissants = current_positions[CROISSANTS] // 4
            max_basket2_create = min(max_conversions_jams, max_conversions_croissants)
            
            # Check position limit for PICNIC_BASKET2
            basket2_headroom = POSITION_LIMITS[PICNIC_BASKET2] - current_positions[PICNIC_BASKET2]
            max_basket2_create = min(max_basket2_create, basket2_headroom)
            
            if max_basket2_create > 0:
                conversions += max_basket2_create
                
        elif basket2_arb < -basket2_threshold:  # Break baskets
            # Check how many baskets we have
            max_basket2_break = current_positions[PICNIC_BASKET2]
            
            # Check position limits for components
            jams_headroom = (POSITION_LIMITS[JAMS] - current_positions[JAMS]) // 2
            croissants_headroom = (POSITION_LIMITS[CROISSANTS] - current_positions[CROISSANTS]) // 4
            max_basket2_break = min(max_basket2_break, jams_headroom, croissants_headroom)
            
            if max_basket2_break > 0:
                conversions -= max_basket2_break
                
        return conversions
        
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Main method to generate trading decisions.
        """
        print(f"\n--- Trader Run - Timestamp {state.timestamp} ---")
        result = {}  # Orders to place for each product
        
        # Store mid prices for this iteration
        current_mid_prices = {}
        
        # Process each product
        for product in state.order_depths.keys():
            order_depth = state.order_depths[product]
            orders: list[Order] = []
            current_position = state.position.get(product, 0)
            
            print(f"Processing {product} - Current position: {current_position}")
            
            # Calculate mid price
            mid_price = self.calculate_mid_price(order_depth)
            if mid_price:
                current_mid_prices[product] = mid_price
                self.price_history[product].append(mid_price)
                
                # Choose strategy based on product characteristics
                # For all products, we'll use a blend of market making and mean reversion
                
                # Market making strategy
                mm_orders = self.market_make_orders(product, order_depth, current_position)
                orders.extend(mm_orders)
                
                # Mean reversion strategy if we have enough price history
                if len(self.price_history[product]) >= PARAMS[product]["sma_window"]:
                    mr_orders = self.mean_reversion_orders(product, mid_price, current_position)
                    orders.extend(mr_orders)
                else:
                    print(f"  {product}: Collecting price history ({len(self.price_history[product])}/{PARAMS[product]['sma_window']})")
            else:
                print(f"  {product}: Could not calculate mid-price (likely insufficient order book data).")
            
        def update_volatility_parameters(self, product: str, mid_price: float):
            """Update volatility parameters based on recent price movements"""
            if len(self.price_history[product]) < 10:
                return
                
            # Calculate recent volatility
            recent_prices = list(self.price_history[product])[-10:]
            recent_volatility = np.std(recent_prices)
            expected_volatility = PARAMS[product]["volatility"]
            
            # Adjust spread_capture based on observed volatility
            volatility_ratio = recent_volatility / expected_volatility
            if volatility_ratio > 1.5:
                # Higher volatility - widen spread
                PARAMS[product]["spread_capture"] *= 1.2
            elif volatility_ratio < 0.7:
                # Lower volatility - tighten spread
                PARAMS[product]["spread_capture"] *= 0.9
                
            # Keep spread_capture within reasonable bounds
            PARAMS[product]["spread_capture"] = max(1.0, min(10.0, PARAMS[product]["spread_capture"]))

            # Add orders to result if we have any
            if orders:
                result[product] = orders
                print(f"  {product}: Placed {len(orders)} orders")
        
        # Store mid prices for next iteration
        self.last_mid_prices = current_mid_prices
        
        # Evaluate basket conversions
        conversions = self.evaluate_basket_conversion(state)
        
        # Store position for next iteration
        self.previous_position = {
            product: state.position.get(product, 0) 
            for product in state.position.keys()
        }
        
        # Print order summary
        print(f"--- Orders Generated: {[(k, [(o.symbol, o.price, o.quantity) for o in v]) for k, v in result.items()]} ---")
        print(f"--- Conversions: {conversions} ---")
        
        # Can include trader data if needed for state persistence
        traderData = ""
        
        return result, conversions, traderData
