import json
from typing import Dict, List, Any, Tuple
from datamodel import OrderDepth, TradingState, Order, Symbol, Listing, Observation, Trade, ProsperityEncoder
from collections import deque
import numpy as np
import math


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


# Global logger instance
logger = Logger()

# Define product names as constants
RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"
CROISSANTS = "CROISSANTS"
JAMS = "JAMS" 
DJEMBES = "DJEMBES"
PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"

# Round 3 products
VOLCANIC_ROCK = "VOLCANIC_ROCK"
VRV_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
VRV_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
VRV_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
VRV_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
VRV_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

# Position limits for each product
POSITION_LIMITS = {
    RAINFOREST_RESIN: 50,
    KELP: 50,
    SQUID_INK: 50,
    CROISSANTS: 250,
    JAMS: 350,
    DJEMBES: 60,
    PICNIC_BASKET1: 60,
    PICNIC_BASKET2: 100,
    # Round 3 position limits
    VOLCANIC_ROCK: 400,
    VRV_9500: 200,
    VRV_9750: 200,
    VRV_10000: 200,
    VRV_10250: 200,
    VRV_10500: 200
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
    PICNIC_BASKET2: 30260,    # Adjust from 30251
    VOLCANIC_ROCK: 10000      # Based on historical pricing of Rainforest Resin
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

# Options (Volcanic Rock Vouchers) parameters
OPTION_STRIKES = {
    VRV_9500: 9500,
    VRV_9750: 9750, 
    VRV_10000: 10000,
    VRV_10250: 10250,
    VRV_10500: 10500
}

# Option parameters
DAYS_TO_EXPIRY_START = 7  # Total days to expiration from round 1
OPTION_VOLATILITY = 0.20  # Initial estimate of underlying volatility
RISK_FREE_RATE = 0.01     # Risk-free rate for option pricing

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
        
        logger.print("Trader initialized with strategies for all products.")
        
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
        
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """
        Main method to generate trading decisions.
        """
        logger.print(f"\n--- Trader Run - Timestamp {state.timestamp} ---")
        result = {}  # Orders to place for each product
        
        # Store mid prices for this iteration
        current_mid_prices = {}
        
        # Calculate days to expiry for options
        days_to_expiry = self.calculate_days_to_expiry(state.timestamp)
        logger.print(f"Days to option expiry: {days_to_expiry}")
        
        # Get VOLCANIC_ROCK price first (needed for option pricing)
        volcanic_rock_price = None
        if VOLCANIC_ROCK in state.order_depths:
            volcanic_rock_depth = state.order_depths[VOLCANIC_ROCK]
            volcanic_rock_price = self.calculate_mid_price(volcanic_rock_depth)
            if volcanic_rock_price:
                current_mid_prices[VOLCANIC_ROCK] = volcanic_rock_price
                self.price_history[VOLCANIC_ROCK].append(volcanic_rock_price)
                logger.print(f"VOLCANIC_ROCK price: {volcanic_rock_price}")
        
        # Define lists to track different product types
        regular_products = [p for p in state.order_depths.keys() if not p.startswith("VOLCANIC_ROCK_VOUCHER")]
        option_products = [p for p in state.order_depths.keys() if p.startswith("VOLCANIC_ROCK_VOUCHER")]
        
        # Process regular products
        for product in regular_products:
            order_depth = state.order_depths[product]
            orders: list[Order] = []
            current_position = state.position.get(product, 0)
            
            logger.print(f"Processing {product} - Current position: {current_position}")
            
            # Calculate mid price
            mid_price = self.calculate_mid_price(order_depth)
            if mid_price:
                current_mid_prices[product] = mid_price
                self.price_history[product].append(mid_price)
                
                # Skip options strategies for regular products
                if product in PARAMS:
                    # Market making strategy
                    mm_orders = self.market_make_orders(product, order_depth, current_position)
                    orders.extend(mm_orders)
                    
                    # Mean reversion strategy if we have enough price history
                    if len(self.price_history[product]) >= PARAMS[product]["sma_window"]:
                        mr_orders = self.mean_reversion_orders(product, mid_price, current_position)
                        orders.extend(mr_orders)
                    else:
                        logger.print(f"  {product}: Collecting price history ({len(self.price_history[product])}/{PARAMS[product]['sma_window']})")
                else:
                    logger.print(f"  {product}: No strategy parameters defined, skipping.")
            else:
                logger.print(f"  {product}: Could not calculate mid-price (likely insufficient order book data).")
            
            # Add orders to result if we have any
            if orders:
                result[product] = orders
                logger.print(f"  {product}: Placed {len(orders)} orders")
        
        # Process option products if we have VOLCANIC_ROCK price
        if volcanic_rock_price and option_products:
            logger.print(f"Processing option products: {option_products}")
            
            # Market making for each option
            for option_product in option_products:
                if option_product in state.order_depths:
                    current_position = state.position.get(option_product, 0)
                    logger.print(f"Processing {option_product} - Current position: {current_position}")
                    
                    # Market make on options
                    option_orders = self.option_market_make_orders(
                        option_product=option_product,
                        order_depth=state.order_depths[option_product],
                        days_to_expiry=days_to_expiry,
                        underlying_price=volcanic_rock_price,
                        current_position=current_position
                    )
                    
                    if option_orders:
                        result[option_product] = option_orders
                        logger.print(f"  {option_product}: Placed {len(option_orders)} market making orders")
            
            # Try to find arbitrage opportunities between options
            arbitrage_orders = self.option_arbitrage_orders(
                option_products=option_products,
                state=state,
                underlying_price=volcanic_rock_price,
                days_to_expiry=days_to_expiry
            )
            
            # Add arbitrage orders to result
            for product, orders in arbitrage_orders.items():
                if orders:
                    if product in result:
                        result[product].extend(orders)
                    else:
                        result[product] = orders
                    logger.print(f"  {product}: Added {len(orders)} arbitrage orders")
        
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
        logger.print(f"--- Orders Generated: {[(k, [(o.symbol, o.price, o.quantity) for o in v]) for k, v in result.items()]} ---")
        logger.print(f"--- Conversions: {conversions} ---")
        
        # Can include trader data if needed for state persistence
        trader_data = ""
        
        # Flush logs before returning
        logger.flush(state, result, conversions, trader_data)
        
        return result, conversions, trader_data
    
    def calculate_days_to_expiry(self, timestamp: int) -> int:
        """
        Calculate days remaining until option expiry based on the current timestamp.
        Round 1 is day 1, Round 5 is day 5, options expire after day 7.
        """
        # Timestamps are in microseconds from start of round
        day = (timestamp // 1000000 // 60 // 60 // 24) + 1  # Add 1 because day count starts at 1
        days_remaining = DAYS_TO_EXPIRY_START - day
        
        # Ensure we don't return negative days
        return max(0, days_remaining)
        
    def calculate_option_price(self, underlying_price: float, strike: float, days_to_expiry: int, 
                              volatility: float = OPTION_VOLATILITY, is_call: bool = True) -> float:
        """
        Calculate theoretical option price using Black-Scholes model.
        
        Parameters:
        - underlying_price: Current price of VOLCANIC_ROCK
        - strike: Strike price of the option
        - days_to_expiry: Days remaining until expiration
        - volatility: Implied volatility (annualized)
        - is_call: True for call options, False for put options
        
        Returns:
        - Theoretical option price
        """
        # Convert days to years for B-S formula
        T = days_to_expiry / 365.0
        
        # If option is about to expire, use intrinsic value
        if days_to_expiry <= 0:
            if is_call:
                return max(0, underlying_price - strike)
            else:
                return max(0, strike - underlying_price)
                
        # Handle case where time to expiry is very small but positive
        if T < 0.0001:
            T = 0.0001
            
        # Black-Scholes components
        d1 = (math.log(underlying_price / strike) + (RISK_FREE_RATE + 0.5 * volatility ** 2) * T) / (volatility * math.sqrt(T))
        d2 = d1 - volatility * math.sqrt(T)
        
        # Calculate cumulative normal distribution
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
            
        if is_call:
            price = underlying_price * norm_cdf(d1) - strike * math.exp(-RISK_FREE_RATE * T) * norm_cdf(d2)
        else:
            price = strike * math.exp(-RISK_FREE_RATE * T) * norm_cdf(-d2) - underlying_price * norm_cdf(-d1)
            
        return price
        
    def calculate_option_delta(self, underlying_price: float, strike: float, days_to_expiry: int,
                               volatility: float = OPTION_VOLATILITY, is_call: bool = True) -> float:
        """
        Calculate option delta (sensitivity to underlying price changes).
        
        Returns:
        - Delta value between 0-1 for calls, -1-0 for puts
        """
        # Convert days to years
        T = days_to_expiry / 365.0
        
        # Handle expiry or near-expiry
        if days_to_expiry <= 0:
            if is_call:
                return 1.0 if underlying_price > strike else 0.0
            else:
                return -1.0 if underlying_price < strike else 0.0
        
        # Handle case where time to expiry is very small but positive
        if T < 0.0001:
            T = 0.0001
            
        # Calculate d1 from Black-Scholes
        d1 = (math.log(underlying_price / strike) + (RISK_FREE_RATE + 0.5 * volatility ** 2) * T) / (volatility * math.sqrt(T))
        
        # Calculate cumulative normal distribution
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
            
        if is_call:
            return norm_cdf(d1)
        else:
            return norm_cdf(d1) - 1
            
    def calculate_option_gamma(self, underlying_price: float, strike: float, days_to_expiry: int,
                               volatility: float = OPTION_VOLATILITY) -> float:
        """
        Calculate option gamma (second derivative of price with respect to underlying).
        
        Returns:
        - Gamma value (same for both calls and puts)
        """
        # Convert days to years
        T = days_to_expiry / 365.0
        
        if T <= 0.0001:
            return 0.0  # At expiry, gamma is effectively 0
            
        # Calculate d1 term
        d1 = (math.log(underlying_price / strike) + (RISK_FREE_RATE + 0.5 * volatility ** 2) * T) / (volatility * math.sqrt(T))
        
        # Standard normal PDF
        def norm_pdf(x):
            return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
            
        # Gamma formula
        gamma = norm_pdf(d1) / (underlying_price * volatility * math.sqrt(T))
        return gamma
    
    def calculate_option_spread(self, option_product: str, order_depth: OrderDepth) -> Tuple[float, float, float]:
        """
        Calculate the spread and fair value for an option product.
        
        Returns:
        - Tuple of (fair_value, bid_adjust, ask_adjust)
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None, None, None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2.0
        
        # Calculate wider spreads for less liquid options
        bid_adjust = 2
        ask_adjust = 2
        
        # Adjust spread based on strike - farther OTM options tend to be less liquid
        if option_product in [VRV_9500, VRV_10500]:  # Further ITM/OTM
            bid_adjust = 3
            ask_adjust = 3
            
        return mid_price, bid_adjust, ask_adjust
        
    def option_market_make_orders(self, option_product: str, order_depth: OrderDepth, days_to_expiry: int, 
                                  underlying_price: float, current_position: int) -> List[Order]:
        """
        Generate market making orders for option products.
        """
        orders = []
        strike = OPTION_STRIKES[option_product]
        
        # Skip if insufficient data in the order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        # Calculate theoretical option price
        theo_price = self.calculate_option_price(
            underlying_price=underlying_price,
            strike=strike,
            days_to_expiry=days_to_expiry
        )
        
        # Calculate option greeks
        delta = self.calculate_option_delta(
            underlying_price=underlying_price,
            strike=strike,
            days_to_expiry=days_to_expiry
        )
        
        # Get current market spread
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # Base our pricing on theoretical price with wider spreads for less liquid options
        mid_price, bid_adjust, ask_adjust = self.calculate_option_spread(option_product, order_depth)
        
        # Use theoretical price as a guide, but stay within market spread to avoid adverse selection
        our_bid = min(theo_price - bid_adjust, best_ask - 1) 
        our_ask = max(theo_price + ask_adjust, best_bid + 1)
        
        # Adjust for current position - be more aggressive on the side that reduces position
        position_scale = 0.7  # Position at which we become more aggressive
        if abs(current_position) > POSITION_LIMITS[option_product] * position_scale:
            if current_position > 0:  # We need to sell
                our_ask = min(our_ask, best_bid + 1)  # More competitive ask
            else:  # We need to buy
                our_bid = max(our_bid, best_ask - 1)  # More competitive bid
                
        # Convert prices to integers - IMC Prosperity requires integer prices
        our_bid = int(our_bid)
        our_ask = int(our_ask)
        
        # Calculate order sizes - smaller for options than for the underlying
        base_volume = int(POSITION_LIMITS[option_product] * 0.05)  # 5% of position limit
        
        # Adjust volume based on delta and days to expiry
        # Trade less as expiry approaches or for deep ITM/OTM options
        volume_adjust = abs(delta) * min(1, days_to_expiry / 3)
        base_volume = max(1, int(base_volume * volume_adjust))
        
        # Calculate imbalance for asymmetric sizing
        imbalance = self.calculate_order_imbalance(order_depth)
        
        # Adjust volumes based on imbalance
        bid_volume = int(base_volume * (1 + imbalance * 0.5))
        ask_volume = int(base_volume * (1 - imbalance * 0.5))
        
        # Ensure we don't exceed position limits
        bid_volume = min(bid_volume, POSITION_LIMITS[option_product] - current_position)
        ask_volume = min(ask_volume, POSITION_LIMITS[option_product] + current_position)
        
        # Only place orders if volumes are positive
        if bid_volume > 0:
            orders.append(Order(option_product, our_bid, bid_volume))
        if ask_volume > 0:
            orders.append(Order(option_product, our_ask, -ask_volume))
            
        return orders
        
    def option_arbitrage_orders(self, option_products: List[str], state: TradingState, 
                              underlying_price: float, days_to_expiry: int) -> Dict[str, List[Order]]:
        """
        Generate arbitrage orders between options and between options and the underlying.
        
        Returns a dictionary of orders for each product involved in arbitrage opportunities.
        """
        orders_dict = {}
        
        # Only proceed if we have sufficient option products available
        if not all(option in state.order_depths for option in option_products):
            return orders_dict
            
        # Calculate implied volatilities from market prices of each option
        implied_vols = {}
        implied_skew = {}
        
        for option in option_products:
            # Skip if insufficient data in the order book
            if not state.order_depths[option].buy_orders or not state.order_depths[option].sell_orders:
                continue
                
            mid_price = self.calculate_mid_price(state.order_depths[option])
            if mid_price is None:
                continue
                
            strike = OPTION_STRIKES[option]
            
            # Simple approximation - in a real system we'd use a more robust IV calculation
            # Try different volatilities to find the one that matches market price
            best_vol = OPTION_VOLATILITY
            best_diff = float('inf')
            
            for vol in [v * 0.01 for v in range(10, 50)]:  # 0.10 to 0.50
                theo_price = self.calculate_option_price(
                    underlying_price=underlying_price,
                    strike=strike,
                    days_to_expiry=days_to_expiry,
                    volatility=vol
                )
                
                diff = abs(theo_price - mid_price)
                if diff < best_diff:
                    best_diff = diff
                    best_vol = vol
            
            implied_vols[option] = best_vol
            
            # Calculate implied skew (relative to ATM option)
            implied_skew[option] = best_vol - implied_vols.get(VRV_10000, OPTION_VOLATILITY)
            
        # If we have enough data, look for arbitrage opportunities
        if len(implied_vols) >= 3:
            # Calculate average implied volatility to use as a reference
            avg_vol = sum(implied_vols.values()) / len(implied_vols)
            
            # Look for options that are significantly mispriced relative to others
            for option in option_products:
                if option not in implied_vols:
                    continue
                    
                # Skip if the order book is too thin
                if (len(state.order_depths[option].buy_orders) < 2 or
                    len(state.order_depths[option].sell_orders) < 2):
                    continue
                
                # If implied vol is much higher than average, consider selling
                vol_diff = implied_vols[option] - avg_vol
                current_position = state.position.get(option, 0)
                
                # Apply more conservative pricing for arbitrage to ensure profitability
                if vol_diff > 0.03 and current_position < POSITION_LIMITS[option] * 0.8:
                    # Option potentially overpriced - sell it
                    best_bid = max(state.order_depths[option].buy_orders.keys())
                    # Only if the best bid is still above our theoretical price
                    strike = OPTION_STRIKES[option]
                    theo_price = self.calculate_option_price(
                        underlying_price=underlying_price,
                        strike=strike,
                        days_to_expiry=days_to_expiry,
                        volatility=avg_vol
                    )
                    
                    if best_bid > theo_price * 1.02:  # 2% cushion
                        # Calculate volume - more conservative for arbitrage
                        volume = min(5, POSITION_LIMITS[option] - current_position)
                        orders = orders_dict.get(option, [])
                        orders.append(Order(option, best_bid, -volume))
                        orders_dict[option] = orders
                        
                elif vol_diff < -0.03 and current_position > -POSITION_LIMITS[option] * 0.8:
                    # Option potentially underpriced - buy it
                    best_ask = min(state.order_depths[option].sell_orders.keys())
                    # Only if the best ask is still below our theoretical price
                    strike = OPTION_STRIKES[option]
                    theo_price = self.calculate_option_price(
                        underlying_price=underlying_price,
                        strike=strike,
                        days_to_expiry=days_to_expiry,
                        volatility=avg_vol
                    )
                    
                    if best_ask < theo_price * 0.98:  # 2% cushion
                        # Calculate volume - more conservative for arbitrage
                        volume = min(5, POSITION_LIMITS[option] + current_position)
                        orders = orders_dict.get(option, [])
                        orders.append(Order(option, best_ask, volume))
                        orders_dict[option] = orders
                        
        return orders_dict
