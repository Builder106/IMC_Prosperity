import json
from typing import Dict, List, Tuple, Any
from datamodel import OrderDepth, TradingState, Order, Observation, Symbol, Trade, Listing, ProsperityEncoder
import numpy as np

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


logger = Logger()

class Trader:
    def __init__(self):
        # Product symbols
        self.products = {
            # Round 1
            'RAINFOREST_RESIN': 'RR',
            'KELP': 'KELP',
            'SQUID_INK': 'SI',
            # Round 2
            'PICNIC_BASKET1': 'PB1',
            'PICNIC_BASKET2': 'PB2', 
            'CROISSANTS': 'CR',
            'JAMS': 'JAM',
            'DJEMBES': 'DJ',
            # Round 3
            'VOLCANIC_ROCK': 'VR',
            'VOLCANIC_ROCK_VOUCHER_9500': 'VRV_9500',
            'VOLCANIC_ROCK_VOUCHER_9750': 'VRV_9750',
            'VOLCANIC_ROCK_VOUCHER_10000': 'VRV_10000',
            'VOLCANIC_ROCK_VOUCHER_10250': 'VRV_10250',
            'VOLCANIC_ROCK_VOUCHER_10500': 'VRV_10500',
            # Round 4
            'MAGNIFICENT_MACARONS': 'MM'
        }

        # Position limits
        self.position_limits = {
            'RAINFOREST_RESIN': 50,
            'KELP': 50,
            'SQUID_INK': 50,
            'PICNIC_BASKET1': 60,
            'PICNIC_BASKET2': 100,
            'CROISSANTS': 250,
            'JAMS': 350,
            'DJEMBES': 60,
            'VOLCANIC_ROCK': 400,
            'VOLCANIC_ROCK_VOUCHER_9500': 200,
            'VOLCANIC_ROCK_VOUCHER_9750': 200,
            'VOLCANIC_ROCK_VOUCHER_10000': 200,
            'VOLCANIC_ROCK_VOUCHER_10250': 200,
            'VOLCANIC_ROCK_VOUCHER_10500': 200,
            'MAGNIFICENT_MACARONS': 75
        }

        # Strategy parameters
        self.params = {
            'VOLCANIC_ROCK': {
                'volatility': 98.37,        # Actual volatility from data
                'spread_capture': 0.6,      # Much tighter spreads
                'mean_reversion_strength': 0.05,  # Very weak mean reversion
                'sma_window': 20,          # Longer window for stability
                'position_scale': 0.3,     # Smaller positions
                'min_spread': 1.2,         # Minimum spread requirement
                'momentum_window': 5,      # Short momentum window
                'momentum_threshold': 0.0005,  # Tight momentum threshold
                'max_position_util': 0.4   # Very conservative positions
            },
            'CROISSANTS': {
                'volatility': 9.27,          
                'spread_capture': 0.4,       # Much tighter spreads
                'mean_reversion_strength': 0.2,  # Weaker mean reversion
                'sma_window': 10,           
                'position_scale': 0.3,      # Much smaller positions
                'min_spread': 1.0,
                'top_book_threshold': 0.85,
                'min_book_depth': 100,      # Minimum liquidity requirement
                'max_position_util': 0.5    # Conservative position usage
            },
            'JAMS': {
                'volatility': 9.97,          
                'spread_capture': 0.5,       # Tighter spreads
                'mean_reversion_strength': 0.15,  # Weaker mean reversion
                'sma_window': 12,            
                'position_scale': 0.25,      # Much smaller positions
                'min_spread': 1.2,           
                'min_book_depth': 150,
                'max_position_util': 0.4     # Very conservative positions
            },
            'DJEMBES': {
                'volatility': 14.81,         
                'spread_capture': 0.3,       # Much tighter spreads
                'mean_reversion_strength': 0.1,  # Very weak mean reversion
                'sma_window': 15,           
                'position_scale': 0.2,      # Much smaller positions
                'min_spread': 1.0,          
                'top_book_threshold': 0.85,
                'momentum_window': 5,
                'max_position_util': 0.3    # Extremely conservative
            },
            'RAINFOREST_RESIN': {
                'volatility': 2.97,          # Based on trading data
                'spread_capture': 1.2,       # Conservative spread target
                'mean_reversion_strength': 0.4,
                'sma_window': 12,
                'position_scale': 0.6,       # Moderate position sizing
                'min_spread': 0.8,           # Minimum spread requirement
                'momentum_window': 5,        # Short momentum lookback
                'momentum_threshold': 0.0003, # Momentum signal threshold
                'max_position_util': 0.7     # Position limit usage
            },
            'KELP': {
                'volatility': 1.74,          # Based on trading data
                'spread_capture': 1.3,       # Tighter spreads
                'mean_reversion_strength': 0.8,
                'sma_window': 6,
                'position_scale': 0.8,
                'min_spread': 0.5,
                'top_book_threshold': 0.7,
                'book_imbalance_threshold': 0.05,
                'max_position_util': 0.8
            },
            'SQUID_INK': {
                'volatility': 9.22,          # Based on trading data
                'spread_capture': 1.0,
                'mean_reversion_strength': 0.6,
                'sma_window': 5,
                'position_scale': 0.7,
                'min_spread': 0.8,
                'momentum_window': 4,
                'momentum_threshold': 0.0003,
                'min_order_size': 5,
                'max_position_util': 0.7
            },
            'PICNIC_BASKET1': {
                'volatility': 48.59,
                'spread_capture': 5.5,
                'mean_reversion_strength': 0.2,
                'sma_window': 10,
                'position_scale': 0.5,
                'momentum_window': 8,
                'momentum_threshold': 0.0004,
                'min_spread': 5.0,
                'max_position_util': 0.6
            },
            'PICNIC_BASKET2': {
                'volatility': 12.57,
                'spread_capture': 3.5,
                'mean_reversion_strength': 0.3,
                'sma_window': 10,
                'position_scale': 0.4,
                'momentum_window': 6,
                'momentum_threshold': 0.0003,
                'min_spread': 3.0,
                'max_position_util': 0.7
            },
            'MAGNIFICENT_MACARONS': {
                'volatility': 25.0,
                'spread_capture': 2.0,
                'mean_reversion_strength': 0.3,
                'sma_window': 10,
                'position_scale': 0.8,
                'storage_cost': 0.1,
                'conversion_limit': 10,
                'momentum_window': 5,
                'momentum_threshold': 0.0004,
                'max_position_util': 0.6
            }
        }

        # Basket composition
        self.basket1_components = {
            'DJEMBES': 1,
            'JAMS': 3, 
            'CROISSANTS': 6
        }
        
        self.basket2_components = {
            'JAMS': 2,
            'CROISSANTS': 4
        }

        # Option parameters
        self.option_params = {
            'days_to_expiry': 7,
            'risk_free_rate': 0.01,
            'base_volatility': 0.45,
            'strikes': {
                'VOLCANIC_ROCK_VOUCHER_9500': 9500,
                'VOLCANIC_ROCK_VOUCHER_9750': 9750,
                'VOLCANIC_ROCK_VOUCHER_10000': 10000,
                'VOLCANIC_ROCK_VOUCHER_10250': 10250,
                'VOLCANIC_ROCK_VOUCHER_10500': 10500
            }
        }

        # Price history for each product
        self.price_history = {}
        self.mid_price_history = {}
        
        # SMA trackers
        self.sma_values = {}

    def calculate_option_metrics(self, state: TradingState, product: str):
        """Calculate implied volatility and other option metrics"""
        if 'VOLCANIC_ROCK_VOUCHER' not in product:
            return None
            
        strike = self.option_params['strikes'][product]
        days_left = max(0, self.option_params['days_to_expiry'] - state.timestamp // 1000000)
        
        if 'VOLCANIC_ROCK' not in state.order_depths:
            return None
            
        # Get underlying price
        vr_depth = state.order_depths['VOLCANIC_ROCK']
        if not vr_depth.buy_orders and not vr_depth.sell_orders:
            return None
            
        underlying_price = self.estimate_fair_price('VOLCANIC_ROCK', vr_depth)
        
        time_to_expiry = days_left / 365  # Convert to years
        if time_to_expiry <= 0:
            return None
            
        return {
            'strike': strike,
            'underlying': underlying_price,
            'time_to_expiry': time_to_expiry,
            'implied_vol': self.option_params['base_volatility']
        }

    def estimate_fair_price(self, product: str, order_depth: OrderDepth) -> float:
        """Estimate the fair price of a product based on order book, history, and microstructure"""
        if not order_depth.buy_orders and not order_depth.sell_orders:
            if product in self.mid_price_history and len(self.mid_price_history[product]) > 0:
                return self.mid_price_history[product][-1]
            return None

        # Get order book stats
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        total_bid_volume = sum(abs(vol) for vol in order_depth.buy_orders.values())
        total_ask_volume = sum(abs(vol) for vol in order_depth.sell_orders.values())
        
        # Calculate volume-weighted prices
        bid_vwap = sum(p * abs(v) for p, v in order_depth.buy_orders.items()) / total_bid_volume if total_bid_volume > 0 else best_bid
        ask_vwap = sum(p * abs(v) for p, v in order_depth.sell_orders.items()) / total_ask_volume if total_ask_volume > 0 else best_ask

        # Special handling for VOLCANIC_ROCK
        if product == 'VOLCANIC_ROCK':
            # Short-term price momentum (5 ticks)
            momentum = 0
            if product in self.mid_price_history:
                window = min(5, len(self.mid_price_history[product]))
                if window > 1:
                    recent_prices = self.mid_price_history[product][-window:]
                    momentum = (recent_prices[-1] / recent_prices[0] - 1)
            
            # Calculate imbalance impact
            imbalance = 0
            if total_bid_volume + total_ask_volume > 0:
                raw_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
                # Dampen imbalance impact based on total volume
                min_vol = 100  # Minimum volume for full impact
                vol_scalar = min(1.0, (total_bid_volume + total_ask_volume) / (2 * min_vol))
                imbalance = raw_imbalance * vol_scalar
            
            # Price is weighted combination of:
            # 1. Current mid price (most weight)
            # 2. VWAP (some weight if good volume)
            # 3. Momentum adjustment (small weight)
            # 4. Imbalance adjustment (tiny weight)
            
            vwap = (bid_vwap + ask_vwap) / 2
            mid_weight = 0.7
            vwap_weight = 0.2
            momentum_weight = 0.08
            imbalance_weight = 0.02
            
            fair_price = (
                mid_price * mid_weight +
                vwap * vwap_weight +
                mid_price * (1 + momentum) * momentum_weight +
                mid_price * (1 + imbalance * 0.1) * imbalance_weight
            )
            
        # Special handling for basket components
        elif product in ['CROISSANTS', 'JAMS', 'DJEMBES']:
            # These need more stable pricing due to basket arb
            if product in self.sma_values:
                sma = self.sma_values[product]
                # Heavily weight towards SMA
                fair_price = sma * 0.8 + mid_price * 0.2
                
                # Tiny adjustment for order book pressure
                if total_bid_volume + total_ask_volume > 0:
                    imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
                    fair_price *= (1 + imbalance * 0.0001)  # Very small impact
            else:
                fair_price = mid_price
                
        # Regular handling for other products
        else:
            if product in self.sma_values:
                sma = self.sma_values[product]
                params = self.params[product]
                
                # Calculate imbalance
                imbalance = 0
                if total_bid_volume + total_ask_volume > 0:
                    imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
                
                # Weight between SMA and current price
                sma_weight = 0.5  # Base weight
                if abs(imbalance) > 0.2:  # Strong imbalance
                    # Reduce SMA weight when strong imbalance exists
                    sma_weight = 0.3
                
                fair_price = sma * sma_weight + mid_price * (1 - sma_weight)
            else:
                fair_price = mid_price
            
        # Update price history
        if product not in self.mid_price_history:
            self.mid_price_history[product] = []
        self.mid_price_history[product].append(fair_price)
        
        # Calculate SMA if enough history
        if len(self.mid_price_history[product]) >= self.params[product]['sma_window']:
            window = self.params[product]['sma_window']
            sma = sum(self.mid_price_history[product][-window:]) / window
            self.sma_values[product] = sma
            
        return fair_price

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate mid price from order book"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2
        
    def calculate_vwap(self, order_depth: OrderDepth, side: str) -> float:
        """Calculate volume-weighted average price for a given side"""
        if side == 'buy' and not order_depth.buy_orders:
            return None
        if side == 'sell' and not order_depth.sell_orders:
            return None
            
        orders = order_depth.buy_orders if side == 'buy' else order_depth.sell_orders
        total_volume = sum(abs(volume) for volume in orders.values())
        if total_volume == 0:
            return None
            
        weighted_price = sum(price * abs(volume) for price, volume in orders.items())
        return weighted_price / total_volume

    def calculate_order_volume(self, product: str, fair_price: float, order_depth: OrderDepth, position: int) -> dict:
        """Calculate the volume to trade based on strategy parameters and market conditions"""
        params = self.params[product]
        position_limit = self.position_limits[product]
        
        # Initialize final orders dictionary
        orders = {}
        
        # Get best bid/ask from order book
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        
        if not best_bid or not best_ask:
            return orders
            
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2

        # Special handling for VOLCANIC_ROCK due to high volatility
        if product == 'VOLCANIC_ROCK':
            # Calculate total book depth
            total_bid_volume = sum(abs(vol) for vol in order_depth.buy_orders.values())
            total_ask_volume = sum(abs(vol) for vol in order_depth.sell_orders.values())
            
            # Only trade if enough liquidity
            min_depth = 100
            if total_bid_volume < min_depth or total_ask_volume < min_depth:
                return orders
                
            # Calculate weighted average prices
            bid_wap = sum(p * abs(v) for p, v in order_depth.buy_orders.items()) / total_bid_volume
            ask_wap = sum(p * abs(v) for p, v in order_depth.sell_orders.items()) / total_ask_volume
            
            # Calculate price momentum
            if product in self.mid_price_history:
                window = params['momentum_window']
                if len(self.mid_price_history[product]) >= window:
                    momentum = (mid_price / self.mid_price_history[product][-window] - 1)
                    
                    # Trade only in momentum direction with strict risk controls
                    max_pos = int(position_limit * params['max_position_util'])
                    base_vol = int(min(5, position_limit * 0.05))  # Very small base volume
                    
                    if abs(momentum) > params['momentum_threshold']:
                        if momentum > 0 and position < max_pos:
                            # Strong upward momentum - buy carefully
                            buy_vol = int(min(base_vol, max_pos - position))
                            if best_ask < ask_wap:  # Only buy below WAP
                                return {best_ask: buy_vol}
                                
                        elif momentum < 0 and position > -max_pos:
                            # Strong downward momentum - sell carefully
                            sell_vol = int(min(base_vol, max_pos + position))
                            if best_bid > bid_wap:  # Only sell above WAP
                                return {best_bid: -sell_vol}
            
            return orders  # Don't trade if no clear signal
            
        # Regular handling for other products continues...
        # Calculate price deviation from SMA if available
        price_deviation = 0
        if product in self.sma_values:
            price_deviation = (fair_price - self.sma_values[product]) / self.sma_values[product]
        
        # Product-specific signal calculations
        if product in ['CROISSANTS', 'JAMS', 'DJEMBES']:
            # More conservative trading for basket components
            total_bid_volume = sum(abs(vol) for vol in order_depth.buy_orders.values())
            total_ask_volume = sum(abs(vol) for vol in order_depth.sell_orders.values())
            
            # Require minimum liquidity
            min_depth = params.get('min_book_depth', 50)
            if total_bid_volume < min_depth or total_ask_volume < min_depth:
                return orders
                
            # Calculate order book imbalance
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            # Only trade if imbalance supports our direction
            max_pos = position_limit * params['max_position_util']
            base_vol = position_limit * params['position_scale']
            
            if abs(imbalance) > 0.1:  # Significant imbalance
                max_pos = int(position_limit * params['max_position_util'])
                if imbalance > 0 and position < max_pos:
                    # Strong buying pressure
                    buy_vol = int(min(base_vol * abs(imbalance), max_pos - position))
                    if best_ask < fair_price * (1 + params['spread_capture']/10000):
                        return {best_ask: buy_vol}
                        
                elif imbalance < 0 and position > -max_pos:
                    # Strong selling pressure
                    sell_vol = int(min(base_vol * abs(imbalance), max_pos + position))
                    if best_bid > fair_price * (1 - params['spread_capture']/10000):
                        return {best_bid: -sell_vol}
                        
            return orders  # Don't trade without clear signal

        elif product == 'RAINFOREST_RESIN':
            if product in self.mid_price_history and len(self.mid_price_history[product]) >= params['momentum_window']:
                momentum = (fair_price / self.mid_price_history[product][-params['momentum_window']] - 1)
                if abs(momentum) > params['momentum_threshold']:
                    price_deviation = momentum
        
        elif product == 'KELP':
            total_bid_volume = sum(abs(vol) for vol in order_depth.buy_orders.values())
            total_ask_volume = sum(abs(vol) for vol in order_depth.sell_orders.values())
            top_bid_concentration = abs(order_depth.buy_orders[best_bid]) / total_bid_volume if total_bid_volume > 0 else 0
            top_ask_concentration = abs(order_depth.sell_orders[best_ask]) / total_ask_volume if total_ask_volume > 0 else 0
            
            if top_bid_concentration < params['top_book_threshold'] or top_ask_concentration < params['top_book_threshold']:
                return orders
        
        elif product == 'SQUID_INK':
            if product in self.mid_price_history and len(self.mid_price_history[product]) >= params['momentum_window']:
                recent_prices = self.mid_price_history[product][-params['momentum_window']:]
                price_trend = sum(1 if recent_prices[i] < recent_prices[i+1] else -1 
                                for i in range(len(recent_prices)-1))
                if abs(price_trend) >= params['momentum_window'] - 1:
                    price_deviation *= 1.5
        
        # Calculate market signals
        order_imbalance = self.calculate_order_imbalance(order_depth)
        
        # Base volume calculations with position limit constraints
        max_buy_volume = position_limit - position
        max_sell_volume = position_limit + position
        
        # Scale volume based on market conditions
        base_volume = min(max_buy_volume, max_sell_volume) * params['position_scale']
        spread_scalar = params['spread_capture'] / max(spread, params.get('min_spread', params['spread_capture']))
        
        # Dynamic volume adjustment based on price deviation and order imbalance
        price_signal = -price_deviation * params['mean_reversion_strength']
        imbalance_signal = order_imbalance * 0.2
        
        # Combine signals for final volume scalar
        volume_scalar = spread_scalar * (1 + price_signal + imbalance_signal)
        volume_scalar = max(0.2, min(2.0, volume_scalar))
        
        trade_volume = int(base_volume * volume_scalar)
        
        # Enforce minimum order size if specified
        min_size = params.get('min_order_size', 1)
        if trade_volume < min_size:
            return orders
        
        # Get available liquidity
        buy_liquidity = sum(abs(vol) for _, vol in order_depth.sell_orders.items())
        sell_liquidity = sum(vol for _, vol in order_depth.buy_orders.items())
        
        # Maximum position utilization
        max_util = params.get('max_position_util', 0.8)
        
        # Place buy orders
        if position < position_limit * max_util:
            buy_volume = int(min(trade_volume, max_buy_volume, buy_liquidity))
            if buy_volume >= min_size and best_ask < fair_price * (1 + params['spread_capture']/10000):
                orders[best_ask] = buy_volume
        
        # Place sell orders  
        if position > -position_limit * max_util:
            sell_volume = int(min(trade_volume, max_sell_volume, sell_liquidity))
            if sell_volume >= min_size and best_bid > fair_price * (1 - params['spread_capture']/10000):
                orders[best_bid] = -sell_volume
                
        return orders

    def calculate_order_imbalance(self, order_depth: OrderDepth) -> float:
        """Calculate order imbalance from the order book."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
            
        total_buy_volume = sum(order_depth.buy_orders.values())
        total_sell_volume = sum(abs(volume) for volume in order_depth.sell_orders.values())
        
        if total_buy_volume + total_sell_volume == 0:
            return 0
            
        return (total_buy_volume - total_sell_volume) / (total_buy_volume + total_sell_volume)

    def handle_magnificent_macarons(self, state: TradingState) -> List[Order]:
        """Special handling for Magnificent Macarons with observable factors"""
        orders = []
        product = 'MAGNIFICENT_MACARONS'
        
        if product not in state.order_depths:
            return orders
            
        position = state.position.get(product, 0)
        order_depth = state.order_depths[product]
        observation = state.observations.conversionObservations.get(product)
        
        if not observation:
            return orders
            
        # Calculate fair price incorporating all factors
        sugar_impact = 0.3 * (observation.sugarPrice - 200) / 200  # Normalize around 200
        sunlight_impact = -0.2 * (observation.sunlightIndex - 50) / 50  # Normalize around 50
        transport_impact = -0.1 * observation.transportFees
        tariff_impact = -0.15 * (observation.importTariff + observation.exportTariff)
        
        # Base price from order book
        fair_price = self.estimate_fair_price(product, order_depth)
        if not fair_price:
            return orders
            
        # Adjust fair price based on observable factors
        adjusted_price = fair_price * (1 + sugar_impact + sunlight_impact + transport_impact + tariff_impact)
        
        # Get conversion prices
        convert_buy_price = observation.askPrice + observation.transportFees + observation.importTariff
        convert_sell_price = observation.bidPrice - observation.transportFees - observation.exportTariff
        
        # Market making orders
        mm_orders = self.calculate_order_volume(product, adjusted_price, order_depth, position)
        
        # Convert orders to Order objects
        for price, volume in mm_orders.items():
            orders.append(Order(product, price, volume))
            
        # Check if conversion is profitable
        if position > 0 and convert_sell_price > adjusted_price * 1.001:  # 0.1% profit threshold
            # Sell to Pristine Cuisine
            conversion_volume = min(position, self.params[product]['conversion_limit'])
            return [Order(product, convert_sell_price, -conversion_volume)]
            
        elif position < 0 and convert_buy_price < adjusted_price * 0.999:  # 0.1% profit threshold
            # Buy from Pristine Cuisine
            conversion_volume = min(-position, self.params[product]['conversion_limit'])
            return [Order(product, convert_buy_price, conversion_volume)]
            
        return orders

    def handle_volcanic_options(self, state: TradingState) -> Dict[str, List[Order]]:
        """Handle volcanic rock options trading"""
        all_orders = {}
        
        # First handle the underlying
        if 'VOLCANIC_ROCK' in state.order_depths:
            vr_orders = []
            order_depth = state.order_depths['VOLCANIC_ROCK']
            fair_price = self.estimate_fair_price('VOLCANIC_ROCK', order_depth)
            
            if fair_price:
                position = state.position.get('VOLCANIC_ROCK', 0)
                orders_dict = self.calculate_order_volume('VOLCANIC_ROCK', fair_price, order_depth, position)
                
                for price, volume in orders_dict.items():
                    vr_orders.append(Order('VOLCANIC_ROCK', price, volume))
                    
                all_orders['VOLCANIC_ROCK'] = vr_orders
        
        # Then handle each option
        for product in self.products:
            if 'VOLCANIC_ROCK_VOUCHER' in product and product in state.order_depths:
                option_orders = []
                order_depth = state.order_depths[product]
                metrics = self.calculate_option_metrics(state, product)
                
                if not metrics:
                    continue
                    
                # Simple market making for options
                position = state.position.get(product, 0)
                
                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                
                if best_bid and best_ask:
                    spread = best_ask - best_bid
                    mid_price = (best_bid + best_ask) / 2
                    
                    # Place orders with tighter spread
                    max_position = self.position_limits[product]
                    buy_volume = min(20, max_position - position)
                    sell_volume = min(20, max_position + position)
                    
                    if buy_volume > 0:
                        bid_price = int(best_bid + spread * 0.25)  # Convert to integer
                        option_orders.append(Order(product, bid_price, buy_volume))
                    if sell_volume > 0:
                        ask_price = int(best_ask - spread * 0.25)  # Convert to integer
                        option_orders.append(Order(product, ask_price, -sell_volume))
                        
                all_orders[product] = option_orders
                
        return all_orders

    def handle_picnic_baskets(self, state: TradingState) -> Dict[str, List[Order]]:
        """Handle picnic basket arbitrage opportunities"""
        all_orders = {}
        
        # Track spread and volume metrics for dynamic thresholds
        metrics = {}
        for product in ['CROISSANTS', 'JAMS', 'DJEMBES', 'PICNIC_BASKET1', 'PICNIC_BASKET2']:
            if product not in state.order_depths:
                continue
                
            depth = state.order_depths[product]
            if not depth.buy_orders or not depth.sell_orders:
                continue
                
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate effective volumes at different price levels
            buy_volume = {}
            sell_volume = {}
            total_buy = 0
            total_sell = 0
            
            for price, vol in sorted(depth.buy_orders.items(), reverse=True):
                total_buy += abs(vol)
                buy_volume[price] = total_buy
                
            for price, vol in sorted(depth.sell_orders.items()):
                total_sell += abs(vol)
                sell_volume[price] = total_sell
                
            metrics[product] = {
                'mid_price': mid_price,
                'spread': best_ask - best_bid,
                'spread_pct': (best_ask - best_bid) / mid_price,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'total_depth': min(total_buy, total_sell)
            }

        # Handle each basket separately with custom thresholds
        for basket, components in [('PICNIC_BASKET1', self.basket1_components), 
                                 ('PICNIC_BASKET2', self.basket2_components)]:
            if basket not in metrics:
                continue
                
            basket_metrics = metrics[basket]
            basket_orders = []
            
            # Calculate theoretical value and execution constraints
            theo_value = 0
            min_volume = float('inf')
            component_costs = {'buy': 0, 'sell': 0}
            can_trade = True
            
            for component, quantity in components.items():
                if component not in metrics:
                    can_trade = False
                    break
                    
                comp_metrics = metrics[component]
                theo_value += comp_metrics['mid_price'] * quantity
                
                # Track minimum executable volume
                comp_volume = comp_metrics['total_depth'] / quantity
                min_volume = min(min_volume, comp_volume)
                
                # Calculate actual execution costs
                component_costs['buy'] += comp_metrics['best_ask'] * quantity
                component_costs['sell'] += comp_metrics['best_bid'] * quantity
            
            if not can_trade:
                continue
                
            # Dynamic arbitrage thresholds based on components
            base_threshold = 0.003 if basket == 'PICNIC_BASKET2' else 0.004
            volume_factor = min(1.0, min_volume / 30)  # Scale based on depth
            spread_factor = 1.0 + sum(metrics[c]['spread_pct'] for c in components) / len(components)
            
            # Different thresholds for buy vs sell
            buy_threshold = theo_value * base_threshold * spread_factor / volume_factor
            sell_threshold = theo_value * base_threshold * spread_factor / volume_factor
            
            position = state.position.get(basket, 0)
            max_position = self.position_limits[basket]
            
            # Buy basket when profitable vs component costs
            if position < max_position and basket_metrics['best_ask'] < component_costs['sell'] - buy_threshold:
                # Calculate optimal trade size
                trade_size = min(
                    int(min_volume * 0.7),  # Use 70% of available volume
                    max_position - position,
                    15 if basket == 'PICNIC_BASKET1' else 20  # Size based on basket type
                )
                
                if trade_size >= 3:  # Minimum viable trade size
                    basket_orders.append(Order(basket, basket_metrics['best_ask'], trade_size))
                    
                    # Sell components
                    for component, quantity in components.items():
                        comp_orders = []
                        comp_metrics = metrics[component]
                        
                        # Calculate optimal component execution
                        comp_volume = trade_size * quantity
                        comp_orders.append(Order(component, comp_metrics['best_bid'], -comp_volume))
                        
                        if component in all_orders:
                            all_orders[component].extend(comp_orders)
                        else:
                            all_orders[component] = comp_orders
            
            # Sell basket when profitable vs component costs
            elif position > -max_position and basket_metrics['best_bid'] > component_costs['buy'] + sell_threshold:
                # Calculate optimal trade size
                trade_size = min(
                    int(min_volume * 0.7),
                    max_position + position,
                    15 if basket == 'PICNIC_BASKET1' else 20
                )
                
                if trade_size >= 3:
                    basket_orders.append(Order(basket, basket_metrics['best_bid'], -trade_size))
                    
                    # Buy components
                    for component, quantity in components.items():
                        comp_orders = []
                        comp_metrics = metrics[component]
                        
                        comp_volume = trade_size * quantity
                        comp_orders.append(Order(component, comp_metrics['best_ask'], comp_volume))
                        
                        if component in all_orders:
                            all_orders[component].extend(comp_orders)
                        else:
                            all_orders[component] = comp_orders
            
            if basket_orders:
                all_orders[basket] = basket_orders
                
        return all_orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Main method called by the simulator to get the trader's orders.
        """
        result = {}
        trader_data = ""
        conversions = 0
        
        # Group products by type for specialized handling
        component_products = ['CROISSANTS', 'JAMS', 'DJEMBES']
        basket_products = ['PICNIC_BASKET1', 'PICNIC_BASKET2']
        option_products = ['VOLCANIC_ROCK'] + [p for p in self.products if 'VOLCANIC_ROCK_VOUCHER' in p]
        basic_products = ['RAINFOREST_RESIN', 'KELP', 'SQUID_INK']
        
        # Handle individual components first since they affect basket arbitrage
        for product in component_products:
            if product in state.order_depths:
                orders = []
                order_depth = state.order_depths[product]
                fair_price = self.estimate_fair_price(product, order_depth)
                
                if fair_price:
                    position = state.position.get(product, 0)
                    # Get extra market data for better sizing
                    imbalance = self.calculate_order_imbalance(order_depth)
                    buy_vwap = self.calculate_vwap(order_depth, 'buy')
                    sell_vwap = self.calculate_vwap(order_depth, 'sell')
                    
                    # More aggressive sizing if favorable order imbalance
                    if (imbalance > 0.2 and position < 0) or (imbalance < -0.2 and position > 0):
                        self.params[product]['position_scale'] *= 1 + abs(imbalance)
                    
                    orders_dict = self.calculate_order_volume(product, fair_price, order_depth, position)
                    
                    for price, volume in orders_dict.items():
                        orders.append(Order(product, price, volume))
                        
                if orders:
                    result[product] = orders
        
        # Handle Magnificent Macarons with its special features
        mm_orders = self.handle_magnificent_macarons(state)
        if mm_orders:
            result['MAGNIFICENT_MACARONS'] = mm_orders
            
        # Handle Volcanic Rock and options
        option_orders = self.handle_volcanic_options(state)
        result.update(option_orders)
        
        # Handle Picnic Baskets after individual components
        basket_orders = self.handle_picnic_baskets(state)
        result.update(basket_orders)
        
        # Handle remaining products
        for product in basic_products:
            if product in state.order_depths:
                orders = []
                order_depth = state.order_depths[product]
                fair_price = self.estimate_fair_price(product, order_depth)
                
                if fair_price:
                    position = state.position.get(product, 0)
                    orders_dict = self.calculate_order_volume(product, fair_price, order_depth, position)
                    
                    for price, volume in orders_dict.items():
                        orders.append(Order(product, price, volume))
                        
                if orders:
                    result[product] = orders
        
        # Log results
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
