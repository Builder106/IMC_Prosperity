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

        # Position limits for each product
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
            'RAINFOREST_RESIN': {
                'volatility': 2.97,
                'spread_capture': 5.5,
                'mean_reversion_strength': 0.4,
                'sma_window': 12,
                'position_scale': 1.5
            },
            'KELP': {
                'volatility': 1.74,
                'spread_capture': 2.1,
                'mean_reversion_strength': 0.6,
                'sma_window': 10,
                'position_scale': 0.6
            },
            'SQUID_INK': {
                'volatility': 9.22,
                'spread_capture': 1.5,
                'mean_reversion_strength': 0.8,
                'sma_window': 8,
                'position_scale': 1.0
            },
            'CROISSANTS': {
                'volatility': 1.95,
                'spread_capture': 1.0,
                'mean_reversion_strength': 0.4,
                'sma_window': 10,
                'position_scale': 0.8
            },
            'JAMS': {
                'volatility': 5.89,
                'spread_capture': 1.5,
                'mean_reversion_strength': 0.5,
                'sma_window': 12,
                'position_scale': 1.5
            },
            'DJEMBES': {
                'volatility': 26.92,
                'spread_capture': 1.5,
                'mean_reversion_strength': 0.4,
                'sma_window': 15,
                'position_scale': 0.6
            },
            'PICNIC_BASKET1': {
                'volatility': 48.59,
                'spread_capture': 5.5,
                'mean_reversion_strength': 0.2,
                'sma_window': 10,
                'position_scale': 0.5
            },
            'PICNIC_BASKET2': {
                'volatility': 12.57,
                'spread_capture': 3.5,
                'mean_reversion_strength': 0.3,
                'sma_window': 10,
                'position_scale': 0.4
            },
            'VOLCANIC_ROCK': {
                'volatility': 18.68,
                'spread_capture': 1.2,
                'mean_reversion_strength': 0.1,
                'sma_window': 8,
                'position_scale': 1.2,
                'momentum_lookback': 5,
                'momentum_threshold': 0.0003,
                'momentum_scale': 2.0,
                'aggressive_market_taking': True
            },
            'MAGNIFICENT_MACARONS': {
                'volatility': 25.0,  # Will be adjusted based on observations
                'spread_capture': 2.0,
                'mean_reversion_strength': 0.3,
                'sma_window': 10,
                'position_scale': 0.8,
                'storage_cost': 0.1,
                'conversion_limit': 10
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
        """Estimate the fair price of a product based on order book and history"""
        if not order_depth.buy_orders and not order_depth.sell_orders:
            if product in self.mid_price_history and len(self.mid_price_history[product]) > 0:
                return self.mid_price_history[product][-1]
            return None
            
        # Calculate mid price from current order book
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        
        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
        elif best_bid:
            mid_price = best_bid
        elif best_ask:
            mid_price = best_ask
        else:
            return None
            
        # Update price history
        if product not in self.mid_price_history:
            self.mid_price_history[product] = []
        self.mid_price_history[product].append(mid_price)
        
        # Calculate SMA if enough history
        if len(self.mid_price_history[product]) >= self.params[product]['sma_window']:
            sma = sum(self.mid_price_history[product][-self.params[product]['sma_window']:]) / self.params[product]['sma_window']
            self.sma_values[product] = sma
            
        return mid_price

    def calculate_order_volume(self, product: str, fair_price: float, order_depth: OrderDepth, position: int) -> dict:
        """Calculate the volume to trade based on strategy parameters"""
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
        
        # Calculate price deviation from SMA if available
        price_deviation = 0
        if product in self.sma_values:
            price_deviation = (fair_price - self.sma_values[product]) / self.sma_values[product]
        
        # Base volume on position limits and current position
        max_buy_volume = position_limit - position
        max_sell_volume = position_limit + position
        
        # Scale volume based on spread and volatility
        base_volume = min(max_buy_volume, max_sell_volume) * params['position_scale']
        spread_scalar = params['spread_capture'] / max(spread, params['spread_capture'])
        volume_scalar = spread_scalar * (1 - abs(price_deviation) * params['mean_reversion_strength'])
        
        trade_volume = int(base_volume * volume_scalar)
        
        # Adjust orders based on mean reversion signal
        if price_deviation < 0:  # Price below SMA - buy signal
            buy_volume = min(trade_volume, max_buy_volume)
            if best_ask < fair_price * (1 + params['spread_capture']/10000):
                orders[best_ask] = buy_volume
                
        elif price_deviation > 0:  # Price above SMA - sell signal
            sell_volume = min(trade_volume, max_sell_volume)
            if best_bid > fair_price * (1 - params['spread_capture']/10000):
                orders[best_bid] = -sell_volume
                
        return orders

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
        
        # First handle individual components with market making
        for product in ['CROISSANTS', 'JAMS', 'DJEMBES']:
            if product in state.order_depths:
                orders = []
                order_depth = state.order_depths[product]
                fair_price = self.estimate_fair_price(product, order_depth)
                
                if fair_price:
                    position = state.position.get(product, 0)
                    orders_dict = self.calculate_order_volume(product, fair_price, order_depth, position)
                    
                    for price, volume in orders_dict.items():
                        orders.append(Order(product, price, volume))
                        
                all_orders[product] = orders
        
        # Then look for basket arbitrage opportunities
        for basket, components in [('PICNIC_BASKET1', self.basket1_components), 
                                 ('PICNIC_BASKET2', self.basket2_components)]:
            if basket not in state.order_depths:
                continue
                
            basket_orders = []
            order_depth = state.order_depths[basket]
            
            # Calculate basket fair value from components
            basket_value = 0
            can_arbitrage = True
            
            for component, quantity in components.items():
                if component not in state.order_depths:
                    can_arbitrage = False
                    break
                    
                comp_price = self.estimate_fair_price(component, state.order_depths[component])
                if not comp_price:
                    can_arbitrage = False
                    break
                    
                basket_value += comp_price * quantity
                
            if not can_arbitrage:
                continue
                
            # Check for arbitrage opportunities
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            
            position = state.position.get(basket, 0)
            max_position = self.position_limits[basket]
            
            # Buy basket, sell components
            if best_ask and best_ask < basket_value * 0.995:  # 0.5% profit threshold
                buy_volume = min(20, max_position - position)
                if buy_volume > 0:
                    basket_orders.append(Order(basket, best_ask, buy_volume))
                    
                    # Sell components
                    for component, quantity in components.items():
                        comp_orders = []
                        comp_depth = state.order_depths[component]
                        comp_position = state.position.get(component, 0)
                        
                        if comp_depth.buy_orders:
                            sell_price = max(comp_depth.buy_orders.keys())
                            sell_volume = min(buy_volume * quantity, 
                                           self.position_limits[component] + comp_position)
                            comp_orders.append(Order(component, sell_price, -sell_volume))
                            
                        if component in all_orders:
                            all_orders[component].extend(comp_orders)
                        else:
                            all_orders[component] = comp_orders
            
            # Sell basket, buy components
            if best_bid and best_bid > basket_value * 1.005:  # 0.5% profit threshold
                sell_volume = min(20, max_position + position)
                if sell_volume > 0:
                    basket_orders.append(Order(basket, best_bid, -sell_volume))
                    
                    # Buy components
                    for component, quantity in components.items():
                        comp_orders = []
                        comp_depth = state.order_depths[component]
                        comp_position = state.position.get(component, 0)
                        
                        if comp_depth.sell_orders:
                            buy_price = min(comp_depth.sell_orders.keys())
                            buy_volume = min(sell_volume * quantity,
                                          self.position_limits[component] - comp_position)
                            comp_orders.append(Order(component, buy_price, buy_volume))
                            
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
        
        # Handle Magnificent Macarons separately due to unique features
        mm_orders = self.handle_magnificent_macarons(state)
        if mm_orders:
            result['MAGNIFICENT_MACARONS'] = mm_orders
            
        # Handle Volcanic Rock and options
        option_orders = self.handle_volcanic_options(state)
        result.update(option_orders)
        
        # Handle Picnic Baskets and their components
        basket_orders = self.handle_picnic_baskets(state)
        result.update(basket_orders)
        
        # Handle remaining products with basic market making
        for product in ['RAINFOREST_RESIN', 'KELP', 'SQUID_INK']:
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
