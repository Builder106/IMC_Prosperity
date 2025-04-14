# Plan: Implement VOLCANIC_ROCK Trading Strategy in round_3_alpha.py

## Objective

Implement a specific trading strategy for the `VOLCANIC_ROCK` product in `src/algorithms/round_3/scripts/round_3_alpha.py`. This strategy should leverage the product's high volatility, narrow spreads, and significant price movements, incorporating momentum trading and tight-spread market making, while using the existing increased position scale (1.2).

## Analysis

*   **`round_3_alpha.py`:** The code already defines parameters for `VOLCANIC_ROCK` (lines 235-245) that align well with the request (e.g., `position_scale: 1.2`, `spread_capture: 1.2`, momentum-related parameters). However, the core logic in `Trader.run` currently applies standard market making and mean reversion to all "regular" products and doesn't explicitly use the defined momentum parameters for `VOLCANIC_ROCK`.
*   **`round_3_web.json`:** The data for `VOLCANIC_ROCK` (lines 59-60) confirms the characteristics: `volatility: 18.6785` and `avg_spread: 1.4107`.
*   **Backtesting:** The results show `VOLCANIC_ROCK` has high profit potential but also risk, reinforcing the need for a strategy that can capture strong moves but manage risk.
*   **Parameters:** We will use the existing parameters defined in the code:
    *   `momentum_lookback`: 5
    *   `momentum_threshold`: 0.0003
    *   `momentum_scale`: 2.0
    *   `position_scale`: 1.2
    *   `spread_capture`: 1.2
    *   `aggressive_market_taking`: True

## Implementation Strategy

Introduce specific logic for `VOLCANIC_ROCK` that implements the momentum strategy while retaining the tight-spread market making when momentum is not strong.

1.  **Implement Momentum Calculation:**
    *   Add a new method to the `Trader` class, like `calculate_momentum(product, prices_deque)`, which calculates a momentum score based on recent price changes (using `momentum_lookback`).
2.  **Implement Momentum Trading Logic:**
    *   Add another method, like `generate_momentum_orders(product, momentum_score, mid_price, current_position, order_depth)`, which:
        *   Checks if the `momentum_score` exceeds `momentum_threshold`.
        *   If yes, calculates the target position size using `momentum_scale` and the overall `position_scale` (1.2).
        *   Generates aggressive buy/sell orders (market orders or crossing the spread slightly, based on `aggressive_market_taking: True`) to achieve the target position.
3.  **Update `Trader.run` Method:**
    *   Modify the main loop (around line 1160) to add a specific condition for `product == VOLCANIC_ROCK`.
    *   Inside this condition:
        *   Calculate the momentum score using `calculate_momentum`.
        *   If the momentum score is above the threshold, call `generate_momentum_orders` to get the trading orders for `VOLCANIC_ROCK`.
        *   If the momentum score is *below* the threshold, fall back to the existing `market_make_orders` function, ensuring it uses the tight `spread_capture` of 1.2 for `VOLCANIC_ROCK`. (This handles the "Tight-spread market making" aspect when momentum isn't active).

## Visual Plan (Mermaid Diagram)

```mermaid
graph TD
    A[Start Trader.run for VOLCANIC_ROCK] --> B{Calculate Momentum Score};
    B --> C{Momentum > Threshold?};
    C -- Yes --> D[Generate Momentum Orders (Aggressive, Scaled)];
    C -- No --> E[Generate Market Making Orders (Tight Spread)];
    D --> F[Add Orders to Result];
    E --> F;
```

## Summary of Changes

1.  Add two new methods to the `Trader` class: `calculate_momentum` and `generate_momentum_orders`.
2.  Modify the `Trader.run` method to include specific logic for `VOLCANIC_ROCK`, choosing between momentum trading and tight-spread market making based on the calculated momentum score.
3.  Ensure the existing `PARAMS` for `VOLCANIC_ROCK` are used correctly by the new logic.