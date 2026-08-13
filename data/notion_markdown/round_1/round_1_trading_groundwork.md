# Round 1 - Trading groundwork

You have reached Intara.

Your team establishes a Trade Outpost and aims to earn a net profit of 200,000 XIRECs or more before the beginning of the third trading day.

The first goods available for trade are:

- `ASH_COATED_OSMIUM`
- `INTARIAN_PEPPER_ROOT`

Trading days on Intara last 72 hours.

## Round objective

Translate your first trading strategy into a Python program that trades `ASH_COATED_OSMIUM`and`INTARIAN_PEPPER_ROOT` on your behalf. Participate in the Exchange Auction to generate additional profit.

## Algorithmic trading challenge: First Intarian Goods

`INTARIAN_PEPPER_ROOT` is described as relatively steady.

`ASH_COATED_OSMIUM` is rumored to be more volatile and potentially pattern-driven.

Position limits:

- `ASH_COATED_OSMIUM`: 80
- `INTARIAN_PEPPER_ROOT`: 80

## Manual trading challenge: An Intarian Welcome

Opening auctions occur for:

- `DRYLAND_FLAX`
- `EMBER_MUSHROOM`

You submit your order last. No other bids or asks arrive after your submission.

### Auction rules

Submit a single limit order (price, quantity). The clearing price is chosen to:

1. Maximize total traded volume
2. Break ties by choosing the higher price

Orders execute at the clearing price according to price priority then time priority. Because you submit last, you are last in queue at your chosen price.

### Guaranteed buyback after the auction

After the auction, Merchant Guild buyback prices:

- `DRYLAND_FLAX`: 30 per unit (no fees)
- `EMBER_MUSHROOM`: 20 per unit (fee: 0.10 per unit traded)

### Submission flow

Choose bid price and quantity per product. Re-submit until round end. The last submission is executed.
