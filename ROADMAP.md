# IMC Prosperity Roadmap

Trading algorithmic strategy and simulation roadmap.

## v1.1 — RAG Evaluation & Strategy Decoupling

- **Strategy Decoupling**: Modular strategy architecture per [`docs/specs/rag-eval-and-decoupling-plan.md`](docs/specs/rag-eval-and-decoupling-plan.md).
- **Backtesting Harness**: Deterministic historical market data replay with order fill modeling.

## v1.2 — Multi-Asset Arbitrage & Machine Learning Signals

- **Volatility Surface Modeling**: Real-time implied volatility fitting and ETF basket arbitrage strategies.
- **Risk Ceilings & Position Limits**: Hard automated safeguards on inventory drawdown.

## Out of Scope

- Real-money exchange execution (competition / simulation only)
- Non-deterministic random seed backtests

---
For technical specifications, see [`docs/specs/`](docs/specs/).
