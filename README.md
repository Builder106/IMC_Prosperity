# IMC Prosperity Trading Assistant

A comprehensive AI-powered trading assistant for IMC Prosperity trading competition, providing insights and answers about trading data, strategies, and IMC Prosperity documentation.

## Overview

This project creates an intelligent assistant that helps with understanding, analyzing, and developing trading strategies for the IMC Prosperity trading competition. It combines Notion wiki data, trading logs, and code examples into a powerful RAG (Retrieval Augmented Generation) system that can:

- Answer questions about IMC Prosperity rules, mechanics, and concepts
- Analyze trading logs and provide insights
- Assist with developing and improving trading algorithms
- Visualize trading data and performance metrics

## Features

- **Streamlit Web Interface**: User-friendly interface for interacting with the assistant
- **RAG System**: Combines retrieval-based and generative AI to provide accurate answers
- **Knowledge Base**: Processes and indexes Notion wiki content, code examples, and trading data
- **Trading Log Analysis**: Summarizes and extracts insights from trading logs
- **Vector Database**: Efficient storage and retrieval of embedded documents

## Project Structure

- **`app.py`**: Main Streamlit application entry point
- **`IMC_visualizer_prereqs.py`**: Prerequisites for visualization functionality
- **`src/`**: Source code for various components
  - **`algorithms/`**: Trading algorithms for different competition rounds
  - **`rag/`**: RAG system implementation
    - **`build_rag_system.py`**: Core RAG system construction
    - **`process_raw_trading_data.py`**: Processing for trading data
  - **`utils/`**: Utility functions
    - **`notion_scraper/`**: Tools for scraping Notion wiki content
    - **`summarize_trading_logs.py`**: Trading log analysis tools
- **`data/`**: Data files
  - **`prosperity_wiki/`**: Processed Notion wiki content
  - **`trading_data/`**: Trading data from various rounds
- **`vectordb/`**: Vector databases for efficient retrieval
- **`backtests/`**: Trading backtest logs

## Getting Started

### Prerequisites

- Python 3.9+
- Required Python packages (install via `pip install -r requirements.txt`)
- Google API key for access to Google Generative AI models (stored in `.env` file)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/imc_prosperity.git
   cd imc_prosperity
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

### Running the Application

Start the Streamlit web interface:
```bash
streamlit run app.py
```

## Usage

1. Enter questions about IMC Prosperity in the text input field
2. View AI-generated answers with source information
3. Analyze trading logs by uploading them through the interface
4. Get insights and recommendations for improving trading strategies

## Working with Trading Logs

To summarize trading logs:
```bash
python src/utils/summarize_trading_logs.py
```
Follow the prompts to input your log file path and receive a detailed summary.

## License

[Add your license information here]

## Acknowledgments

- IMC Prosperity for the trading competition
- LangChain for the RAG framework
- Google Generative AI for the language models
- Streamlit for the web interface
