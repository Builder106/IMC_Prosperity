import pandas as pd
import os
import json
import numpy as np
import re
from pathlib import Path
import argparse # For command-line arguments later

# --- Import metric calculation functions ---
# Assuming the refactored functions are in process_prices_data.py
# Adjust the import path if necessary
try:
    from process_round_trading_data import (
        calculate_metrics,
        calculate_hurst_exponent # Make sure this is also importable or defined here
        # If calculate_metrics calls sub-functions, they don't need explicit import
        # unless you call them directly here.
    )
    print("Successfully imported metric functions from process_round_trading_data.py")
except ImportError:
    print("Error: Could not import metric functions.")
    print("Make sure 'process_round_trading_data.py' is in the Python path and contains the required functions.")
    exit()

def main(prices_filepath, trades_filepath, output_filepath):
    """
    Loads, merges, analyzes price and trade data, and saves results.
    """
    print(f"Starting analysis for:")
    print(f"  Prices: {prices_filepath}")
    print(f"  Trades: {trades_filepath}")

    # --- Check if files exist ---
    if not os.path.exists(prices_filepath):
        print(f"Error: Prices file not found at {prices_filepath}")
        return
    if not os.path.exists(trades_filepath):
        print(f"Error: Trades file not found at {trades_filepath}")
        return

    # --- Load DataFrames ---
    try:
        print("Loading prices...")
        prices_df = pd.read_csv(prices_filepath, sep=';', comment='/')
        prices_df.columns = prices_df.columns.str.strip() # Clean column names
        print(f"  Loaded {len(prices_df)} rows. Columns: {prices_df.columns.tolist()}")

        print("Loading trades...")
        trades_df = pd.read_csv(trades_filepath, sep=';')
        trades_df.columns = trades_df.columns.str.strip() # Clean column names
        print(f"  Loaded {len(trades_df)} rows. Columns: {trades_df.columns.tolist()}")

    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        return

    # --- Prepare for Merging ---
    # Rename 'product' in prices_df to 'symbol'
    if 'product' in prices_df.columns:
        print("Renaming 'product' to 'symbol' in prices DataFrame.")
        prices_df = prices_df.rename(columns={'product': 'symbol'})
    elif 'symbol' not in prices_df.columns:
         print("Error: Prices DataFrame missing 'product' or 'symbol' column.")
         return

    if 'symbol' not in trades_df.columns:
         print("Error: Trades DataFrame missing 'symbol' column.")
         return

    # Sort by timestamp
    print("Sorting DataFrames by timestamp...")
    prices_df = prices_df.sort_values(by='timestamp')
    trades_df = trades_df.sort_values(by='timestamp')

    # --- Merge DataFrames ---
    print("Merging trades with price data using merge_asof...")
    merged_df = pd.merge_asof(
        trades_df.copy(), # Use copy to avoid modifying original trades_df
        prices_df,
        on='timestamp',
        by='symbol',
        direction='backward'
    )
    print(f"Merged DataFrame created with {len(merged_df)} rows.")
    print(f"Merged columns: {merged_df.columns.tolist()}")

    # --- Calculate Metrics per Product ---
    all_documents = []
    print("Calculating metrics for each product...")

    # Determine the day - assuming it's consistent in the prices file
    day = prices_df['day'].iloc[0] if 'day' in prices_df.columns else 0 # Default to 0 if no day column

    for symbol, group in merged_df.groupby('symbol'):
        print(f"  Processing symbol: {symbol}")
        if group.empty:
            print(f"    Skipping {symbol} - no data after merge.")
            continue

        try:
            # Calculate metrics using the imported function
            metrics = calculate_metrics(group.copy()) # Pass a copy to avoid SettingWithCopyWarning in metric functions

            # Create document content string
            content = f"Merged Trading Analysis for {symbol} on day {day}:\n"
            content += f"Number of trades: {len(group)}\n"
            if 'price' in group.columns:
                 content += f"Average trade price: {group['price'].mean():.2f}\n"
            if 'mid_price' in group.columns:
                 content += f"Average mid price at trade time: {group['mid_price'].mean():.2f}\n"
            content += "\nCalculated Metrics:\n"
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                    content += f"{metric_name}: {metric_value:.4f}\n"

            # Create document dictionary
            document = {
                "metadata": {
                    "prices_source": os.path.basename(prices_filepath),
                    "trades_source": os.path.basename(trades_filepath),
                    "day": int(day),
                    "product": symbol, # Use 'product' for consistency in metadata
                    "type": "merged_trading_analysis"
                },
                "content": content
            }
            all_documents.append(document)
            print(f"    Finished processing {symbol}.")

        except Exception as e:
            print(f"    Error calculating metrics for {symbol}: {e}")
            # Optionally add traceback: import traceback; traceback.print_exc()

    # --- Save Results ---
    print(f"\nSaving {len(all_documents)} documents to {output_filepath}")
    try:
        # Ensure output directory exists
        output_dir = Path(output_filepath).parent
        os.makedirs(output_dir, exist_ok=True)

        with open(output_filepath, 'w') as f:
            json.dump(all_documents, f, indent=2)
        print("Analysis complete and saved.")
    except Exception as e:
        print(f"Error saving output JSON: {e}")


if __name__ == "__main__":
    print("Welcome to the Merged Round Data Analyzer!")

    while True:
        # --- Get Input File Paths ---
        prices_file_input = input("Enter the path to the prices CSV file (or 'q' to quit): ").strip()
        if prices_file_input.lower() == 'q':
            print("Goodbye!")
            break

        trades_file_input = input("Enter the path to the trades CSV file (or 'q' to quit): ").strip()
        if trades_file_input.lower() == 'q':
            print("Goodbye!")
            break

        # --- Validate Input Files ---
        if not os.path.exists(prices_file_input):
            print(f"Error: Prices file not found at '{prices_file_input}'. Please try again.")
            continue
        if not prices_file_input.endswith('.csv'):
             print("Warning: Prices file does not end with .csv. Proceeding anyway.")

        if not os.path.exists(trades_file_input):
            print(f"Error: Trades file not found at '{trades_file_input}'. Please try again.")
            continue
        if not trades_file_input.endswith('.csv'):
             print("Warning: Trades file does not end with .csv. Proceeding anyway.")

        # --- Get Output File Path ---
        output_file_input = input("Enter the desired path for the output JSON file (e.g., processed_data/merged_analysis.json): ").strip()
        if not output_file_input:
            print("Error: Output file path cannot be empty.")
            continue
        if not output_file_input.endswith('.json'):
            output_file_input += '.json'
            print(f"Appending '.json' to output file path: {output_file_input}")

        # --- Run Main Analysis Function ---
        try:
            main(prices_file_input, trades_file_input, output_file_input)
        except Exception as e:
            print(f"\nAn unexpected error occurred during analysis: {e}")
            # Optionally add traceback: import traceback; traceback.print_exc()

        # --- Ask to process another pair ---
        another = input("\nAnalyze another pair of files? (y/n): ").strip().lower()
        if another != 'y':
            print("Goodbye!")
            break