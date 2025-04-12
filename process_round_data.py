import pandas as pd
import json
import os
from pathlib import Path
import numpy as np

def process_round_csv(file_path, output_dir="processed_data"):
    """
    Process the round.csv file into documents.
    
    Args:
        file_path: Path to the round.csv file
        output_dir: Directory to save processed documents
    
    Returns:
        List of documents with trading data and metadata
    """
    print(f"Processing {file_path}...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, delimiter=';')
        
        # Clean up column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        documents = []
        
        # Determine how to group the data based on the columns present
        if 'product' in df.columns and 'day' in df.columns:
            # Standard market data format
            for (day, product), group in df.groupby(['day', 'product']):
                process_group(day, product, group, file_path, output_dir, documents)
                
        elif 'symbol' in df.columns:
            # Trade data format with symbol instead of product
            if 'timestamp' in df.columns and not 'day' in df.columns:
                # Try to extract day from timestamp if possible
                # This is a simplified assumption - adjust based on actual data
                for symbol, group in df.groupby('symbol'):
                    day = 0  # Default day if can't determine
                    process_group(day, symbol, group, file_path, output_dir, documents)
            else:
                for symbol, group in df.groupby('symbol'):
                    day = group['day'].iloc[0] if 'day' in group.columns else 0
                    process_group(day, symbol, group, file_path, output_dir, documents)
        else:
            print(f"Warning: Unrecognized CSV format for {file_path}, processing as single document")
            process_group(0, "unknown", df, file_path, output_dir, documents)
            
        # Save all documents in one file for convenience
        with open(Path(output_dir) / "all_round_data.json", 'w') as f:
            json.dump(documents, f, indent=2)
            
        return documents
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def process_group(day, product, group, file_path, output_dir, documents):
    """Process a group of data rows"""
    product_dir = Path(output_dir) / product.lower()
    os.makedirs(product_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = calculate_metrics(group)
    
    # Create document content
    content = f"Trading data for {product} on day {day}:\n"
    
    if 'mid_price' in group.columns:
        content += f"Average mid price: {group['mid_price'].mean():.2f}\n"
        content += f"Price range: {group['mid_price'].min():.2f} to {group['mid_price'].max():.2f}\n"
    
    if 'price' in group.columns:
        content += f"Average price: {group['price'].mean():.2f}\n"
        content += f"Price range: {group['price'].min():.2f} to {group['price'].max():.2f}\n"
    
    content += f"Data points: {len(group)}\n\n"
    
    # Add calculated metrics
    content += "Trading metrics:\n"
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
            content += f"{metric_name}: {metric_value:.4f}\n"
    
    # Create document with metadata
    document = {
        "metadata": {
            "source": os.path.basename(file_path),
            "day": int(day),
            "product": product,
            "type": "trading_data",
            "round": "round"
        },
        "content": content
    }
    
    documents.append(document)
    
    # Save individual document
    doc_filename = product_dir / f"{product}_day_{day}_data.json"
    with open(doc_filename, 'w') as f:
        json.dump(document, f, indent=2)

def calculate_metrics(df):
    """
    Calculate trading metrics for the data.
    
    Args:
        df: DataFrame containing data for a specific product/day
    
    Returns:
        Dict of calculated metrics
    """
    metrics = {}
    
    # Basic metrics
    if 'mid_price' in df.columns:
        metrics["volatility"] = df["mid_price"].std()
        metrics["price_momentum"] = df["mid_price"].diff().mean()
    
    if 'price' in df.columns:
        metrics["price_volatility"] = df["price"].std()
        metrics["price_mean"] = df["price"].mean()
    
    # Volume-related metrics
    if 'quantity' in df.columns:
        metrics["total_volume"] = df["quantity"].sum()
        metrics["avg_trade_size"] = df["quantity"].mean()
    
    # Calculate bid-ask spread if available
    if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
        df['spread'] = df['ask_price_1'] - df['bid_price_1']
        metrics["avg_spread"] = df['spread'].mean()
        metrics["max_spread"] = df['spread'].max()
    
    # Order book depth metrics
    bid_cols = [col for col in df.columns if col.startswith('bid_volume_')]
    ask_cols = [col for col in df.columns if col.startswith('ask_volume_')]
    
    if bid_cols and ask_cols:
        # Total visible liquidity
        df['total_bid_volume'] = df[bid_cols].sum(axis=1)
        df['total_ask_volume'] = df[ask_cols].sum(axis=1)
        metrics["avg_total_bid_volume"] = df['total_bid_volume'].mean()
        metrics["avg_total_ask_volume"] = df['total_ask_volume'].mean()
        
        # Order book imbalance
        df['book_imbalance'] = (df['total_bid_volume'] - df['total_ask_volume']) / (df['total_bid_volume'] + df['total_ask_volume'])
        metrics["avg_book_imbalance"] = df['book_imbalance'].mean()
        metrics["book_pressure"] = metrics["avg_book_imbalance"] * 100  # as percentage
        
        # Top of book concentration
        if 'bid_volume_1' in df.columns and 'ask_volume_1' in df.columns:
            df['bid_top_concentration'] = df['bid_volume_1'] / df['total_bid_volume']
            df['ask_top_concentration'] = df['ask_volume_1'] / df['total_ask_volume']
            metrics["avg_bid_top_concentration"] = df['bid_top_concentration'].mean()
            metrics["avg_ask_top_concentration"] = df['ask_top_concentration'].mean()
    
    # TIME-BASED METRICS - NEW ADDITION
    if 'timestamp' in df.columns:
        # Ensure data is sorted by timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Time range metrics
        metrics["time_span"] = df_sorted["timestamp"].max() - df_sorted["timestamp"].min()
        
        # Calculate average time between observations
        if len(df_sorted) > 1 and metrics["time_span"] > 0:
            metrics["avg_time_between_obs"] = metrics["time_span"] / (len(df_sorted) - 1)
            metrics["data_frequency"] = (len(df_sorted) - 1) / metrics["time_span"] if metrics["time_span"] > 0 else 0
        
        # Price velocity (rate of change over time)
        if 'mid_price' in df_sorted.columns:
            df_sorted['price_diff'] = df_sorted['mid_price'].diff()
            df_sorted['time_diff'] = df_sorted['timestamp'].diff()
            
            # Only calculate where we have valid differences
            valid_idx = (df_sorted['time_diff'] > 0) & df_sorted['price_diff'].notna()
            if valid_idx.any():
                df_sorted.loc[valid_idx, 'price_velocity'] = df_sorted.loc[valid_idx, 'price_diff'] / df_sorted.loc[valid_idx, 'time_diff']
                
                metrics["avg_price_velocity"] = df_sorted.loc[valid_idx, 'price_velocity'].mean()
                metrics["max_price_velocity"] = df_sorted.loc[valid_idx, 'price_velocity'].abs().max()
                
                # Price acceleration (second derivative)
                df_sorted['velocity_diff'] = df_sorted['price_velocity'].diff()
                valid_acc_idx = valid_idx & df_sorted['velocity_diff'].notna()
                if valid_acc_idx.any():
                    df_sorted.loc[valid_acc_idx, 'price_acceleration'] = df_sorted.loc[valid_acc_idx, 'velocity_diff'] / df_sorted.loc[valid_acc_idx, 'time_diff']
                    metrics["avg_price_acceleration"] = df_sorted.loc[valid_acc_idx, 'price_acceleration'].mean()
        
        # Moving averages if enough data points
        if 'mid_price' in df_sorted.columns and len(df_sorted) >= 5:
            # Short term moving average (5 periods)
            df_sorted['sma_5'] = df_sorted['mid_price'].rolling(window=5, min_periods=1).mean()
            # Medium term moving average (10 periods)
            if len(df_sorted) >= 10:
                df_sorted['sma_10'] = df_sorted['mid_price'].rolling(window=10, min_periods=1).mean()
                
                # Moving average crossover analysis
                if len(df_sorted) >= 10:
                    # Detect crossovers (1 when short crosses above long, -1 when short crosses below long)
                    df_sorted['ma_cross'] = ((df_sorted['sma_5'] > df_sorted['sma_10']) & 
                                           (df_sorted['sma_5'].shift(1) <= df_sorted['sma_10'].shift(1))).astype(int) - \
                                          ((df_sorted['sma_5'] < df_sorted['sma_10']) & 
                                           (df_sorted['sma_5'].shift(1) >= df_sorted['sma_10'].shift(1))).astype(int)
                    
                    metrics["ma_crossovers_up"] = (df_sorted['ma_cross'] == 1).sum()
                    metrics["ma_crossovers_down"] = (df_sorted['ma_cross'] == -1).sum()
        
        # Time-based volatility
        if 'mid_price' in df_sorted.columns:
            # Calculate rolling volatility over different windows
            if len(df_sorted) >= 5:
                df_sorted['rolling_vol_5'] = df_sorted['mid_price'].rolling(window=5, min_periods=3).std()
                metrics["avg_rolling_volatility_5"] = df_sorted['rolling_vol_5'].mean()
            
            if len(df_sorted) >= 10:
                df_sorted['rolling_vol_10'] = df_sorted['mid_price'].rolling(window=10, min_periods=5).std()
                metrics["avg_rolling_volatility_10"] = df_sorted['rolling_vol_10'].mean()
    
    # Profit and Loss metrics
    if 'profit_and_loss' in df.columns:
        # Basic P&L statistics
        metrics["total_pnl"] = df['profit_and_loss'].sum()
        metrics["avg_pnl"] = df['profit_and_loss'].mean()
        metrics["pnl_volatility"] = df['profit_and_loss'].std()
        
        # Min/Max P&L values
        metrics["max_pnl"] = df['profit_and_loss'].max()
        metrics["min_pnl"] = df['profit_and_loss'].min()
        
        # Sort by timestamp for cumulative calculations if available
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
        else:
            df_sorted = df
            
        # Cumulative P&L
        df_sorted['cum_pnl'] = df_sorted['profit_and_loss'].cumsum()
        metrics["final_cum_pnl"] = df_sorted['cum_pnl'].iloc[-1]
        
        # Calculate drawdown
        running_max = df_sorted['cum_pnl'].cummax()
        drawdown = running_max - df_sorted['cum_pnl']
        metrics["max_drawdown"] = drawdown.max()
        
        # Calculate simple Sharpe-like ratio if we have enough data points
        if len(df) > 2 and df['profit_and_loss'].std() > 0:
            metrics["pnl_sharpe"] = df['profit_and_loss'].mean() / df['profit_and_loss'].std()
        
        # Count of profitable vs unprofitable data points
        profitable_points = (df['profit_and_loss'] > 0).sum()
        metrics["profitable_points"] = profitable_points
        metrics["profit_ratio"] = profitable_points / len(df) if len(df) > 0 else 0
    
    return metrics

if __name__ == "__main__":
    print("Welcome to the Round Data Processor!")
    
    # Get file path from user
    while True:
        file_path = input("Please enter the path to your CSV file (or 'q' to quit): ").strip()
        
        if file_path.lower() == 'q':
            print("Goodbye!")
            break
            
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist. Please try again.")
            continue
            
        if not file_path.endswith('.csv'):
            print("Warning: File does not have .csv extension. Are you sure this is a CSV file?")
            proceed = input("Do you want to continue? (y/n): ").strip().lower()
            if proceed != 'y':
                continue
        
        # Get output directory
        output_dir = input("Enter output directory path (press Enter for default 'processed_data'): ").strip()
        if not output_dir:
            output_dir = "processed_data"
        
        # Process the CSV file
        try:
            process_round_csv(file_path, output_dir)
            print(f"Processing complete! Output saved to {output_dir}")
        except Exception as e:
            print(f"Error processing file: {e}")
        
        # Ask if user wants to process another file
        another = input("\nWould you like to process another file? (y/n): ").strip().lower()
        if another != 'y':
            print("Goodbye!")
            # break