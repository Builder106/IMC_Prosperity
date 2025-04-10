import pandas as pd
import json
import os
from pathlib import Path
import numpy as np

def process_trading_csv(file_path, output_dir="processed_trading_data"):
    """
    Process a trading CSV file into documents suitable for RAG.
    Each document represents aggregated data for a day and product.
    
    Args:
        file_path: Path to the CSV file
        output_dir: Directory to save processed documents
    
    Returns:
        List of documents with trading data and metadata
    """
    print(f"Processing {file_path}...")
    
    # Read the CSV using semicolon delimiter
    df = pd.read_csv(file_path, sep=';', comment='//')
    
    # Clean up column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    documents = []
    
    # Group by day and product
    if 'product' in df.columns:
        for (day, product), group in df.groupby(['day', 'product']):
            # Calculate trading metrics
            metrics = calculate_trading_metrics(group)
            
            # Create document content
            content = f"Trading data for {product} on day {day}:\n"
            
            if 'mid_price' in group.columns:
                content += f"Average mid price: {group['mid_price'].mean():.2f}\n"
                content += f"Price range: {group['mid_price'].min():.2f} to {group['mid_price'].max():.2f}\n"
            
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
                    "file_type": "price" if "prices" in file_path.lower() else "trade"
                },
                "content": content
            }
            
            documents.append(document)
            
            # Save individual document
            doc_filename = f"{output_dir}/{product}_day_{day}_{os.path.basename(file_path).replace('.csv', '.json')}"
            with open(doc_filename, 'w') as f:
                json.dump(document, f, indent=2)
                
    return documents

def calculate_trading_metrics(df):
    """
    Calculate additional trading metrics for a product.
    
    Args:
        df: DataFrame containing trading data for a specific product and day
    
    Returns:
        Dict of calculated metrics
    """
    metrics = {}
    
    # Basic metrics
    if 'mid_price' in df.columns:
        metrics["volatility"] = df["mid_price"].std()
        metrics["price_momentum"] = df["mid_price"].diff().mean()
    
    # Volume-weighted metrics
    bid_cols = [col for col in df.columns if 'bid_volume' in col]
    ask_cols = [col for col in df.columns if 'ask_volume' in col]
    
    if bid_cols and ask_cols and 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
        try:
            metrics["volume_weighted_price"] = (df["bid_volume_1"] * df["bid_price_1"] + 
                                          df["ask_volume_1"] * df["ask_price_1"]) / \
                                         (df["bid_volume_1"] + df["ask_volume_1"])
            metrics["volume_weighted_price"] = metrics["volume_weighted_price"].mean()
        except:
            metrics["volume_weighted_price"] = np.nan
    
    # Trading activity
    metrics["trading_activity"] = len(df)
    
    # Calculate bid-ask spread if available
    if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
        df['spread'] = df['ask_price_1'] - df['bid_price_1']
        metrics["avg_spread"] = df['spread'].mean()
        metrics["max_spread"] = df['spread'].max()
        metrics["min_spread"] = df['spread'].min()
    
    # Calculate market depth
    depth_columns = sum(1 for col in df.columns if 'bid_volume' in col or 'ask_volume' in col)
    if depth_columns > 0:
        metrics["market_depth"] = depth_columns // 2  # Divide by 2 to get depth on each side
    
    return metrics

def process_all_csv_files(data_dir="round_1_island_data", output_dir="processed_trading_data"):
    """
    Process all CSV files in a directory.
    
    Args:
        data_dir: Directory containing CSV files
        output_dir: Directory to save processed documents
    
    Returns:
        List of all processed documents
    """
    data_path = Path(data_dir)
    all_documents = []
    
    # Process price files first, then trade files
    for file_pattern in ["prices_*.csv", "trades_*.csv"]:
        for csv_file in sorted(data_path.glob(file_pattern)):
            try:
                documents = process_trading_csv(csv_file, output_dir)
                all_documents.extend(documents)
                print(f"Processed {len(documents)} documents from {csv_file}")
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
    
    # Save all documents in one file for convenience
    with open(f"{output_dir}/all_trading_data.json", 'w') as f:
        json.dump(all_documents, f, indent=2)
    
    return all_documents

if __name__ == "__main__":
    # Process all CSV files in the round_1_island_data directory
    documents = process_all_csv_files()
    print(f"Processed a total of {len(documents)} documents")
