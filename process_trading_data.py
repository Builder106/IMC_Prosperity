import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
import argparse
import re

def process_trading_csv(file_path, output_dir):
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
    df = pd.read_csv(file_path, sep=';')
    
    # Clean up column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    documents = []
    
    # Group by day and product
    if 'product' in df.columns:
        for (day, product), group in df.groupby(['day', 'product']):
            # Create product-specific directory
            product_dir = Path(output_dir) / product.lower()
            os.makedirs(product_dir, exist_ok=True)
            
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
            
            # Extract round information from file path
            round_match = re.search(r'round_(\d+)', str(file_path), re.IGNORECASE)
            round_info = f"round_{round_match.group(1)}" if round_match else "unknown_round"
            
            # Create document with metadata
            document = {
                "metadata": {
                    "source": os.path.basename(file_path),
                    "day": int(day),
                    "product": product,
                    "type": "trading_data",
                    "file_type": "price" if "prices" in str(file_path).lower() else "trade",
                    "round": round_info,
                    "basket": get_basket_info(file_path)
                },
                "content": content
            }
            
            documents.append(document)
            
            # Save individual document
            file_base = os.path.basename(file_path).replace('.csv', '.json')
            doc_filename = product_dir / f"{product}_day_{day}_{file_base}"
            with open(doc_filename, 'w') as f:
                json.dump(document, f, indent=2)
                
    return documents

def get_basket_info(file_path):
    """Extract basket information from the file path if available"""
    path_str = str(file_path)
    
    # Match any basket pattern like "basket1", "picnicbasket2", etc.
    basket_match = re.search(r'([a-z]+basket\d+)', path_str, re.IGNORECASE)
    if basket_match:
        return basket_match.group(1).lower()
    
    return None

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

def detect_dir_structure(round_dir):
    """
    Detect the directory structure for a round folder.
    
    Args:
        round_dir: Path to round directory
    
    Returns:
        String indicating structure type ("simple", "basket", "unknown")
    """
    raw_data_dir = next((d for d in round_dir.glob("*raw_trading_data")), None)
    
    if not raw_data_dir or not raw_data_dir.exists():
        return "unknown"
    
    # Check for simple structure (prices/ and trades/ directly under raw_data_dir)
    if (raw_data_dir / "prices").exists() and (raw_data_dir / "trades").exists():
        return "simple"
    
    # Check for basket structure (baskets containing prices/ and trades/)
    baskets = list(raw_data_dir.glob("*basket*"))
    if baskets and all((b / "prices").exists() and (b / "trades").exists() for b in baskets if b.is_dir()):
        return "basket"
    
    # Check for other potential structures
    # For now, default to trying the simple structure if we can't determine
    return "unknown"

def process_round_data(round_name, trading_data_dir="trading_data"):
    """
    Process all CSV files for a specific round.
    
    Args:
        round_name: 'round_1', 'round_2', etc.
        trading_data_dir: Base directory containing trading data
    
    Returns:
        List of all processed documents
    """
    base_dir = Path(trading_data_dir) / round_name
    if not base_dir.exists():
        print(f"Warning: Directory for {round_name} not found at {base_dir}")
        return []
        
    raw_data_dir = base_dir / f"{round_name}_raw_trading_data"
    processed_dir = base_dir / f"{round_name}_processed_trading_data"
    
    if not raw_data_dir.exists():
        print(f"Warning: Raw data directory for {round_name} not found at {raw_data_dir}")
        return []
    
    print(f"Processing {round_name} data...")
    print(f"Looking for CSV files in {raw_data_dir}")
    
    all_documents = []
    
    # Determine directory structure
    structure = detect_dir_structure(base_dir)
    print(f"Detected directory structure: {structure}")
    
    # Simple structure (like round_1)
    if structure == "simple":
        for data_type in ["prices", "trades"]:
            type_dir = raw_data_dir / data_type
            if type_dir.exists():
                for csv_file in sorted(type_dir.glob("*.csv")):
                    try:
                        documents = process_trading_csv(csv_file, processed_dir)
                        all_documents.extend(documents)
                        print(f"Processed {len(documents)} documents from {csv_file}")
                    except Exception as e:
                        print(f"Error processing {csv_file}: {e}")
    
    # Basket structure (like round_2)
    elif structure == "basket":
        for basket_dir in raw_data_dir.glob("*basket*"):
            if basket_dir.is_dir():
                basket_name = basket_dir.name
                basket_processed_dir = processed_dir / basket_name
                
                print(f"Processing {basket_name}...")
                
                for data_type in ["prices", "trades"]:
                    type_dir = basket_dir / data_type
                    if type_dir.exists():
                        for csv_file in sorted(type_dir.glob("*.csv")):
                            try:
                                documents = process_trading_csv(csv_file, basket_processed_dir)
                                all_documents.extend(documents)
                                print(f"Processed {len(documents)} documents from {csv_file}")
                            except Exception as e:
                                print(f"Error processing {csv_file}: {e}")
    
    # Fallback - try to find any CSV files with common patterns
    else:
        print(f"Unknown directory structure for {round_name}, trying to find CSV files...")
        
        # Look for CSV files in any prices or trades directories
        for csv_file in raw_data_dir.glob("**/prices/*.csv"):
            try:
                documents = process_trading_csv(csv_file, processed_dir)
                all_documents.extend(documents)
                print(f"Processed {len(documents)} documents from {csv_file}")
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                
        for csv_file in raw_data_dir.glob("**/trades/*.csv"):
            try:
                documents = process_trading_csv(csv_file, processed_dir)
                all_documents.extend(documents)
                print(f"Processed {len(documents)} documents from {csv_file}")
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
    
    # Save all documents for this round in one file for convenience
    if all_documents:
        os.makedirs(processed_dir, exist_ok=True)  # Ensure directory exists
        with open(processed_dir / f"all_{round_name}_trading_data.json", 'w') as f:
            json.dump(all_documents, f, indent=2)
    
    return all_documents

def discover_rounds(trading_data_dir="trading_data"):
    """Discover all available round directories in the trading data directory"""
    try:
        base_dir = Path(trading_data_dir)
        if not base_dir.exists():
            print(f"Trading data directory not found: {base_dir}")
            return []
        
        # Find all directories that match the pattern round_X
        round_dirs = [d for d in base_dir.iterdir() 
                     if d.is_dir() and re.match(r'round_\d+', d.name, re.IGNORECASE)]
        
        # Sort by round number
        round_dirs.sort(key=lambda d: int(re.search(r'round_(\d+)', d.name).group(1)))
        
        return [d.name for d in round_dirs]
    except Exception as e:
        print(f"Error discovering rounds: {e}")
        return []

def main():
    """Main execution function with command-line argument parsing"""
    parser = argparse.ArgumentParser(description='Process trading data from CSV files')
    parser.add_argument('--rounds', type=str, default='all', 
                        help='Which rounds to process: comma-separated list like "round_1,round_3" or "all"')
    parser.add_argument('--data_dir', type=str, default='trading_data',
                        help='Base directory containing trading data')
    args = parser.parse_args()
    
    # Discover available rounds
    available_rounds = discover_rounds(args.data_dir)
    print(f"Discovered rounds: {available_rounds}")
    
    all_documents = []
    
    # Process specific rounds or all available rounds
    if args.rounds.lower() == 'all':
        rounds_to_process = available_rounds
    else:
        rounds_to_process = [r.strip() for r in args.rounds.split(',')]
        
    print(f"Will process the following rounds: {rounds_to_process}")
    
    for round_name in rounds_to_process:
        if round_name in available_rounds:
            docs = process_round_data(round_name, args.data_dir)
            all_documents.extend(docs)
        else:
            print(f"Warning: {round_name} not found in available rounds, skipping.")
    
    print(f"Processed a total of {len(all_documents)} documents")

if __name__ == "__main__":
    main()