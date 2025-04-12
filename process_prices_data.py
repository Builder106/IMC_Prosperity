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

        # MARKET MICROSTRUCTURE - SPREAD METRICS
        # Relative spread (as percentage of mid-price)
        df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
        df['relative_spread'] = df['spread'] / df['mid_price'] * 100  # as percentage
        metrics["avg_relative_spread"] = df['relative_spread'].mean()
        
        # Effective spread (if we have trade price data)
        if 'price' in df.columns:
            df['effective_spread'] = 2 * abs(df['price'] - df['mid_price'])
            metrics["avg_effective_spread"] = df['effective_spread'].mean()
            
            # Implementation shortfall (if we have sufficient trade data)
            if 'quantity' in df.columns:
                buy_trades = df[df['quantity'] > 0]
                sell_trades = df[df['quantity'] < 0]
                
                if not buy_trades.empty:
                    metrics["buy_implementation_shortfall"] = ((buy_trades['price'] - buy_trades['mid_price']) * buy_trades['quantity']).sum() / buy_trades['quantity'].sum()
                
                if not sell_trades.empty:
                    metrics["sell_implementation_shortfall"] = ((sell_trades['mid_price'] - sell_trades['price']) * abs(sell_trades['quantity'])).sum() / abs(sell_trades['quantity']).sum()
    
    # Order book depth metrics
    bid_cols = [col for col in df.columns if col.startswith('bid_volume_')]
    ask_cols = [col for col in df.columns if col.startswith('ask_volume_')]
    bid_price_cols = [col for col in df.columns if col.startswith('bid_price_')]
    ask_price_cols = [col for col in df.columns if col.startswith('ask_price_')]
    
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
            
        # MARKET MICROSTRUCTURE - ORDER BOOK METRICS
        # Order book slope (price change per unit volume)
        if len(bid_price_cols) >= 2 and len(ask_price_cols) >= 2:
            # Calculate average price changes between levels
            df['bid_slope'] = abs((df['bid_price_1'] - df['bid_price_2']) / 
                              (df['bid_volume_1'] + df['bid_volume_2']))
            df['ask_slope'] = abs((df['ask_price_2'] - df['ask_price_1']) / 
                              (df['ask_volume_1'] + df['ask_volume_2']))
            
            metrics["avg_bid_slope"] = df['bid_slope'].mean()
            metrics["avg_ask_slope"] = df['ask_slope'].mean()
            metrics["book_slope"] = (metrics["avg_bid_slope"] + metrics["avg_ask_slope"]) / 2
            
        # Price impact estimation
        if 'mid_price' in df.columns:
            # Standard volume to measure price impact
            standard_volume = 10
            
            # Simple price impact estimation (how much the price might move for standard_volume units)
            if 'bid_price_1' in df.columns and 'bid_volume_1' in df.columns:
                # If standard_volume exceeds first level, estimate impact
                df['bid_impact'] = df.apply(lambda row: 
                    abs(row['bid_price_1'] - row['bid_price_2']) if row['bid_volume_1'] < standard_volume and 'bid_price_2' in df.columns
                    else 0, axis=1)
                metrics["avg_bid_impact"] = df['bid_impact'].mean()
            
            if 'ask_price_1' in df.columns and 'ask_volume_1' in df.columns:
                df['ask_impact'] = df.apply(lambda row: 
                    abs(row['ask_price_2'] - row['ask_price_1']) if row['ask_volume_1'] < standard_volume and 'ask_price_2' in df.columns 
                    else 0, axis=1)
                metrics["avg_ask_impact"] = df['ask_impact'].mean()
            
            # Kyle's Lambda estimation (price impact coefficient)
            # Requires price changes and volume data
            if 'price' in df.columns and 'quantity' in df.columns and len(df) > 10:
                # Sort by timestamp if available
                df_sorted = df.sort_values('timestamp') if 'timestamp' in df.columns else df
                
                # Calculate price changes
                df_sorted['price_change'] = df_sorted['price'].diff()
                df_sorted['signed_volume'] = df_sorted['quantity']  # Could improve with trade sign detection
                
                # Only use rows with valid data
                valid_rows = df_sorted['price_change'].notna() & df_sorted['signed_volume'].notna() & (df_sorted['signed_volume'] != 0)
                
                if valid_rows.sum() > 5:  # Need enough data points
                    from scipy import stats
                    # Kyle's Lambda is the regression coefficient
                    slope, _, _, _, _ = stats.linregress(
                        df_sorted.loc[valid_rows, 'signed_volume'], 
                        df_sorted.loc[valid_rows, 'price_change']
                    )
                    metrics["kyles_lambda"] = abs(slope)
                    
                    # Amihud's illiquidity measure
                    df_sorted['abs_return'] = abs(df_sorted['price_change'] / df_sorted['price'].shift(1))
                    df_sorted['amihud'] = df_sorted['abs_return'] / abs(df_sorted['signed_volume'])
                    metrics["amihud_illiquidity"] = df_sorted.loc[valid_rows, 'amihud'].mean()
        
        # Order flow toxicity (VPIN estimation)
        if 'quantity' in df.columns and len(df) > 20:
            # Time-based buckets for volume
            df_sorted = df.sort_values('timestamp') if 'timestamp' in df.columns else df
            
            # Create buy/sell volume imbalance
            df_sorted['signed_volume'] = df_sorted['quantity']
            bucket_size = max(1, len(df_sorted) // 10)  # At least 10 buckets
            
            buy_volumes = []
            sell_volumes = []
            
            for i in range(0, len(df_sorted), bucket_size):
                bucket = df_sorted.iloc[i:i+bucket_size]
                buy_vol = bucket.loc[bucket['signed_volume'] > 0, 'signed_volume'].sum()
                sell_vol = abs(bucket.loc[bucket['signed_volume'] < 0, 'signed_volume'].sum())
                
                buy_volumes.append(buy_vol)
                sell_volumes.append(sell_vol)
            
            if len(buy_volumes) >= 5:  # Need enough buckets
                # Calculate VPIN (Volume-Synchronized Probability of Informed Trading)
                imbalances = [abs(b - s) / (b + s) if b + s > 0 else 0 for b, s in zip(buy_volumes, sell_volumes)]
                metrics["vpin"] = sum(imbalances) / len(imbalances)
    
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

    # STATISTICAL METRICS - NEW ADDITION
    if 'mid_price' in df.columns and len(df) >= 4:  # Need sufficient data points
        from scipy import stats
        import numpy as np
        
        # Higher-order moments of price distribution
        metrics["price_skewness"] = stats.skew(df['mid_price'])
        metrics["price_kurtosis"] = stats.kurtosis(df['mid_price'])  # Excess kurtosis (normal = 0)
        
        # Normality tests
        shapiro_stat, shapiro_pval = stats.shapiro(df['mid_price']) if len(df) <= 5000 else (np.nan, np.nan)
        jb_stat, jb_pval = stats.jarque_bera(df['mid_price'])
        metrics["shapiro_normality_pvalue"] = shapiro_pval
        metrics["jarque_bera_normality_pvalue"] = jb_pval
        
        # Price distribution percentiles
        for pct in [1, 5, 25, 50, 75, 95, 99]:
            metrics[f"price_percentile_{pct}"] = np.percentile(df['mid_price'], pct)
        
        # Interquartile range
        q1 = np.percentile(df['mid_price'], 25)
        q3 = np.percentile(df['mid_price'], 75)
        metrics["price_iqr"] = q3 - q1
        
        # Outlier detection
        iqr = metrics["price_iqr"]
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df['mid_price'] < lower_bound) | (df['mid_price'] > upper_bound)]
        metrics["price_outlier_count"] = len(outliers)
        metrics["price_outlier_percentage"] = (len(outliers) / len(df)) * 100 if len(df) > 0 else 0
        
        # Returns-based metrics if we have timestamps
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            df_sorted['log_returns'] = np.log(df_sorted['mid_price'] / df_sorted['mid_price'].shift(1))
            valid_returns = df_sorted['log_returns'].dropna()
            
            if len(valid_returns) >= 4:
                metrics["returns_mean"] = valid_returns.mean()
                metrics["returns_volatility"] = valid_returns.std()
                metrics["returns_skewness"] = stats.skew(valid_returns)
                metrics["returns_kurtosis"] = stats.kurtosis(valid_returns)
                
                # Ljung-Box test for autocorrelation (p < 0.05 suggests autocorrelation)
                lb_stat, lb_pval = stats.acorr_ljungbox(valid_returns, lags=[5], return_df=False)
                metrics["returns_autocorr_pvalue"] = lb_pval[0]
                
                # Autocorrelation at different lags
                for lag in [1, 5, 10]:
                    if len(valid_returns) > lag:
                        metrics[f"returns_autocorr_lag_{lag}"] = valid_returns.autocorr(lag=lag)
        
        # Hurst exponent (indicator of mean-reversion vs trend)
        if len(df) > 100:
            try:
                metrics["hurst_exponent"] = calculate_hurst_exponent(df['mid_price'])
            except:
                metrics["hurst_exponent"] = np.nan
                
        # Correlation metrics if multiple price columns exist
        price_cols = [col for col in df.columns if 'price' in col]
        if len(price_cols) > 1:
            # Create correlation matrix
            corr_matrix = df[price_cols].corr()
            # Average correlation between mid_price and other prices
            if 'mid_price' in corr_matrix.columns:
                other_cols = [col for col in price_cols if col != 'mid_price']
                if other_cols:
                    metrics["avg_price_correlation"] = corr_matrix.loc['mid_price', other_cols].mean()
    
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

def calculate_hurst_exponent(time_series, max_lag=20):
    """
    Calculate the Hurst exponent of a time series.
    
    Args:
        time_series: Array-like price or return series
        max_lag: Maximum lag to consider
    
    Returns:
        Hurst exponent (H):
        H < 0.5: mean-reverting
        H = 0.5: geometric random walk
        H > 0.5: trending
    """
    import numpy as np
    
    # Ensure the time series is a numpy array
    ts = np.array(time_series)
    
    # Remove NaN values
    ts = ts[~np.isnan(ts)]
    
    if len(ts) < max_lag + 10:  # Need enough data
        return np.nan
    
    # Calculate range of lags
    lags = range(2, max_lag)
    
    # Calculate variance of the log return
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    
    # Calculate the Hurst exponent as the slope
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    return poly[0]  # Hurst exponent is the slope

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
            break