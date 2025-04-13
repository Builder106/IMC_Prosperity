import re
import json
import os
from collections import defaultdict

def summarize_trading_logs(log_file_path):
    """Summarize IMC Prosperity trading logs to make them digestible for an LLM."""
    
    # Initialize data structures
    position_history = defaultdict(list)
    order_patterns = defaultdict(int)
    price_history_progress = {}
    special_order_patterns = defaultdict(int)
    
    # Read log file
    with open(log_file_path, 'r') as file:
        log_content = file.read()
    
    # Extract timestamps and log blocks
    log_blocks = re.findall(r'{"sandboxLog": "", "lambdaLog": "(.*?)", "timestamp": (\d+)}', 
                           log_content, re.DOTALL)
    
    # Process each log block
    for log_text, timestamp in log_blocks:
        timestamp = int(timestamp)
        
        # Extract positions for each product
        positions = re.findall(r'Processing (\w+) - Current position: (-?\d+)', log_text)
        for product, position in positions:
            position_history[product].append((timestamp, int(position)))
        
        # Track price history collection progress
        history_progress = re.findall(r'(\w+): Collecting price history \((\d+)/(\d+)\)', log_text)
        for product, current, total in history_progress:
            price_history_progress[product] = (int(current), int(total))
        
        # Count order pattern types (regular vs special handling)
        if "Placed 2 orders" in log_text:
            for product in re.findall(r'(\w+): Placed 2 orders', log_text):
                order_patterns[f"{product}_standard"] += 1
                
        if "Placed 3 orders" in log_text:
            for product in re.findall(r'(\w+): Placed 3 orders', log_text):
                special_order_patterns[f"{product}_position_management"] += 1
    
    # Create summary output
    summary = {
        "position_trends": {product: [history[0], history[-1]] for product, history in position_history.items()},
        "price_history_collection": price_history_progress,
        "order_patterns": dict(order_patterns),
        "special_order_handling": dict(special_order_patterns),
        "representative_examples": extract_representative_examples(log_content)
    }
    
    # Format output
    return format_summary_for_llm(summary)

def extract_representative_examples(log_content):
    """Extract representative examples of important log patterns."""
    examples = {}
    
    # Find a good example of standard orders
    standard_match = re.search(r'--- Orders Generated: \[(.*?)\]', log_content, re.DOTALL)
    if standard_match:
        examples["standard_orders"] = standard_match.group(0)[:500]  # Limit length
    
    # Find an example of special orders (like SQUID_INK with 3 orders)
    special_match = re.search(r"SQUID_INK.*?(\[\('SQUID_INK'.*?\)\])", log_content, re.DOTALL)
    if special_match:
        examples["special_orders"] = special_match.group(0)[:500]  # Limit length
    
    return examples

def format_summary_for_llm(summary):
    """Format the summary in a way that's easy for an LLM to understand."""
    output = "# IMC Prosperity Trading Bot Log Summary\n\n"
    
    # Position trends section
    output += "## Position Trends\n"
    for product, positions in summary["position_trends"].items():
        start_ts, start_pos = positions[0]
        end_ts, end_pos = positions[1]
        change = end_pos - start_pos
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
        output += f"{product}: {start_pos} → {end_pos} ({direction} {abs(change)})\n"
    
    # Price history collection section
    output += "\n## Price History Collection\n"
    for product, (current, total) in summary["price_history_collection"].items():
        output += f"{product}: {current}/{total} data points collected\n"
    
    # Order patterns section
    output += "\n## Order Patterns\n"
    for pattern, count in summary["order_patterns"].items():
        product = pattern.split('_')[0]
        output += f"{product}: Standard 2-order pattern executed {count} times\n"
    
    for pattern, count in summary["special_order_patterns"].items():
        product = pattern.split('_')[0]
        output += f"{product}: Special order pattern executed {count} times\n"
    
    # Example section (truncated)
    output += "\n## Representative Examples\n"
    for example_type, text in summary["examples"].items():
        output += f"### {example_type}\n```\n{text}\n```\n"
    
    return output

# Interactive script execution
if __name__ == "__main__":
    print("=== IMC Prosperity Trading Log Summarizer ===")
    
    # Ask for input file path
    while True:
        log_file_path = input("\nEnter the path to the trading log file (or 'q' to quit): ").strip()
        
        if log_file_path.lower() == 'q':
            print("Exiting program.")
            break
            
        if not os.path.exists(log_file_path):
            print(f"Error: File not found at '{log_file_path}'. Please try again.")
            continue
        
        # Ask for output file path
        output_file = input("\nEnter the path for the summary output file (or press Enter for default 'log_summary_for_llm.md'): ").strip()
        if not output_file:
            output_file = "log_summary_for_llm.md"
        
        # Process the log file
        try:
            print(f"\nProcessing log file: {log_file_path}...")
            summary_content = summarize_trading_logs(log_file_path)
            
            # Save summary to file
            with open(output_file, "w") as f:
                f.write(summary_content)
                
            print(f"Summary successfully saved to: {output_file}")
            
            # Ask if user wants to display summary in terminal
            show_summary = input("\nDisplay summary in terminal? (y/n): ").strip().lower()
            if show_summary == 'y':
                print("\n" + summary_content)
                
        except Exception as e:
            print(f"Error processing log file: {e}")
        
        # Ask if user wants to process another file
        another = input("\nProcess another log file? (y/n): ").strip().lower()
        if another != 'y':
            print("Exiting program.")
            break