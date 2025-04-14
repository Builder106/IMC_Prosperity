import json
import csv
import sys
import re

def extract_trade_history_to_csv(log_file_path, csv_file_path):
    """
    Extracts trade history JSON from a log file and converts it to a CSV file.

    Args:
        log_file_path (str): The path to the input log file.
        csv_file_path (str): The path to the output CSV file.
    """
    trade_history_marker = "Trade History:"
    json_buffer = []
    in_trade_history_block = False
    found_start_bracket = False
    bracket_level = 0

    print(f"Reading log file: {log_file_path}")
    try:
        with open(log_file_path, 'r', encoding='utf-8') as log_file:
            for line in log_file:
                stripped_line = line.strip()

                # Check if we found the marker line
                if not in_trade_history_block and trade_history_marker in line:
                    in_trade_history_block = True
                    print("Found 'Trade History:' marker.")
                    continue # Skip the marker line itself, start processing next line

                # If we are inside the block, start accumulating lines
                if in_trade_history_block:
                    # Find the start of the JSON array '['
                    if not found_start_bracket:
                        start_bracket_index = stripped_line.find('[')
                        if start_bracket_index != -1:
                            found_start_bracket = True
                            print("Found start bracket '['.")
                            # Take only the part from '[' onwards
                            stripped_line = stripped_line[start_bracket_index:]
                        else:
                            # Skip lines before the actual JSON array starts
                            continue

                    # If we have found the start, accumulate the line content
                    if found_start_bracket:
                        json_buffer.append(stripped_line)
                        # Track nested brackets to find the correct closing ']'
                        bracket_level += stripped_line.count('[')
                        bracket_level -= stripped_line.count(']')

                        # Check if we reached the end of the main array
                        if bracket_level == 0 and stripped_line.endswith(']'):
                            print("Found end bracket ']' for the main array.")
                            break # Stop reading lines once the JSON block is complete

        if not json_buffer:
            print(f"Error: '{trade_history_marker}' not found or JSON block could not be identified in {log_file_path}", file=sys.stderr)
            return

        # Join the buffer lines into a single string
        json_string = "".join(json_buffer)

        # Clean up potential extra characters before/after JSON if needed (though the bracket counting should handle it)
        # Basic check: ensure it starts with '[' and ends with ']'
        if not (json_string.startswith('[') and json_string.endswith(']')):
             print(f"Warning: Extracted text might not be a valid JSON array. Attempting parse anyway.", file=sys.stderr)
             # Attempt to find the main JSON array using regex if simple extraction failed
             match = re.search(r'(\[.*\])', json_string, re.DOTALL)
             if match:
                 json_string = match.group(1)
                 print("Used regex to refine JSON block.")
             else:
                 print(f"Error: Could not refine JSON block from extracted text:\n{json_string}", file=sys.stderr)
                 return
        
        # Clean up the JSON string to handle common formatting issues
        print("Cleaning JSON string to fix potential formatting issues...")
        # Remove trailing commas (present in round_3_bt.log format but invalid in JSON)
        json_string = re.sub(r',(\s*})', r'\1', json_string)  # Replace ',}' with '}'
        json_string = re.sub(r',(\s*])', r'\1', json_string)  # Replace ',]' with ']'


        print("Attempting to parse JSON data...")
        try:
            # Parse the JSON string
            trade_data = json.loads(json_string)
            if not isinstance(trade_data, list):
                print(f"Error: Parsed JSON is not a list (array).", file=sys.stderr)
                return
            print(f"Successfully parsed {len(trade_data)} trade records.")

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}", file=sys.stderr)
            print("--- Extracted JSON String Attempted ---", file=sys.stderr)
            print(json_string[:500] + "..." if len(json_string) > 500 else json_string, file=sys.stderr) # Print partial string if too long
            print("------------------------------------", file=sys.stderr)
            return
        except Exception as e:
            print(f"An unexpected error occurred during JSON parsing: {e}", file=sys.stderr)
            return


        # Define CSV headers based on the keys in the first trade record (if available)
        if not trade_data:
            print("Warning: No trade data found in the JSON array.", file=sys.stderr)
            # Create an empty CSV with headers anyway
            csv_headers = ['timestamp', 'buyer', 'seller', 'symbol', 'currency', 'price', 'quantity']
        else:
            # Use keys from the first dictionary as headers, assuming consistency
            csv_headers = list(trade_data[0].keys())
            # Ensure standard order if possible
            standard_order = ['timestamp', 'buyer', 'seller', 'symbol', 'currency', 'price', 'quantity']
            ordered_headers = [h for h in standard_order if h in csv_headers] + \
                              [h for h in csv_headers if h not in standard_order]
            csv_headers = ordered_headers


        print(f"Writing data to CSV file: {csv_file_path}")
        # Write data to CSV
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            # Use DictWriter for convenience, handles missing keys gracefully if needed (though not expected here)
            # Use semicolon as delimiter to match the example CSV
            writer = csv.DictWriter(csv_file, fieldnames=csv_headers, delimiter=';')

            # Write the header row
            writer.writeheader()

            # Write the trade data rows
            writer.writerows(trade_data)

        print(f"Successfully extracted trade history to {csv_file_path}")

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

# --- Script Execution ---
if __name__ == "__main__":
    import os
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Extract trade history from log files to CSV')
    parser.add_argument('-i', '--input', type=str, help='Input log file path')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file path')
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite output file if it exists')
    args = parser.parse_args()
    
    # Prompt for input file if not provided as argument
    input_log_file = args.input
    if not input_log_file:
        input_log_file = input("Enter the path to the log file: ")
    
    # Generate a default output file name based on input file if not provided
    output_csv_file = args.output
    if not output_csv_file:
        # Extract base name without extension and append _trades.csv
        base_name = os.path.basename(input_log_file)
        name_without_ext, _ = os.path.splitext(base_name)
        output_csv_file = f"{name_without_ext}_trades.csv"
    
    # Check if output file exists and confirm overwrite if not force flag
    if os.path.exists(output_csv_file) and not args.force:
        confirm = input(f"Output file {output_csv_file} already exists. Overwrite? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
    
    # Run the extraction function
    extract_trade_history_to_csv(input_log_file, output_csv_file)
    
    # After extraction, check if the user wants to process another file
    another = input("Would you like to process another log file? (y/n): ")
    if another.lower() == 'y':
        # Create a recursive call with clean arguments
        sys.argv = [sys.argv[0]]  # Reset args for clean interactive prompting
        os.execv(sys.executable, [sys.executable] + sys.argv)