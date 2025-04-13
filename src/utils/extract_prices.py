import csv
import os
import sys
import json
import io

def extract_price_data_to_csv(log_filepath, csv_filepath):
    """
    Extracts price data from a log file and saves it to a CSV file.
    
    Args:
        log_filepath (str): Path to the input log file
        csv_filepath (str): Path to the output CSV file
    """
    activities_marker = "Activities log:"
    price_data = []
    headers = []
    reading_activities = False
    
    print(f"Reading log file: {log_filepath}")
    try:
        with open(log_filepath, 'r', encoding='utf-8') as log_file:
            for line in log_file:
                stripped_line = line.strip()
                
                # Check if we found the Activities log marker
                if activities_marker in stripped_line:
                    reading_activities = True
                    # Read the next line which should contain headers
                    headers_line = next(log_file).strip()
                    headers = [h.strip() for h in headers_line.split(';')]
                    print(f"Found Activities log section with headers: {headers}")
                    continue
                
                # If we're in the Activities log section, process data lines
                if reading_activities and stripped_line:
                    # Skip if we encounter a new section
                    if stripped_line == "Trade History:" or stripped_line == "Sandbox logs:":
                        reading_activities = False
                        break
                    
                    # Process the line as CSV
                    line_io = io.StringIO(stripped_line)
                    reader = csv.reader(line_io, delimiter=';')
                    try:
                        row = next(reader)
                        if len(row) == len(headers):
                            record = {}
                            for i, header in enumerate(headers):
                                value = row[i].strip()
                                if value == '':
                                    record[header] = None
                                else:
                                    # Attempt to convert to number (int or float)
                                    try:
                                        if '.' in value:
                                            record[header] = float(value)
                                        else:
                                            record[header] = int(value)
                                    except ValueError:
                                        record[header] = value  # Keep as string if conversion fails
                            price_data.append(record)
                        else:
                            print(f"Warning: Skipping malformed line: {stripped_line}")
                    except StopIteration:  # Handle empty lines
                        continue
                    except Exception as e:
                        print(f"Error processing line: {stripped_line} - {e}")
        
        if not price_data:
            print(f"No price data found in the log file: {log_filepath}")
            return False
        
        # Write data to CSV
        print(f"Writing {len(price_data)} price data rows to CSV file: {csv_filepath}")
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers, delimiter=';')
            writer.writeheader()
            writer.writerows(price_data)
        
        print(f"Successfully extracted price data to {csv_filepath}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_filepath}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def main():
    """
    Main function to handle user input and control the extraction process.
    """
    print("Price Data Extraction Tool")
    print("-------------------------")
    
    # Get input file path
    log_file = input("Please enter the path to the log file: ").strip()
    
    if not os.path.exists(log_file):
        print(f"Error: The file '{log_file}' does not exist.")
        return
    
    # Define the output CSV file path
    base_name = os.path.basename(log_file)
    name_part, _ = os.path.splitext(base_name)
    output_dir = os.path.dirname(log_file)
    
    # Allow user to specify output path or use default
    use_default = input(f"Use default output path ({output_dir}/{name_part}_prices.csv)? (y/n): ").strip().lower()
    
    if use_default == 'y' or use_default == '':
        csv_file = os.path.join(output_dir, f"{name_part}_prices.csv")
    else:
        csv_file = input("Enter the path for the output CSV file: ").strip()
        # Add .csv extension if not provided
        if not csv_file.endswith('.csv'):
            csv_file += '.csv'
    
    # Check if output file already exists
    if os.path.exists(csv_file):
        overwrite = input(f"The file {csv_file} already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Operation cancelled.")
            return
    
    # Execute extraction
    success = extract_price_data_to_csv(log_file, csv_file)
    
    if success:
        # Ask if user wants to view a sample of extracted data
        show_sample = input("Would you like to display a sample of the extracted data? (y/n): ").strip().lower()
        if show_sample == 'y':
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    sample_lines = [f.readline() for _ in range(6)]  # Header + 5 data lines
                    print("\nSample of extracted data:")
                    for line in sample_lines:
                        print(line.strip())
            except Exception as e:
                print(f"Error displaying sample data: {e}")
                
        # Ask if user wants to process another file
        another = input("\nWould you like to extract price data from another file? (y/n): ").strip().lower()
        if another == 'y':
            main()  # Recursive call for another file
        else:
            print("Done. Goodbye!")
    else:
        retry = input("Would you like to try again with a different file? (y/n): ").strip().lower()
        if retry == 'y':
            main()  # Recursive call to try again
        else:
            print("Operation failed. Exiting.")

if __name__ == "__main__":
    main()
