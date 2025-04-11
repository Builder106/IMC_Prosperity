import csv
import json
import io
import os # Import the os module

def process_log_file(log_filepath, output_filepath):
    """
    Processes the IMC Prosperity log file to extract and structure data.

    Args:
        log_filepath (str): Path to the input log file.
        output_filepath (str): Path to save the processed output file.
    """
    activities_data = []
    trade_history_data = None
    activities_header = []
    reading_section = None
    trade_history_lines = []

    try:
        with open(log_filepath, 'r') as infile:
            for line in infile:
                stripped_line = line.strip()

                if stripped_line == "Sandbox logs:":
                    reading_section = "sandbox"
                    continue
                elif stripped_line == "Activities log:":
                    reading_section = "activities"
                    # Read the next line as header
                    header_line = next(infile).strip()
                    activities_header = [h.strip() for h in header_line.split(';')]
                    continue
                elif stripped_line == "Trade History:":
                    reading_section = "trades"
                    continue
                elif not stripped_line: # Skip empty lines
                    continue

                if reading_section == "activities":
                    # Use io.StringIO to treat the line as a file for csv.reader
                    line_io = io.StringIO(stripped_line)
                    reader = csv.reader(line_io, delimiter=';')
                    try:
                        row = next(reader)
                        if len(row) == len(activities_header):
                            record = {}
                            for i, header in enumerate(activities_header):
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
                                        record[header] = value # Keep as string if conversion fails
                            activities_data.append(record)
                        else:
                            print(f"Warning: Skipping malformed activities line: {stripped_line}")
                    except StopIteration: # Handle potential empty lines parsed by csv reader
                         print(f"Warning: Skipping empty or invalid CSV line: {stripped_line}")
                    except Exception as e:
                        print(f"Error processing activities line: {stripped_line} - {e}")

                elif reading_section == "trades":
                    trade_history_lines.append(stripped_line)

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_filepath}")
        return
    except Exception as e:
        print(f"An error occurred during file reading: {e}")
        return

    # --- Process Trade History ---
    if trade_history_lines:
        trade_history_string = "".join(trade_history_lines)
        try:
            # Find the start and end of the JSON array
            start_index = trade_history_string.find('[')
            end_index = trade_history_string.rfind(']')
            if start_index != -1 and end_index != -1:
                json_string = trade_history_string[start_index:end_index+1]
                trade_history_data = json.loads(json_string)
            else:
                 print("Warning: Could not find valid JSON array in Trade History section.")
                 trade_history_data = [] # Assign empty list if parsing fails
        except json.JSONDecodeError as e:
            print(f"Error parsing Trade History JSON: {e}")
            print(f"Attempted to parse: {trade_history_string[:500]}...") # Print snippet for debugging
            trade_history_data = [] # Assign empty list if parsing fails
        except Exception as e:
            print(f"An unexpected error occurred during trade history processing: {e}")
            trade_history_data = []


    # --- Write Output ---
    try:
        with open(output_filepath, 'w') as outfile:
            outfile.write("--- Processed Log File ---\n\n")

            outfile.write("## Sandbox Logs Summary\n")
            outfile.write("NOTE: Sandbox logs were present but detailed content is omitted for brevity.\n")
            outfile.write("Check the original log file if details are needed.\n\n")


            outfile.write("## Activities Log (JSON)\n")
            if activities_data:
                json.dump(activities_data, outfile, indent=2)
            else:
                outfile.write("No activities data found or processed.\n")
            outfile.write("\n\n")

            outfile.write("## Trade History (JSON)\n")
            if trade_history_data is not None:
                 json.dump(trade_history_data, outfile, indent=2)
            else:
                 outfile.write("No trade history data found or processed.\n")

        print(f"Successfully processed log file. Output saved to: {output_filepath}")

    except IOError as e:
        print(f"Error writing output file {output_filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during output writing: {e}")


# --- Get Input File Path ---
log_file = input("Please enter the path to the log file: ")

# --- Define Output File Path (based on input filename) ---
if os.path.exists(log_file):
    base_name = os.path.basename(log_file)
    name_part, _ = os.path.splitext(base_name)
    output_dir = os.path.dirname(log_file) # Get directory of input file
    processed_output_file = os.path.join(output_dir, f"{name_part}_processed.txt") # Place output in same directory

    # --- Call the processing function ---
    process_log_file(log_file, processed_output_file)
else:
    print(f"Error: The file '{log_file}' does not exist.")