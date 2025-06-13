import csv
import json
import os
import time
from auto_causal.agent import run_causal_analysis
import auto_causal.components.output_formatter as cs_output_formatter
# Remove the direct import of cs_method_executor if it causes issues, we'll use importlib
# import auto_causal.tools.method_executor_tool as cs_method_executor
import importlib # Import importlib

# --- Configuration ---
# Absolute path as specified by user for the output log file
OUTPUT_LOG_FILE = "Project/fork_/causalscientist/tests/output/qr_data_4o-mini_latest"
# Relative path to the input CSV file from the workspace root
INPUT_CSV_PATH = "benchmark/qr_revised.csv"
# Prefix for constructing dataset paths
DATA_FILES_BASE_DIR = "benchmark/all_data_1/"

# --- Placeholder for the core analysis function ---
# This function needs to be implemented or imported from elsewhere.
# For the purpose of this script, it's a placeholder.
def benchmark_causal_analysis(natural_language_query: str, dataset_path: str, data_description: str):
    """
    Placeholder for the actual causal analysis function.
    This function would typically perform the analysis based on the inputs.
    """
    print(f"[INFO] run_causal_analysis called with:")
    print(f"  Natural Language Query: '{natural_language_query}'")
    print(f"  Dataset Path: '{dataset_path}'")
    # print(f"  Data Description: '{data_description[:100]}...' (truncated)") # Truncate for brevity if needed
    
    # Simulate some processing time
    # time.sleep(0.1) # Optional: Simulate work
    
    run_causal_analysis(natural_language_query, dataset_path, data_description)

    # TODO: Replace this with actual analysis logic.
    # Example: Simulate failure for demonstration purposes.
    # import random
    # # Fail if "example_fail_condition" is in the query or randomly
    # if "example_fail_condition" in natural_language_query.lower() or random.random() < 0.1: # ~10% chance of failure
    #     print("[WARN] Simulating a failure in run_causal_analysis.")
    #     raise Exception("Simulated analysis error from run_causal_analysis")
    
    print(f"[INFO] run_causal_analysis for '{dataset_path}' completed successfully.")
    # Actual implementation might return a result or have side effects.


def main():
    # Set the log file path for the output_formatter module
    cs_output_formatter.CURRENT_OUTPUT_LOG_FILE = OUTPUT_LOG_FILE
    
    # Set the log file path for the method_executor_tool module using importlib
    try:
        method_executor_module_name = "auto_causal.tools.method_executor_tool"
        cs_method_executor_module = importlib.import_module(method_executor_module_name)
        cs_method_executor_module.CURRENT_OUTPUT_LOG_FILE = OUTPUT_LOG_FILE
        print(f"[INFO] Successfully set CURRENT_OUTPUT_LOG_FILE for {method_executor_module_name} to: {OUTPUT_LOG_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to set CURRENT_OUTPUT_LOG_FILE for method_executor_tool: {e}")
        # Decide if you want to return or continue if this fails
        return

    # Ensure the output directory for the log file exists
    output_log_dir = os.path.dirname(OUTPUT_LOG_FILE)
    if not os.path.exists(output_log_dir):
        try:
            os.makedirs(output_log_dir)
            print(f"Created directory: {output_log_dir}")
        except OSError as e:
            print(f"[ERROR] Failed to create directory '{output_log_dir}': {e}")
            return # Stop if we can't create the log directory

    current_query_sequence_number = 0
    processed_csv_rows = 0

    print(f"Starting processing of CSV: {INPUT_CSV_PATH}")
    print(f"Output log will be written to: {OUTPUT_LOG_FILE}")

    try:
        with open(INPUT_CSV_PATH, mode='r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            if not csv_reader.fieldnames:
                print(f"[ERROR] CSV file '{INPUT_CSV_PATH}' is empty or has no header.")
                return

            required_columns = ['data_description', 'natural_language_query', 'data_files']
            missing_cols = [col for col in required_columns if col not in csv_reader.fieldnames]
            if missing_cols:
                print(f"[ERROR] Missing required columns in CSV file '{INPUT_CSV_PATH}': {', '.join(missing_cols)}")
                print(f"Available columns: {csv_reader.fieldnames}")
                return

            for row_number, row in enumerate(csv_reader, 1):
                processed_csv_rows += 1
                data_description = row.get('data_description', '').strip()
                natural_language_query = row.get('natural_language_query', '').strip()
                data_files_string = row.get('data_files', '').strip()
                answer = row.get('answer', '').strip()

                if not data_files_string:
                    print(f"[WARN] CSV Row {row_number}: 'data_files' field is empty. Skipping.")
                    continue

                individual_files = [f.strip() for f in data_files_string.split(',') if f.strip()]

                if not individual_files:
                    print(f"[WARN] CSV Row {row_number}: 'data_files' contained only separators or was effectively empty after stripping. Original: '{data_files_string}'. Skipping.")
                    continue
                    
                for file_name in individual_files:
                    current_query_sequence_number += 1
                    
                    dataset_path = os.path.join(DATA_FILES_BASE_DIR, file_name)
                    
                    log_data = {
                        "query_number": current_query_sequence_number,
                        "natural_language_query": natural_language_query,
                        "dataset_path": dataset_path,
                        "answer": answer
                    }
                    
                    try:
                        with open(OUTPUT_LOG_FILE, mode='a', encoding='utf-8') as log_file:
                            log_file.write('\n' + json.dumps(log_data) + '\n')
                    except IOError as e:
                        print(f"[ERROR] Failed to write pre-analysis log for query #{current_query_sequence_number} to '{OUTPUT_LOG_FILE}': {e}")
                        continue # Skip to next file/row if logging fails

                    successful_analysis = False
                    for attempt in range(2): # Attempt 0 (first try), Attempt 1 (retry)
                        try:
                            print(f"[INFO] --- Starting Analysis (Attempt {attempt + 1}/2) ---")
                            print(f"[INFO] Query Sequence #: {current_query_sequence_number}")
                            print(f"[INFO] CSV Row: {row_number}, File: '{file_name}'")
                            benchmark_causal_analysis(
                                natural_language_query=natural_language_query,
                                dataset_path=dataset_path,
                                data_description=data_description
                            )
                            successful_analysis = True
                            print(f"[INFO] --- Analysis Successful (Attempt {attempt + 1}/2) ---")
                            break 
                        except Exception as e:
                            print(f"[ERROR] run_causal_analysis failed on attempt {attempt + 1}/2 for query #{current_query_sequence_number}: {e}")
                            if attempt == 1: # This was the retry, and it also failed
                                print(f"[INFO] Both attempts failed for query #{current_query_sequence_number}.")
                                try:
                                    with open(OUTPUT_LOG_FILE, mode='a', encoding='utf-8') as log_file:
                                        log_file.write(f"\n{current_query_sequence_number}:Failed\n")
                                except IOError as ioe_fail:
                                     print(f"[ERROR] Failed to write failure status for query #{current_query_sequence_number} to '{OUTPUT_LOG_FILE}': {ioe_fail}")
                            else:
                                print(f"[INFO] Will retry query #{current_query_sequence_number}.")
                                # time.sleep(1) # Optional: wait a bit before retrying
                                
    except FileNotFoundError:
        print(f"[ERROR] Input CSV file not found: '{INPUT_CSV_PATH}'")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during script execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"--- Script finished ---")
        print(f"Total CSV rows processed: {processed_csv_rows}")
        print(f"Total analysis calls attempted (query_number): {current_query_sequence_number}")
        print(f"Log file: {OUTPUT_LOG_FILE}")

if __name__ == "__main__":
    main() 