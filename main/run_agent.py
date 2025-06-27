import csv
import os
import argparse
from pathlib import Path
from auto_causal.agent import run_causal_analysis
import logging
import logging.config
import json 


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run causal analysis over CSV-specified queries.")
    parser.add_argument("-f", "--source_file", type=str, required=True,
                        help="Path to the CSV file containing queries and dataset references.")
    parser.add_argument("-d", "--data_folder", type=str, required=True,
                        help="Path to the folder containing dataset CSV files.")
    parser.add_argument("-t", "--data_collection", type=str, required=True,
                        help="Name of the dataset collection (e.g., synthetic, real, qrdata).")
    parser.add_argument("-o", "--output_folder", type=str, default="output/",
                        help="Directory to save result outputs.")
    parser.add_argument("-l", "--llm", type=str, default="gpt-4o-mini",
                        help="LLM model name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    ## output paths. The output will be file in json like structure
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    output_filename = output_path / f"{args.data_collection}_{args.llm}.txt"

    # logs 
    log_path = Path("reproduce_results/logs/runs")
    log_path.mkdir(parents=True, exist_ok=True)
    logging.config.fileConfig('reproduce_results/log_config.ini')
    logger = logging.getLogger(f"runs_logger")

    logger.info(f"Input CSV : {args.source_file}")
    logger.info(f"Output file: {output_filename}")
    logger.info(f"LLM: {args.llm}")

    queries_processed = 0

    try:
        with open(args.source_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                raise ValueError(f"[ERROR] CSV file '{args.source_file}' has no headers.")

            required_columns = ['data_description', 'natural_language_query', 'data_files']
            missing_cols = [col for col in required_columns if col not in reader.fieldnames]
            if missing_cols:
                raise ValueError(f"[ERROR] Missing required columns: {', '.join(missing_cols)}")

            for row_num, row in enumerate(reader, start=1):
                data_description = row['data_description'].strip()
                query = row['natural_language_query'].strip()
                csv_file = row['data_files'].strip()
                answer = row.get('answer', '').strip()

                if not data_description or not query or not csv_file:
                    logger.warning(f"Row {row_num}: Missing required fields. Skipping.")
                    continue

                csv_path = os.path.join(args.data_folder, csv_file)

                logger.info("--------------------------------------------------")
                logger.info(f"Query #{row_num}: {query}")
                logger.info(f"Dataset file: {csv_file}")

                try:
                    result = run_causal_analysis(query, csv_path, data_description)
                    print("Result:", result)
                    result_str = json.dumps(result, indent=2)
                    queries_processed += 1
                    logger.info("Analysis completed successfully.")
                except TypeError as e:
                    result_str = str(result)
                except Exception as e:
                    logger.error(f"Failed to process row {row_num}: {e}")
                with open(output_filename, 'a', encoding='utf-8') as result_file:
                        result_file.write(result_str + "\n")
                

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
  
    logger.info(f"Total queries successfully processed: {queries_processed}")
