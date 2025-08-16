## This file generates realistic contexts for synthetic datasets. It uses GPT to create
## columns names, dataset description, and causal query for synthetic data

import argparse
import os
import pandas as pd
import json
from auto_causal.synthetic.prompts import generate_data_summary, create_prompt, filter_question
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List
import logging 
import logging.config

Path("reproduce_results/logs").mkdir(parents=True, exist_ok=True)
logging.config.fileConfig('reproduce_results/log_config.ini')
logger = logging.getLogger("description_logger")

MODEL = "gpt-4"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--metadata_path', type=str, required=True,
                        help='Path to the file containing metadata json files.')
    parser.add_argument('-d', '--dataset_folder', type=str, required=True,
                        help='Path to a .csv file **or** a folder containing .csv files.')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                        help='Path to the folder where the output json files will be saved.')
    parser.add_argument('-m', '--method', type=str, required=True,
                        help="Method corresponding to the dataset")
    parser.add_argument('-do', '--domain', type=str, default="social science",
                        help="Domain of the dataset")
    return parser.parse_args()

def get_dataset_files(path):
    """
    Get the list of CSV files in a directory or the single CSV file in case a file path is provided.

    Args:
        path (str): Path to a directory or a single CSV file

    Returns:
        List[str]: list of the paths to the CSV files
    """

    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".csv")]
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        return sorted(files)
    if os.path.isfile(path):
        if path.lower().endswith(".csv"):
            return [os.path.abspath(path)]
        raise ValueError(f"Not a CSV file: {path}")
    raise FileNotFoundError(f"No such file or directory: {path}")

if __name__ == "__main__":

    args = parse_args()
    metadata_path = args.metadata_path
    output_folder = args.output_folder
    method = args.method
    domain = args.domain

    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history = ""
    all_responses = {}

    dataset_files = get_dataset_files(args.dataset_folder)
    for dataset_path in tqdm(dataset_files):
        file_name = os.path.basename(dataset_path)
        logger.info("Generating context for file: %s", file_name)

        df = pd.read_csv(dataset_path)
        if file_name not in all_metadata:
            logger.warning("No metadata found for file: %s, skipping.", file_name)
            continue

        metadata = all_metadata[file_name]
        cutoff = metadata.get("cutoff")

        summary = generate_data_summary(df, metadata.get("continuous"), metadata.get("binary"),
                                        metadata.get("type"), cutoff=cutoff)
        prompt = create_prompt(summary, metadata.get("type"), domain, history)
        response = client.chat.completions.create(model=MODEL,
                                                  messages=[{"role": "user", "content": prompt}],
                                                  temperature=0.7).choices[0].message.content
        response_json = json.loads(response)
        filtered_prompt = filter_question(response_json["question"])
        clean_response = client.chat.completions.create(model=MODEL,
                                                        messages=[{"role": "user", "content": filtered_prompt}],
                                                        temperature=0).choices[0].message.content
        response_json["question"] = clean_response

        data_summary = response_json["summary"]
        history += f"{len(all_responses)+1}. Context summary: {data_summary}\n"
        all_responses[file_name] = response_json
        logger.info("Question: %s", response_json["question"])

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    full_path = output_path / f"{method}.json"
    with open(full_path, 'w') as f:
        json.dump(all_responses, f, indent=4)

    logger.info("All contexts are saved in the file: %s", str(full_path))
