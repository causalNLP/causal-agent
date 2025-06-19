## This file generates realistic contexts for synthetic datasets. It uses GPT to create
## columns names, dataset description, and causal query for synthetic data

## ToDo: change causalscientist after renaming the package 
import argparse
import os
import pandas as pd
import json
from causalscientist.auto_causal.synthetic.prompts import generate_data_summary, create_prompt, filter_question
import sys
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import logging

Path("logs").mkdir(parents=True, exist_ok=True)
logging.config.fileConfig('log_config.ini')


MODEL = "gpt-4"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--metadata_path', type=str, required=True,
                        help='Path to the file containing metadata json files.')
    parser.add_argument('-d', '--dataset_folder', type=str, required=True,
                        help='Path to the folder containing dataset files.')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                        help='Path to the folder where the output json files will be saved.')
    parser.add_argument('-m', '--method', type=str, required=True,
                        help="Method corresponding to the dataset")
    parser.add_argument("-do", "--domain", type=str, default="social science",
                        help="Domain of the dataset")

    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()
    metadata_path = args.metadata_path
    output_folder = args.output_folder
    method = args.method
    domain = args.domain

    # Load metadata files
    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    history = ""
    count = 0
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_responses = {}
    logger = logging.getLogger("description_logger")

    for file in tqdm(sorted(os.listdir(args.dataset_folder))):
        if file.endswith('.csv'):
            logger.info("Generating context for file: %s", file)
            dataset_path = os.path.join(args.dataset_folder, file)
            df = pd.read_csv(dataset_path)
            metadata = all_metadata[file]
            cutoff = None
            if 'cutoff' in metadata:
                cutoff = metadata.get('cutoff')

            ## summary of the raw unlabeled dataset
            summary = generate_data_summary(df, metadata.get('continuous'), metadata.get('binary'),
                                            metadata.get('type'), cutoff=cutoff)
            ## prompt for the LLM
            prompt = create_prompt(summary, metadata.get('type'), domain, history)
            ##print(prompt)
            response = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                                    temperature=0.7)
            response = response.choices[0].message.content
            response_json = json.loads(response)
            filtered_prompt = filter_question(response_json['question'])
            clean_response = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": filtered_prompt}],
                                                            temperature=1.0)
            clean_response = clean_response.choices[0].message.content
            response_json['question'] = clean_response
            data_context = response_json['context']
            data_variables = response_json['variable_labels']
            history += "{}. Context: ".format(count+1) + data_context + "\n"
            ##print(response)
            all_responses[file] = response_json
            count += 1
            #if count == 2:
            #    break
            logger.info("Question: %s", response_json['question'])

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    full_path = output_path / "{}.json".format(method)
    with open(full_path, 'w') as f:
        json.dump(all_responses, f, indent=4)
    logger.info("All contexts are saved in the file: "+ str(full_path))
