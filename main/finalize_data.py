## This file contains the functions that creates the final synthetic dataset by renaming the columns. Additionally, 
## it also create a csv file summarizing the information about the dataset including description, file name, query, etc. 
## This csv file is used to create the input to the pipeline. 

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import json
import os
from tqdm import tqdm


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument("-id", "--input_data_path", type=str, required=True,
                        help="Path to the folder where the raw data is stored")
    parser.add_argument("-od", "--output_data_path", type=str, required=True,
                        help="Path to the folder where the processed data will be saved")
    parser.add_argument("-o", "--output_path", type=str, required=True,
                        help="Path to the folder where the summary will be saved")
    parser.add_argument("-md", "--metadata_path", type=str, required=True,
                        help="Path to the folder where the metadata is stored")
    parser.add_argument("-de", "--description_json", type=str, required=True,
                        help="Path to the json file containing the description and queries")
    parser.add_argument("-m", "--method", type=str, required=True,
                        help="Method associated with the dataset")
    return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()

    with open(args.description_json, 'r') as f:
        description_json = json.load(f)

    with open(args.metadata_path, 'r') as f:
        metadata_json = json.load(f)

    output_info_path = Path(args.output_path)
    output_info_path.mkdir(parents=True, exist_ok=True)

    output_data_path = Path(args.output_data_path)
    output_data_path.mkdir(parents=True, exist_ok=True)


    info_dict = {"paper_name":[], "data_description":[], "natural_language_query":[], "answer":[],
                 "method":[], "data_files":[]}

    for file in tqdm(os.listdir(args.input_data_path)):
        if file.endswith('.csv'):
            dataset_path = os.path.join(args.input_data_path, file)
            df = pd.read_csv(dataset_path)
            metadata = metadata_json[file]
            info = description_json[file]
            df_copy = df.copy()
            var_names = info.get('variable_labels')
            #print(var_names)
            df_copy = df_copy.rename(columns=var_names)
            info_dict["paper_name"].append("Synthetic Dataset")
            info_dict["data_description"].append(info.get('description'))
            info_dict["natural_language_query"].append(info.get('question'))
            info_dict["answer"].append(metadata.get('true_effect'))
            info_dict["method"].append(args.method)
            info_dict["data_files"].append(file)
            df_copy.to_csv(output_data_path / file, index=False)
    df = pd.DataFrame(info_dict)
    df.to_csv(output_info_path/f"{args.method}_info.csv", index=False)
