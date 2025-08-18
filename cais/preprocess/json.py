import pandas as pd
import json 
import argparse 
from pathlib import Path
from typing import List 




def create_json(csv_file_loc:str, output_folder:str, output_file_name:str, 
                base_data_folder:str, data_attribute:str="data_files") -> List:
    """
    Creates a json file containing the causal query and its associated metadata from 
    the csv file 

    Args:
        csv_file_loc: path to the csv file
        output_folder: path to the folder where the json file is saved
        output_file_name: name of the output json file 
        base_data_folder: path to the folder where the data is saved
        data_attribute: name of the column in the csv file containing the data file name
    """

    try:
        df = pd.read_csv(csv_file_loc)
    except FileNotFoundError:
        print(f"File not found:{csv_file_loc}. Make sure the file path is correct.")
    
    json_df = df.to_dict(orient="records")

    print("Checking if referenced csv files are available")
    all_exists = True 
    for data in json_df:
        #print(base_data_folder, data[data_attribute])
        full_path = Path(base_data_folder) / data[data_attribute]
        if not full_path.exists():
            print(f"File not found: {full_path}. Re-check the name of the data file.")
            all_exists = False & all_exists
        else:
            data[data_attribute] = str(full_path)
    
    if not all_exists:
        print("Some data files are missing or incorrectly name")
    else:
        print("All data files are available. Good to go.")
    
    if ".json" not in output_file_name:
        output_file_name = output_file_name + ".json" 

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / output_file_name
    with open(output_file_path, "w") as f:
        json.dump(json_df, f, indent=4)     
    print(f"Json file created at {output_file_path}")
    f.close()   

    return json_df