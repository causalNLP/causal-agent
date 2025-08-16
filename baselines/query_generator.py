import json
from pathlib import Path
from typing import List
import numpy


def export_json(data, output_folder, name):
    """
    Exports the JSON file.
    Args:
        data: (List[dict]) the data to be exported as a JSON file.
        output_folder: (Path) the output folder path.
        name: (str) output file name.
    """

    output_folder.mkdir(exist_ok=True, parents=True)
    if not name.endswith(".json"):
        name = name + ".json"
    full_path = output_folder / name
    with open(full_path, "w") as f:
        json.dump(data, f, indent=4)
        print("JSON file saved as: {}".format(full_path))


class JSONGenerator:
    """
    Base class for generating JSON files.

    Attributes:
        df: (pd.DataFrame) the DataFrame containing the information
    """

    def __init__(self, df):
        self.df = df

    def create_json(
        self,
        columns,
        new_keys,
        save=False,
        output_folder=Path("json_files"),
        name="json_data.json",
    ):
        """
        Creates a JSON file from the source DataFrame

        Args:
            columns: (List[str]) the columns to include.
            new_keys: (dict) (str) mapping of column names in df -> (str) key names in JSON.
            save: (bool) whether to save the JSON file or not.
            output_folder: (Path) path to the output folder.
            name: (str) name of the output JSON file.

        Returns:
            (List[dict])
        """
        new_columns = {}
        for col in columns:
            if col in new_keys:
                new_columns[col] = new_keys[col]
            else:
                new_columns[col] = col

        new_df = self.df[new_columns.keys()].rename(columns=new_columns)
        json_data = new_df.to_dict(orient="records")

        if save:
            export_json(json_data, output_folder, name)

        return json_data


class QueryGenerator:
    """
    Base class for generating query prompts to interface with LLM APIs.

    Attributes:
        json_data: (List[dict]) the source information, originally saved as a JSON file.
        query_key: (str) the key that represents the causal query in the source JSON file
        location_key: (str) the key that represents the filename
        opening_line: (str) the opening sentences of the query prompted to the LLM. The
                      causal query is appended to this.
        key_names: (dict) (str) key name in input file -> (str) key name in output file.
        other_instructions: (str) other instructions to the LLM.
        data_path: (str) the full path to the folder containing the data
    """

    def __init__(
        self,
        json_data,
        query_key,
        location_key,
        opening_line,
        other_instructions,
        key_names,
        data_path,
    ):

        self.json_data = json_data
        self.query_key = query_key
        self.opening_line = opening_line
        self.key_names = key_names
        self.other_instructions = other_instructions
        self.location_key = location_key
        self.data_path = data_path

    def create_query(
        self, save=False, output_folder=Path("json_files"), name="json_data.json"
    ):
        """
        Creates the query in JSON format.

        Args:
            save: (bool) whether to save the JSON file or not.
            output_folder: (Path) path to the output folder.
            name: (str) name of the output JSON file.

        Returns:
            (List[dict])
        """
        all_queries = []
        for data in self.json_data:
            query = {}
            try:
                if str(data[self.query_key]) != "nan":
                    query["query"] = (
                        f"{self.opening_line} {data[self.query_key]} {self.other_instructions}"
                    )
                    query["dataset_path"] = (
                        self.data_path + "/" + str(data[self.location_key])
                    )  # edit this later
                    for key in self.key_names:
                        if key != self.query_key and key != self.location_key:
                            try:
                                query[self.key_names[key]] = data[key]
                            except KeyError:
                                raise KeyError(
                                    f"Missing key '{self.query_key}' in data: {data}"
                                )
                    all_queries.append(query)

            except KeyError:
                raise KeyError(f"Missing key '{self.query_key}' in data: {data}")

        if save:
            export_json(all_queries, output_folder, name)

        return all_queries
