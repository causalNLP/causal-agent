import json 
from pathlib import Path


def export_info(info, folder, name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    if ".json" not in name:
        name = name + ".json"
    with open(f"{folder}/{name}", "w") as f:
        json.dump(info, f, indent=4)
