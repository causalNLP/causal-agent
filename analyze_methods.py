"""
Analyze cais_outputs JSON files to summarize method choices per model and data type.
"""

import json
import os
import pandas as pd
from collections import Counter
from pathlib import Path

OUTPUT_DIR = "cais_outputs"


def parse_filename(fname):
    """Extract model name and data type from filename like 'gpt-4o_real.json'."""
    stem = Path(fname).stem  # e.g. 'gpt-4o_real'
    # Split on last underscore to separate model from data type
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, "unknown"


def get_data_shape(dataset_path):
    """Read csv and return (rows, cols) or None if file not found."""
    try:
        df = pd.read_csv(dataset_path)
        return df.shape
    except Exception:
        return None


def main():
    json_files = sorted(
        f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")
    )

    for fname in json_files:
        model, data_type = parse_filename(fname)
        filepath = os.path.join(OUTPUT_DIR, fname)

        with open(filepath) as f:
            data = json.load(f)

        # Group queries by dataset to show shape info
        dataset_queries = {}  # dataset_path -> list of chosen methods
        for key, entry in data.items():
            dataset_path = entry.get("dataset_path", "unknown")
            chosen_method = entry.get("final_result", {}).get("method", "unknown")
            dataset_queries.setdefault(dataset_path, []).append(chosen_method)

        # Print header
        print("=" * 70)
        print(f"Model: {model}  |  Data Type: {data_type}")
        print(f"Total queries: {len(data)}")
        print("-" * 70)

        # Overall method counts
        all_methods = []
        for methods in dataset_queries.values():
            all_methods.extend(methods)
        method_counts = Counter(all_methods)

        print(f"{'Method':<30} {'Count':>6}")
        print(f"{'------':<30} {'-----':>6}")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"{method:<30} {count:>6}")
        print()

        # Per-dataset breakdown
        for dpath, methods in sorted(dataset_queries.items()):
            shape = get_data_shape(dpath)
            shape_str = f"{shape[0]} x {shape[1]}" if shape else "N/A"
            dataset_name = Path(dpath).stem
            print(f"  Dataset: {dataset_name:<35} Shape: {shape_str}")
            per_ds_counts = Counter(methods)
            for method, count in sorted(per_ds_counts.items(), key=lambda x: -x[1]):
                print(f"    {method:<28} {count:>4}")

        print()


if __name__ == "__main__":
    main()
