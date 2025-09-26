import os, re, io, time, json, logging, contextlib, textwrap
from typing import Dict, Any
import pandas as pd
import argparse
from causal_agent.agent import run_causal_analysis

# Constants
RATE_LIMIT_SECONDS = 2

def run_caia(desc, question, df):
    return run_causal_analysis(query=question, dataset_path=df, dataset_description=desc)

def parse_args():
    parser = argparse.ArgumentParser(description="Run batch causal analysis.")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file with queries, descriptions, and file names.")
    parser.add_argument("--data_folder", type=str, required=True, help="Folder containing data CSVs.")
    parser.add_argument("--data_category", type=str, required=True, help="Dataset category (e.g., real, qrdata, synthetic).")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save output.")
    parser.add_argument("--llm_name", type=str, required=True, help="Name of the LLM used.")
    return parser.parse_args()

def main():
    
    args = parse_args()
    csv_meta = args.csv_meta
    data_dir = args.data_dir
    output_json = args.output_json
    os.environ["LLM_MODEL"] = args.llm_name
    print("[main] Starting batch processing…")

    if not os.path.exists(csv_meta):
        logging.error(f"Meta file not found: {csv_meta}")
        return

    meta_df = pd.read_csv(csv_meta)
    print(f"[main] Loaded metadata CSV with {len(meta_df)} rows.")

    results: Dict[int, Dict[str, Any]] = {}

    for idx, row in meta_df.iterrows():
        data_path = os.path.join(data_dir, str(row["data_files"]))
        print(f"\n[main] Row {idx+1}/{len(meta_df)} → Dataset: {data_path}")

        try:
            res = run_caia(
                desc=row["data_description"],
                question=row["natural_language_query"],
                df=data_path,
            )
            
            # Format result according to specified structure
            formatted_result = {
                "query": row["natural_language_query"],
                "method": row["method"],
                "answer": row["answer"],
                "dataset_description": row["data_description"],
                "dataset_path": data_path,
                "keywords": row.get("keywords", "Causality, Average treatment effect"),
                "final_result": {
                    "method": res['results']['results'].get("method_used"),
                    "causal_effect": res['results']['results'].get("effect_estimate"),
                    "standard_deviation": res['results']['results'].get("standard_error"),
                    "treatment_variable": res['results']['variables'].get("treatment_variable", None),
                    "outcome_variable": res['results']['variables'].get("outcome_variable", None),
                    "covariates": res['results']['variables'].get("covariates", []),
                    "instrument_variable": res['results']['variables'].get("instrument_variable", None),
                    "running_variable": res['results']['variables'].get("running_variable", None),
                    "temporal_variable": res['results']['variables'].get("time_variable", None),
                    "statistical_test_results": res.get("summary", ""),
                    "explanation_for_model_choice": res.get("explanation", ""),
                    "regression_equation": res.get("regression_equation", "")
                }
            }
            results[idx] = formatted_result
            print(type(res))
            print(res)
            print(f"[main] Formatted result for row {idx+1}:", formatted_result)
        except Exception as e:
            logging.error(f"[{idx+1}] Error: {e}")
            results[idx] = {"answer": str(e)}

        time.sleep(RATE_LIMIT_SECONDS)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[main] Done. Predictions saved to {output_json}")

if __name__ == "__main__":
    main()
