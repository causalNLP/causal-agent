"""
Command-line interface for Causal Agent.

This module provides a CLI for running causal analysis either as single queries
or batch processing from metadata files.

Usage:
    # Single analysis
    causal_agent run dataset.csv "What is the effect of treatment on outcome?"
    
    # Batch analysis  
    causal_agent batch metadata.csv data_folder/ results.json
"""

import os
import argparse
from typing import Optional

from .agent import run_causal_analysis


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Causal Agent: Run causal analysis")
    subparsers = parser.add_subparsers(dest="command")

    # Simple one-off run
    single = subparsers.add_parser("run", help="Run a single causal analysis")
    single.add_argument("dataset", help="Path to CSV dataset")
    single.add_argument("query", help="Natural language causal question")
    single.add_argument("--desc", dest="description", default=None, help="Dataset description text")
    single.add_argument("--llm-name", dest="llm_name", default=None, help="LLM model name")
    single.add_argument("--llm-provider", dest="llm_provider", default=None, help="LLM provider (openai, anthropic, together, gemini, deepseek)")

    # Batch run compatible with existing metadata CSVs
    batch = subparsers.add_parser("batch", help="Run batch analyses from a metadata CSV")
    batch.add_argument("csv_path", help="Metadata CSV with columns: natural_language_query, data_description, data_files, etc.")
    batch.add_argument("data_folder", help="Folder containing the datasets")
    batch.add_argument("output_file", help="Path to save output JSON")
    batch.add_argument("--llm-name", dest="llm_name", default=None)
    batch.add_argument("--llm-provider", dest="llm_provider", default=None)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if getattr(args, "llm_name", None):
        os.environ["LLM_MODEL"] = args.llm_name
    if getattr(args, "llm_provider", None):
        os.environ["LLM_PROVIDER"] = args.llm_provider

    if args.command == "run":
        result = run_causal_analysis(query=args.query, dataset_path=args.dataset, dataset_description=args.description)
        import json
        print(json.dumps(result, indent=2))
        return

    if args.command == "batch":
        import json
        import pandas as pd
        from typing import Dict, Any

        meta_df = pd.read_csv(args.csv_path)
        results: Dict[int, Dict[str, Any]] = {}
        for idx, row in meta_df.iterrows():
            data_path = os.path.join(args.data_folder, str(row["data_files"]))
            try:
                res = run_causal_analysis(
                    query=row.get("natural_language_query"),
                    dataset_path=data_path,
                    dataset_description=row.get("data_description"),
                )
                results[idx] = {
                    "query": row.get("natural_language_query"),
                    "method": row.get("method"),
                    "answer": row.get("answer"),
                    "dataset_description": row.get("data_description"),
                    "dataset_path": data_path,
                    "final_result": {
                        "method": res.get('results', {}).get('results', {}).get("method_used"),
                        "causal_effect": res.get('results', {}).get('results', {}).get("effect_estimate"),
                        "standard_deviation": res.get('results', {}).get('results', {}).get("standard_error"),
                        "treatment_variable": res.get('results', {}).get('variables', {}).get("treatment_variable"),
                        "outcome_variable": res.get('results', {}).get('variables', {}).get("outcome_variable"),
                        "covariates": res.get('results', {}).get('variables', {}).get("covariates", []),
                        "instrument_variable": res.get('results', {}).get('variables', {}).get("instrument_variable"),
                        "running_variable": res.get('results', {}).get('variables', {}).get("running_variable"),
                        "temporal_variable": res.get('results', {}).get('variables', {}).get("time_variable"),
                    }
                }
            except Exception as e:
                results[idx] = {"error": str(e)}

        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to {args.output_file}")

