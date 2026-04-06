import os, time, json, logging, signal
from typing import Dict, Any
import pandas as pd
import argparse
from datetime import datetime
from cais.agent import run_causal_analysis, CausalAgent

# Constants
RATE_LIMIT_SECONDS = 1

def parse_args():
    parser = argparse.ArgumentParser(description="Run batch causal analysis (CAIS).")

    # Preferred flags (match README.md)
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=False,
        help="Path to the metadata CSV (columns: natural_language_query, data_description, data_files, etc.)",
    )
    parser.add_argument("--data_dir", type=str, required=False, help="Folder containing dataset CSV files.")
    parser.add_argument("--output_dir", type=str, required=False, help="Folder to save outputs.")
    parser.add_argument("--output_name", type=str, required=False, help="Output JSON filename (with or without .json).")
    parser.add_argument("--llm_name", type=str, required=False, help="LLM model name (e.g., gpt-4o-mini).")
    parser.add_argument("--llm_provider", type=str, required=False, help="LLM provider (e.g., openai, anthropic).")
    parser.add_argument("--skip-method-validator", action="store_true", help="Skip method validation step.")
    parser.add_argument("--use-decision-tree", action="store_true", help="Use LLM-based method selection.")
    parser.add_argument("--rows", type=str, required=False, help="Path to text file containing specific rows to run.")

    # Back-compat aliases (older/other scripts)
    parser.add_argument("--csv_path", type=str, required=False, help=argparse.SUPPRESS)
    parser.add_argument("--data_folder", type=str, required=False, help=argparse.SUPPRESS)
    parser.add_argument("--output_folder", type=str, required=False, help=argparse.SUPPRESS)
    parser.add_argument("--output_json", type=str, required=False, help=argparse.SUPPRESS)
    parser.add_argument("--csv_meta", type=str, required=False, help=argparse.SUPPRESS)
    parser.add_argument("--data_dir_legacy", dest="data_dir", type=str, required=False, help=argparse.SUPPRESS)
    parser.add_argument("--output_dir_legacy", dest="output_dir", type=str, required=False, help=argparse.SUPPRESS)

    # Kept for compatibility; not used by this script.
    parser.add_argument("--data_category", type=str, required=False, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Normalize/resolve aliases
    if not args.metadata_path:
        args.metadata_path = args.csv_path or args.csv_meta
    if not args.data_dir:
        args.data_dir = args.data_folder
    if not args.output_dir:
        args.output_dir = args.output_folder

    # Allow direct output_json path, otherwise build from output_dir + output_name.
    if args.output_json:
        args.output_file = args.output_json
    else:
        output_name = args.output_name
        if output_name and not output_name.endswith(".json"):
            output_name = f"{output_name}.json"
        args.output_file = os.path.join(args.output_dir or "", output_name or "")

    missing = []
    if not args.metadata_path:
        missing.append("--metadata_path")
    if not args.data_dir:
        missing.append("--data_dir")
    if not args.output_json:
        if not args.output_dir:
            missing.append("--output_dir")
        if not args.output_name:
            missing.append("--output_name")
    if not args.output_file:
        missing.append("--output_json or (--output_dir + --output_name)")
    if not args.llm_name:
        missing.append("--llm_name")
    if not args.llm_provider:
        missing.append("--llm_provider")
    if missing:
        parser.error("Missing required arguments: " + ", ".join(missing))

    return args

def get_last_idx(jsonl_path) -> int:
    last_idx = -1
    if not os.path.exists(jsonl_path):
        return last_idx  # nothing processed yet

    with open(jsonl_path, "r") as f:
        for line in f:
            last_idx = next(iter(json.loads(line)))
    return int(last_idx)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException



def main():
    args = parse_args()
    os.makedirs('./logs/', exist_ok=True)
    logging.basicConfig(
        filename=f"logs/{args.output_file.split('/')[-1][:-5]}_{datetime.now():%Y-%m-%d}.log",
        filemode='a',
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s (%(module)s) - %(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)

    to_skip = None
    if args.rows:
        try:
            with open(args.rows, "r") as f:
                to_skip = f.read().splitlines()
                to_skip = set([int(skip) for skip in to_skip])

        except FileNotFoundError as e:
            print(f"Invalid path provided to optional --rows argument. Defaulting to all rows")
            logger.warning("Specific rows provided but path is invalid. Defaulting to all rows.")
        except Exception as e:
            print(f"Error reading rows: {e} Defaulting to using all rows.")
            logger.warning("Specific rows provided but path is invalid. Defaulting to all rows.")
            to_skip = None

    csv_meta = args.metadata_path
    data_dir = args.data_dir
    output_json = args.output_file
    os.environ["LLM_MODEL"] = args.llm_name
    os.environ["LLM_PROVIDER"] = args.llm_provider
    if args.use_decision_tree:
        os.environ["CAIS_USE_DECISION_TREE"] = "1"
    print("[main] Starting batch processing…")

    if not os.path.exists(csv_meta):
        logging.error(f"Meta file not found: {csv_meta}")
        return

    try:
        meta_df = pd.read_csv(csv_meta)
    except:
        meta_df = pd.read_csv(csv_meta, encoding='latin-1')
    print(f"[main] Loaded metadata CSV with {len(meta_df)} rows.")

    
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "a") as file:
        leftoff = get_last_idx(output_json)
        for idx, row in meta_df.iterrows():
            if (idx <= leftoff) or (to_skip and idx not in to_skip): # only process the specified rows provided or skip rows already processed if re-running
                continue

            data_path = os.path.join(data_dir, str(row["data_files"]))
            print(f"\n[main] Row {idx+1}/{len(meta_df)} → Dataset: {data_path}")
            desc=row["description"] if 'description' in row else row["data_description"]
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(600) # timeout after 10 minutes
                logger.info(f"Attempting to run CAIS on row [{idx}/{meta_df.shape[0]}]")

                print('Starting run!')

                cais = CausalAgent()
                
                cais.run_analysis(
                    query=row["natural_language_query"],
                    dataset_path=data_path,
                    dataset_description=desc,
                    use_decision_tree=True
                )

            except Exception as e:
                logging.error(f"[row {idx}] Error: {e}")
                file.write(json.dumps({idx: str(e)}) + "\n")

            time.sleep(RATE_LIMIT_SECONDS)

        print(f"[main] Done. Predictions saved to {output_json}")

if __name__ == "__main__":
    main()
