import json
import os
import tqdm
import argparse
import pandas as pd

# Ensure project root is on sys.path when running this file directly
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Use local baselines package instead of external causalscientist
import baselines as causalscientist
from baselines import (
    CausalQueryFormat,
    CausalQueryVeridicalFormat,
    SequentialCausalThinking,
)
from baselines.query_formats import ProgramOfThoughtsFormat, ReActFormat


def main(args):
    queries_path = args.queries

    # Determine the base path for datasets
    if args.data_type == 'qrdata':
        base_path = 'data/all_data/'
    elif args.data_type == 'real':
        base_path = 'data/real_data/'
    elif args.data_type == 'synthetic':
        base_path = 'data/synthetic_data/'
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")

    # Load queries based on file type
    if queries_path.endswith('.csv'):
        df = pd.read_csv(queries_path)
        # Rename columns to match the expected format
        df = df.rename(columns={
            'natural_language_query': 'query',
            'data_description': 'dataset_description',
            'data_files': 'dataset_path'
        })
        queries = df.to_dict('records')
    elif queries_path.endswith('.json'):
        with open(queries_path, "r") as f:
            queries = json.load(f)
    else:
        raise ValueError("Unsupported file type for --queries. Please use .csv or .json")

    # Unify dataset path construction
    for q in queries:
        filename = os.path.basename(q['dataset_path'])
        q['dataset_path'] = os.path.join(base_path, filename)

    if args.rpc_address:
        chatbot = causalscientist.RPCChatbot(args.rpc_address)
    else:
        # Initialize the chatbot
        if args.api == "test":
            chatbot = causalscientist.TestChatbot()
        elif args.api == "vertex":
            chatbot = causalscientist.VertexAPIChatbot(model=args.model, persistent_mode=args.persistent)
        elif args.api == "azure":
            chatbot = causalscientist.AzureAPIChatbot(model=args.model, persistent_mode=args.persistent)
        elif args.api == "openai":
            chatbot = causalscientist.OpenAIAPIChatbot(model=args.model, persistent_mode=args.persistent)
        elif args.api == "together":
            chatbot = causalscientist.TogetherAPIChatbot(model=args.model, persistent_mode=args.persistent)
        elif args.api == "local":
            raise NotImplementedError("Local chatbot is not implemented yet.")
        else:
            raise ValueError(f"Invalid API: {args.api}")

    # Get query format
    try:
        query_format = getattr(causalscientist, args.query_format)
    except AttributeError:
        raise ValueError(f"Invalid query format: {args.query_format}")

    # Initialize the scientist with persistent mode if requested
    scientist = causalscientist.CAISBaseline(
        chatbot, 
        persistent=args.persistent, 
        session_timeout=args.session_timeout
    )

    # Start persistent session if enabled
    if args.persistent:
        print("Starting persistent Python environment...")
        if scientist.start_persistent_session():
            print("Persistent environment started successfully.")
            
            # If using persistent mode, update the chatbot's system message
            if hasattr(chatbot, 'persistent_mode'):
                chatbot.persistent_mode = True
                print("Updated chatbot to use persistent mode.")
        else:
            print("Failed to start persistent environment. Falling back to one-off mode.")
            args.persistent = False

    output = []
    try:
        for q in tqdm.tqdm(queries):
            query = q["query"]
            dataset_path = q["dataset_path"]
            dataset_description = q["dataset_description"]
            
            # If in persistent mode, upload the dataset file to the container
            if args.persistent and os.path.exists(dataset_path):
                print(f"Uploading dataset file {dataset_path} to container...")
                # Use the same path structure in the container as the original path
                container_path = dataset_path
                upload_result = scientist.upload_file(dataset_path, container_path)
                print(upload_result)
                
                # No need to update the dataset path as we're using the same path structure
                print(f"Dataset uploaded to container at path: {container_path}")

            # Iterate over the queries
            qf = CausalQueryFormat
            if args.veridical:
                qf = CausalQueryVeridicalFormat
            if args.sequential:
                qf = SequentialCausalThinking
            if args.potm:
                qf = ProgramOfThoughtsFormat
            if args.react:
                qf = ReActFormat

            result = scientist.answer(
                query, dataset_path, dataset_description, qf=qf, post_steps=False
            )

            output.append(
                {
                    **q,
                    "result": result,
                }
            )
            print(result)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

    # Save the output
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
        
    # Clean up persistent session if it was used
    if args.persistent:
        print("Stopping persistent Python environment...")
        scientist.stop_persistent_session()
        print("Persistent environment stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_scientist.py",
        description="Run the causal scientist",
        epilog="Example: python run_scientist.py --queries queries/queries.json --output runs/output.json --model google/gemini-1.5-flash-002 --chatbot_type api",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="benchmark/qrdata/qrdata_info.csv",
        help="Path to the queries file (JSON or CSV)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/output.json",
        help="Path to the output json file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-1.5-flash-002",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--query-format",
        type=str,
        default="CausalQueryFormat",
        help="Name of the QueryFormat class to use",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="qrdata",
        choices=['qrdata', 'real', 'synthetic'],
        help="Type of data to process (qrdata, real, or synthetic)",
    )
    parser.add_argument(
        "--api",
        type=str,
        default="azure",
        help="Type of API to use. Options: vertex, azure, test, local, openai, together. Choosing 'local' will use a local chatbot.",
    )
    parser.add_argument(
        "--rpc-address",
        type=str,
        default=None,
        help="Address of the RPC server to connect to (will override the --api flag)",
    )
    parser.add_argument(
        "--veridical",
        action=argparse.BooleanOptionalAction,
        help="Use the veridical prompting method",
    )
    parser.add_argument(
        "--sequential",
        action=argparse.BooleanOptionalAction,
        help="Use the sequential thinking approach for causal analysis",
    )
    parser.add_argument(
        "--potm",
        action=argparse.BooleanOptionalAction,
        help="Use the program of thoughts approach for causal analysis",
    )
    parser.add_argument(
        "--react",
        action=argparse.BooleanOptionalAction,
        help="Use the ReAct approach for causal analysis",
    )
    parser.add_argument(
        "--method-explanation",
        action=argparse.BooleanOptionalAction,
        help="(For the baseline) Use method explanation",
    )
    parser.add_argument(
        "--persistent",
        action=argparse.BooleanOptionalAction,
        help="Use persistent Python environment for code execution",
    )
    parser.add_argument(
        "--session-timeout",
        type=int,
        default=3600,
        help="Timeout for persistent sessions in seconds (default: 3600)",
    )

    args = parser.parse_args()

    main(args)



