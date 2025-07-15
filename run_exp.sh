#!/bin/bash
export ANTHROPIC_API_KEY=""
export GEMINI_API_KEY="AIzaSyDOBFjIBk_fyNuJYf5TAyK8oqpK2nYxNTQ"
export GOOGLE_API_KEY="AIzaSyDOBFjIBk_fyNuJYf5TAyK8oqpK2nYxNTQ"
export TOGETHER_API_KEY="4aada21a6d72ada6310b8617dabdefc8e1ac5bce1df62fcc38a98b8ae9cae215"
export OPENAI_API_KEY="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"
export DEEPSEEK_API_KEY="sk-795319c8be3c4092b6ae767d1f768f29"
export LLM_MODEL="gpt-4o"
export LLM_PROVIDER="openai"

# Function to run test and log output

run_test() {
  local meta=$1
  local data_dir=$2
  local tag=$3
  local log_file="cais_logs/${LLM_MODEL}_${tag}.txt"
  local output_json="cais_outputs/${LLM_MODEL}_${tag}.json"

  echo "=== Running test: $tag with model $LLM_MODEL ==="

  mkdir -p cais_logs
  mkdir -p cais_outputs

  # Remove log file if it already exists
  if [ -f "$log_file" ]; then
    echo "Removing existing log file: $log_file"
    rm "$log_file"
  fi

  # Remove output JSON file if it already exists
  if [ -f "$output_json" ]; then
    echo "Removing existing output file: $output_json"
    rm "$output_json"
  fi

  # Run the test, log output to file only
  python test.py --csv_meta "$meta" --data_dir "$data_dir" --output_json "$output_json" > "$log_file" 2>&1

  echo "=== Finished test: $tag with model $LLM_MODEL ==="
}

# Run tests
run_test data/qr_info.csv data/all_data "qr"
run_test data/real_info.csv data/all_data "real"
run_test data/synthetic_info.csv data/synthetic_data "synthetic"