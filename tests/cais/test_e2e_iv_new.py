import unittest
import json
import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cais.agent import CausalAgent


class TestE2EIVNewPipeline(unittest.TestCase):
    def test_iv_llm_pipeline_bulk(self):
        """Run several queries from the CSV data and log LLM outputs."""
        csv_path = os.path.join(ROOT, "data", "checked_real_data - Final.csv")
        df = pd.read_csv(csv_path)

        # Filter for entries that specifically use IV as the method
        iv_methods = ['iv', 'iva', 'tsls', '2sls', 'iv_reg', 'iv-reg', 'instrumental_variable']
        iv_df = df[df['method'].str.strip().str.lower().isin(iv_methods)]
        
        # Take a subset of unique queries
        sample_df = iv_df.head(3)

        results_log = []
        output_file = os.path.join(os.path.dirname(__file__), "llm_outputs.json")

        print(f"--- Running Bulk E2E Test (5 queries) ---")
        
        for idx, row in sample_df.iterrows():
            query = row["natural_language_query"]
            filename = row["data_files"]
            dataset_description = row["data_description"]
            
            # Find the dataset file in 'data' directory
            dataset_path = None
            search_dirs = [
                os.path.join(ROOT, "data"),
                os.path.join(ROOT, "data", "all_data"),
                os.path.join(ROOT, "data", "synthetic_data")
            ]
            
            # Extract just the filename just in case it has path info
            filename = os.path.basename(filename)
            
            for d in search_dirs:
                potential_path = os.path.join(d, filename)
                if os.path.exists(potential_path):
                    dataset_path = potential_path
                    break
            
            if not dataset_path:
                print(f"Skipping row {idx}: Dataset file {filename} not found.")
                continue

            print(f"\n[Query {idx+1}] File: {filename}")
            print(f"Query: {query}")
            
            try:
                agent = CausalAgent(
                    dataset_path=dataset_path,
                    dataset_description=dataset_description,
                )
                output = agent.run_analysis(
                    query=query,
                )
                
                # Basic correctness check
                self.assertIsNotNone(output, "Agent returned None output.")
                
                # Collect result
                result_entry = {
                    "query": query,
                    "filename": filename,
                    "llm_output": output
                }
                results_log.append(result_entry)
                
            except Exception as e:
                print(f"Error running analysis for row {idx}: {e}")
                results_log.append({
                    "query": query,
                    "filename": filename,
                    "error": str(e)
                })

        # Save all results to JSON
        with open(output_file, "w") as f:
            json.dump(results_log, f, indent=2, default=str)
        
        print(f"\n--- All results logged to {output_file} ---")


if __name__ == "__main__":
    unittest.main()
