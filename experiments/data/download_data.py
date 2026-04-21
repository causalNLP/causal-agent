# experiments/download_data.py
from huggingface_hub import snapshot_download
import os

def download_causcibench(output_dir: str = "experiments/data/causcibench"):
    os.makedirs(output_dir, exist_ok=True)
    snapshot_download(
        repo_id="causal-nlp/causcibench",
        repo_type="dataset",
        local_dir=output_dir,
    )
    print(f"Dataset downloaded to {output_dir}")

if __name__ == "__main__":
    download_causcibench()