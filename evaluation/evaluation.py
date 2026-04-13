import os
import json
import argparse
import numpy as np
import pandas as pd


def standardize_method_name(method):
    """
    Standardize method names to a coarse causal family.
    """
    
    if method is None or not isinstance(method, str):
        return np.nan

    m = method.lower().strip()

    # Explicit failures
    if any(x in m for x in ["null", "na", "n/a", "none"]):
        return np.nan

    # Frontdoor FIRST (important)
    if "frontdoor" in m or "front door" in m:
        return "fd"

    # IV
    if any(x in m for x in ["instrument", "encouragement", "2sls", "iv"]):
        return "iv"

    # RDD
    if any(x in m for x in ["discontinuity", "rdd", "fuzzy"]):
        return "rdd"

    # GLMs
    if any(x in m for x in ["logistic", "probit", "logit", "glm"]):
        return "glm"

    # Observational adjustment
    if any(x in m for x in ["weighting", "ipw", "propensity", "matching", "observational"]):
        return "observational"

    # OLS / means / RCT-style
    if any(x in m for x in ["linear", "means", "ordinary", "ols", "wls", "rct"]):
        return "ols"

    # DiD / panels
    if any(x in m for x in ["difference", "did", "fixed effects", "panel"]):
        return "did"

    return "other"


def match_method(gt_method, pred_method, split):
    """
    Method match based on standardized causal families.

    Returns:
        bool
    """
    gt = standardize_method_name(gt_method)
    pr = standardize_method_name(pred_method)

    # If either side failed to produce a method → incorrect
    if pd.isna(gt) or pd.isna(pr):
        return False

    # Exact family match
    if gt == pr:
        return True

    # -------------------------------
    # Controlled relaxations by split
    # -------------------------------

    # === REAL ===
    if split == "real":
        # GLM is acceptable when GT is OLS (common misuse)
        if gt == "ols" and pr == "glm":
            return True

        # Observational ≠ OLS (do NOT relax)
        return False

    # === QR ===
    if split == "qr":
        # QR ground truth is loose by design
        if gt in {"ols", "observational"} and pr in {"ols", "observational"}:
            return True
        return False

    # === SYNTHETIC ===
    if split == "synthetic":
        # RCT data: OLS is acceptable
        if gt == "rct" and pr == "ols":
            return True

        # Observational ≈ OLS in synthetic benchmarks
        if gt == "observational" and pr == "ols":
            return True

        return False

    return False


def evaluate_result(fname):

    with open(fname, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    if 'qr' in fname:
        split = 'qr'
    elif 'real' in fname:
        split = 'real'
    elif 'synthetic' in fname:
        split = 'synthetic'
    else:
        raise ValueError(f"Unknown split for file: {fname}")

    correct_method = 0
    for i in range(0, len(data)):

        try:
            entry = next(iter(data[i].values()))
            gt_method = entry['method']
            pred_method = entry['final_result']['method']

            # gt_answer = entry['answer']
            # pred_answer = entry['final_result']['causal_effect']

            ans = match_method(gt_method, pred_method, split=split)

            if ans == True:
                correct_method += 1
    
        except Exception as e:
            print(f"[ERROR] Failed to evaluate entry {i}: {e}")
            continue
    
    # Print result name and method accuracy
    print("Method Selection Accuracy: ", (correct_method / len(data)) * 100)
    print('--------')

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load json files from args.out_dir
    for fname in os.listdir(args.results_dir):
        if not fname.endswith(".json"):
            print(f"[WARNING] Skipping non-json file: {fname}")
        
        print(f"[INFO] Evaluating file: {fname}")
        evaluate_result(os.path.join(args.results_dir, fname))


if __name__ == "__main__":
    main()
