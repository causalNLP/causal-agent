import os
import json
import argparse
import numpy as np
import pandas as pd

# =====================================================
# Constants
# =====================================================

PRED_COLS = [
    "_row_id",
    "method",
    "effect",
    "sd",
    "treatment",
    "outcome",
    "covariates",
]

# =====================================================
# Utilities
# =====================================================

def normalize_method(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).lower()
    if "diff" in s:
        return "did"
    if "instrument" in s or "iv" in s:
        return "iv"
    if "rdd" in s or "discontinuity" in s:
        return "rdd"
    if "match" in s:
        return "matching"
    if "regression" in s or "ols" in s:
        return "ols"
    return s.strip()


def canonical_method(method):
    if method is None or (isinstance(method, float) and np.isnan(method)):
        return None
    s = str(method).lower()
    if any(k in s for k in ["matching", "propensity", "ipw", "psm"]):
        return "matching"
    if any(k in s for k in ["ols", "regression", "linear"]):
        return "ols"
    if "did" in s:
        return "did"
    if "iv" in s or "instrument" in s:
        return "iv"
    if "rdd" in s:
        return "rdd"
    return s


def match(a, b):
    """
    Robust string / list matching for treatment & outcome names.
    Returns False if either side is missing.
    """

    # handle missing
    if a is None or b is None:
        return False
    if isinstance(a, float) and pd.isna(a):
        return False
    if isinstance(b, float) and pd.isna(b):
        return False

    # normalize to list
    def to_list(x):
        if isinstance(x, (list, tuple, set)):
            return list(x)
        return [x]

    a_list = to_list(a)
    b_list = to_list(b)

    for ai in a_list:
        for bi in b_list:
            if ai is None or bi is None:
                continue
            try:
                ai_s = str(ai).lower().strip()
                bi_s = str(bi).lower().strip()
            except Exception:
                continue

            if ai_s == bi_s or ai_s in bi_s or bi_s in ai_s:
                return True

    return False

def parse_covariates(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return set()
    if isinstance(x, (list, tuple, set)):
        return {str(v).strip().lower() for v in x if str(v).strip()}
    return {v.strip().lower() for v in str(x).split(",") if v.strip()}


# =====================================================
# Safe result extraction
# =====================================================

def get_result_block(entry):
    """
    Priority:
      1) entry["result"]["final_result"]
      2) entry["final_result"]
      3) entry["result"]
      4) None
    """
    if not isinstance(entry, dict):
        return None

    r = entry.get("result")
    if isinstance(r, dict):
        fr = r.get("final_result")
        if isinstance(fr, dict):
            return fr

    fr = entry.get("final_result")
    if isinstance(fr, dict):
        return fr

    if isinstance(r, dict):
        return r

    return None


def extract_predictions(entries):
    rows = []

    for e in entries:
        r = get_result_block(e)

        def pick(*keys):
            if not isinstance(r, dict):
                return np.nan
            for k in keys:
                if k in r and r[k] is not None:
                    return r[k]
            return np.nan

        rows.append({
            "_row_id": e.get("_row_id", np.nan),
            "method": pick("method", "Method"),
            "effect": pick("causal_effect", "effect", "estimate"),
            "sd": pick("standard_deviation", "std_error", "stderr"),
            "treatment": pick("treatment_variable", "treatment"),
            "outcome": pick("outcome_variable", "outcome"),
            "covariates": pick("covariates", "control_variables"),
        })

    return pd.DataFrame(rows, columns=PRED_COLS)


# =====================================================
# Robust loader
# =====================================================

def load_outputs_auto(path):
    with open(path, "r") as f:
        text = f.read().strip()

    outputs = []
    try:
        with open(path, "r") as f:
            data = json.load(f)

        outputs = []
        if (
            isinstance(data, list)
            and len(data) > 0
            and isinstance(data[0], dict)
            and "query" in data[0]
            and "answer" in data[0]
        ):
            return data
    except Exception:
        pass

    # JSONL or CAIS dict
    if text.startswith("{"):
        lines = text.splitlines()

        for line in lines:
            obj = json.loads(line)

            # CAIS-style numeric keys
            if isinstance(obj, dict) and any(str(k).isdigit() for k in obj):
                for k, v in obj.items():
                    if not str(k).isdigit():
                        continue
                    entry = {"_row_id": int(k)}
                    if isinstance(v, dict):
                        entry.update(v)
                    else:
                        # v might be a string error — keep it
                        entry["final_result"] = v
                    outputs.append(entry)
            else:
                outputs.append(obj)

        return outputs

    raise ValueError(f"Unrecognized JSON format: {path}")


def discover_files(results_dir):
    out = []
    for f in os.listdir(results_dir):
        if f.endswith(".json"):
            parts = f.replace(".json", "").split("_")
            if len(parts) >= 3:
                out.append({
                    "method_file": parts[0],
                    "model": "_".join(parts[1:-1]),
                    "path": os.path.join(results_dir, f),
                })
    return out


def coerce_scalar_effect(x):
    """
    Extract a numeric scalar causal effect from messy model outputs.

    Handles:
      - float / int
      - dicts like {"Hawthorne": 0.02}
      - nested dicts
      - strings convertible to float

    Returns np.nan if not possible.
    """
    if x is None:
        return np.nan

    # already numeric
    if isinstance(x, (int, float, np.number)):
        return float(x)

    # dict: extract numeric values regardless of key
    if isinstance(x, dict):
        numeric_vals = []
        for v in x.values():
            val = coerce_scalar_effect(v)
            if not np.isnan(val):
                numeric_vals.append(val)

        if len(numeric_vals) == 1:
            return numeric_vals[0]
        if len(numeric_vals) > 1:
            return numeric_vals[0]  # deterministic fallback

        return np.nan

    # string case
    try:
        return float(x)
    except Exception:
        return np.nan

# =====================================================
# Evaluation
# =====================================================

def evaluate(source, outputs):
    N = len(source)

    pred = extract_predictions(outputs)

    aligned = pd.DataFrame(index=range(N), columns=PRED_COLS, dtype=object)

    if pred["_row_id"].notna().any():
        for _, r in pred.dropna(subset=["_row_id"]).iterrows():
            idx = int(r["_row_id"])
            if 0 <= idx < N:
                aligned.loc[idx] = r
    else:
        for i in range(min(len(pred), N)):
            for c in PRED_COLS:
                aligned.at[i, c] = pred.at[i, c]

    cov_col = "covariates" if "covariates" in source.columns else "control_variables"

    df = pd.DataFrame({
        "true_effect": source["answer"],
        "pred_effect": aligned["effect"],
        "pred_sd": aligned["sd"],
        "ref_method": source["method"].apply(normalize_method),
        "pred_method": aligned["method"].apply(normalize_method),
        "ref_treatment": source["treatment"],
        "pred_treatment": aligned["treatment"],
        "ref_outcome": source["outcome"],
        "pred_outcome": aligned["outcome"],
        "ref_covariates": source[cov_col],
        "pred_covariates": aligned["covariates"],
    })
    df["pred_effect"] = df["pred_effect"].apply(coerce_scalar_effect).astype(float)
    df["true_effect"] = pd.to_numeric(df["true_effect"], errors="coerce")
    df["pred_sd"] = pd.to_numeric(df["pred_sd"], errors="coerce")

    df["method_correct"] = (
        df["ref_method"].apply(canonical_method)
        == df["pred_method"].apply(canonical_method)
    )

    df["treatment_correct"] = [
        match(a, b) for a, b in zip(df["ref_treatment"], df["pred_treatment"])
    ]
    df["outcome_correct"] = [
        match(a, b) for a, b in zip(df["ref_outcome"], df["pred_outcome"])
    ]

    df["cov_exact"] = [
        parse_covariates(a) == parse_covariates(b)
        for a, b in zip(df["ref_covariates"], df["pred_covariates"])
    ]

    df["scalar_exists"] = df["pred_effect"].notna()

    eps = 1e-8
    r_i = np.minimum(
        np.abs(df["pred_effect"] - df["true_effect"]) /
        np.maximum(np.abs(df["true_effect"]), eps),
        1.0,
    )

    df["softacc_i"] = 0.0
    df["softacc_i"] = df["softacc_i"].astype(float)
    mask = df["method_correct"] & df["scalar_exists"]
    df.loc[mask, "softacc_i"] = (1 - r_i[mask]).clip(0, 1)

    df["hardacc_i"] = (
        df["method_correct"]
        & df["treatment_correct"]
        & df["outcome_correct"]
        & df["cov_exact"]
    ).astype(float)

    summary = {
        "method_acc": 100 * df["method_correct"].mean(),
        "softacc": 100 * df["softacc_i"].mean(),
        "hardacc": 100 * df["hardacc_i"].mean(),
        "coverage": 100 * df["scalar_exists"].mean(),
        "n": N,
    }

    return summary, df

def dump_row_level_csv(df, out_path):
    """
    Writes a full row-level diagnostic CSV for manual inspection.
    """
    cols = [
        # ground truth
        "true_effect",
        "ref_method",
        "ref_treatment",
        "ref_outcome",
        "ref_covariates",

        # predictions
        "pred_effect",
        "pred_sd",
        "pred_method",
        "pred_treatment",
        "pred_outcome",
        "pred_covariates",

        # evaluation flags
        "method_correct",
        "treatment_correct",
        "outcome_correct",
        "cov_exact",
        "scalar_exists",
        "softacc_i",
        "hardacc_i",
    ]

    existing = [c for c in cols if c in df.columns]
    df[existing].to_csv(out_path, index=True)
    print(f"[INFO] wrote row-level diagnostics to {out_path}")


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        source = pd.read_csv(args.csv)
    except UnicodeDecodeError:
        source = pd.read_csv(args.csv, encoding="latin1")

    source["answer"] = (
        source["answer"]
        .astype(str)
        .str.replace("−", "-", regex=False)
        .astype(float)
    )

    rows = []
    for info in discover_files(args.results_dir):
        outputs = load_outputs_auto(info["path"])
        summ, df = evaluate(source, outputs)

        rows.append({
            "model": info["model"],
            "method_file": info["method_file"],
            **summ,
        })

        # write row-level CSV
        safe_model = info["model"].replace("/", "_")
        safe_method = info["method_file"].replace("/", "_")
        csv_path = f"{args.out}_{safe_model}_{safe_method}_rows.csv"
        dump_row_level_csv(df, csv_path)

    out_df = pd.DataFrame(rows).sort_values(["model", "softacc"])
    out_df.to_csv(args.out, index=False)
    print(out_df.to_string(index=False))
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
