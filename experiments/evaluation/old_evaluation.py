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
    "query",
    "method",
    "effect",
    "sd",
    "treatment",
    "outcome",
    "covariates",
]

SPLITS = ["real", "qr", "synthetic"]

# =====================================================
# Utilities
# =====================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def normalize_text(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).lower().strip()


def pred_contains(pred, keywords):
    p = normalize_text(pred)
    return any(k in p for k in keywords)


def match(a, b):
    if a is None or b is None:
        return False
    if isinstance(a, float) and pd.isna(a):
        return False
    if isinstance(b, float) and pd.isna(b):
        return False

    def to_list(x):
        if isinstance(x, (list, tuple, set)):
            return list(x)
        return [x]

    for ai in to_list(a):
        for bi in to_list(b):
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


def coerce_scalar_effect(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if isinstance(x, dict):
        for v in x.values():
            val = coerce_scalar_effect(v)
            if not np.isnan(val):
                return val
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

# =====================================================
# Robust loader
# =====================================================

def load_outputs_auto(path):
    fname = os.path.basename(path).lower()
    is_cais = fname.startswith("cais_")

    with open(path, "r") as f:
        text = f.read().strip()

    # =====================================================
    # CAIS: stream of JSON dicts
    # =====================================================
    if is_cais:
        outputs = []

        decoder = json.JSONDecoder()
        idx = 0
        n = len(text)

        while idx < n:
            # Skip whitespace
            while idx < n and text[idx].isspace():
                idx += 1
            if idx >= n:
                break

            try:
                obj, next_idx = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                # Non-JSON garbage at the end (e.g. "]%")
                break

            idx = next_idx

            if not isinstance(obj, dict):
                continue

            for k, v in obj.items():
                if not str(k).isdigit():
                    continue
                if not isinstance(v, dict):
                    continue

                outputs.append({
                    "_row_id": int(k),
                    **v,
                })

        if not outputs:
            raise ValueError(f"CAIS parsed 0 rows: {path}")

        return outputs

    # =====================================================
    # Baselines (unchanged)
    # =====================================================
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    outputs = []
    bad = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            outputs.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1

    if outputs:
        if bad > 0:
            print(f"[WARN] {os.path.basename(path)}: skipped {bad} non-JSON lines")
        return outputs

    raise ValueError(f"Could not parse any JSON from {path}")

# =====================================================
# Prediction extraction
# =====================================================
def get_result_block(entry):
    """
    Unified extractor for CAIS + baselines.

    Returns:
      - dict: prediction block
      - None: failed / missing prediction
    """
    if not isinstance(entry, dict):
        return None

    # Baselines: result.final_result
    r = entry.get("result")
    if isinstance(r, dict):
        fr = r.get("final_result")
        if isinstance(fr, dict):
            return fr

    # CAIS (and some baselines): final_result
    fr = entry.get("final_result")
    if isinstance(fr, dict):
        return fr

    return None

def extract_predictions(entries):
    rows = []

    for i, e in enumerate(entries):
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
            "query": e.get("query"),
            "method": pick("method", "Method"),
            "effect": pick("causal_effect", "effect", "estimate"),
            "sd": pick("standard_deviation", "std_error", "stderr"),
            "treatment": pick("treatment_variable", "treatment"),
            "outcome": pick("outcome_variable", "outcome"),
            "covariates": pick("covariates", "control_variables"),
        })

    df = pd.DataFrame(rows, columns=PRED_COLS)

    # Ensure _row_id exists
    if "_row_id" not in df.columns:
        df["_row_id"] = np.nan

    # Fill missing row_ids with positional index
    mask = df["_row_id"].isna()
    df.loc[mask, "_row_id"] = np.arange(len(df))[mask]

    # Coerce to integer
    df["_row_id"] = pd.to_numeric(df["_row_id"], errors="coerce").astype("Int64")

    return df

# =====================================================
# Ground-truth–centric method matching
# =====================================================

def match_method_samuel(gt_method, pred_method, split):
    gt = normalize_text(gt_method)
    pred = normalize_text(pred_method)

    # -------- REAL --------
    if split == "real":

        if "frontdoor" in gt:
            return "frontdoor" in pred

        if "ols" in gt or "linear" in gt:
            return pred_contains(pred, [
                "ols", "linear", "regression",
                "matching", "propensity score", "weight"
            ])

        if "matching" in gt:
            return pred_contains(pred, [
                "matching", "propensity score", "weight"
            ])

        if "weight" in gt:
            return pred_contains(pred, [
                "weight", "propensity score", "matching"
            ])

        if "glm" in gt or "logistic" in gt:
            return pred_contains(pred, [
                "glm", "logistic"
            ])

        if "iv" in gt:
            return pred_contains(pred, [
                "iv", "instrument", "2sls", "two-stage"
            ])

        if "did" in gt or "diff-in-diff" in gt:
            return pred_contains(pred, [
                "did", "difference-in-differences", "diff"
            ])

        if "rdd" in gt or "design" in gt:
            return pred_contains(pred, [
                "rdd", "discontinuity", 'design'
            ])

        return False

    # -------- QR --------
    if split == "qr":

        if "frontdoor" in gt:
            return "frontdoor" in pred

        if "linear" in gt or "ols" in gt:
            return pred_contains(pred, [
                "ols", "linear", "regression",
                "matching", "propensity score", "weight"
            ])

        if any(k in gt for k in ["weighting", "matching", "psm", "weigh"]):
            return pred_contains(pred, [
                "matching", "propensity", "weight"
            ])

        if "iv" in gt:
            return pred_contains(pred, [
                "iv", "instrument", "2sls"
            ])

        if "did" in gt:
            return pred_contains(pred, [
                "did", "difference-in-differences", 'diff'
            ])

        if "rdd" in gt:
            return pred_contains(pred, [
                "rdd", "discontinuity"
            ])

        return False

    # -------- SYNTHETIC --------
    if split == "synthetic":

        if "frontdoor" in gt:
            return "frontdoor" in pred

        if "rct" in gt:
            return pred_contains(pred, [
                "linear", "regression",
                "matching", "weight",
                "did"
            ])

        if "observational" in gt:
            return pred_contains(pred, [
                "linear", "regression",
                "matching", "weight"
            ])

        if "iv" in gt:
            return pred_contains(pred, [
                "iv", "instrument", "2sls"
            ])

        if "did" in gt:
            return pred_contains(pred, [
                "did", "difference-in-differences"
            ])

        if "rdd" in gt:
            return pred_contains(pred, [
                "rdd", "discontinuity", 'design'
            ])

        return False

    return False

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

def match_method_sawal(gt_method, pred_method, split):
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

# =====================================================
# Evaluation
# =====================================================

def evaluate(source, outputs, split, agent, args=None):
    N = len(source)

    # Build query → row index map (normalized)
    query_to_idx = None
    if "query" in source.columns:
        query_to_idx = {
            normalize_text(q): i
            for i, q in enumerate(source["query"])
        }

    pred = extract_predictions(outputs)

    aligned = pd.DataFrame(index=range(N), columns=PRED_COLS)

    # -----------------------------------------------------
    # Alignment logic
    # -----------------------------------------------------

    if agent == "cais":
        matched = 0

        # 1) try query alignment
        if query_to_idx is not None and "query" in pred.columns:
            for _, r in pred.iterrows():
                q = normalize_text(r.get("query"))
                if not q:
                    continue
                i = query_to_idx.get(q)
                if i is not None:
                    aligned.loc[i] = r
                    matched += 1

        print(f"[INFO] CAIS query-aligned {matched}/{len(source)} rows")

        # 2) fallback to _row_id alignment for remaining
        if matched < 0.5 * N and pred["_row_id"].notna().any():
            filled = 0
            for _, r in pred.dropna(subset=["_row_id"]).iterrows():
                i = int(r["_row_id"])
                if 0 <= i < N and pd.isna(aligned.loc[i, "method"]):
                    aligned.loc[i] = r
                    filled += 1
            print(f"[INFO] CAIS row_id-filled {filled} additional rows")

    # Fallback (QR split): positional alignment
    else:
        for i in range(min(len(pred), N)):
            aligned.loc[i] = pred.loc[i]

    cov_col = "covariates" if "covariates" in source.columns else "control_variables"

    df = pd.DataFrame({
        "true_effect": source["answer"],
        "pred_effect": aligned["effect"].apply(coerce_scalar_effect),
        "pred_sd": pd.to_numeric(aligned["sd"], errors="coerce"),
        "ref_method": source["method"],
        "pred_method": aligned["method"],
        "ref_treatment": source["treatment"],
        "pred_treatment": aligned["treatment"],
        "ref_outcome": source["outcome"],
        "pred_outcome": aligned["outcome"],
        "ref_covariates": source[cov_col],
        "pred_covariates": aligned["covariates"],
    })
    match_method = {
        "samuel": match_method_samuel,
        "sawal": match_method_sawal,
    }[args.match_method]

    df["method_correct"] = [
        match_method(gt, pred, split)
        for gt, pred in zip(df["ref_method"], df["pred_method"])
    ]

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
    r = np.minimum(
        np.abs(df["pred_effect"] - df["true_effect"]) /
        np.maximum(np.abs(df["true_effect"]), eps),
        1.0,
    )

    df["softacc_i"] = 0.0
    mask = df["method_correct"] & df["scalar_exists"]
    df.loc[mask, "softacc_i"] = (1 - r[mask]).clip(0, 1)

    # df["softacc_i"] = 0.0
    # 
    # mask = (
        # df["method_correct"]
        # & df["scalar_exists"]
        # & (r <= 0.1)
    # )
    # df.loc[mask, "softacc_i"] = 1.0

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

# =====================================================
# Filename parsing
# =====================================================

def parse_result_filename(fname):
    base = fname.replace(".json", "").lower()
    parts = base.split("_")

    agent = parts[0]

    if "qr" in base:
        split = "qr"
    elif "synthetic" in base:
        split = "synthetic"
    else:
        split = "real"

    model = "_".join(p for p in parts[1:] if p not in ["real", "qrdata", "synthetic"])
    return agent, model, split

# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--match-method", choices=["samuel", "sawal"], default="sawal")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "row_level"))
    ensure_dir(os.path.join(args.out_dir, "summaries"))

    # Load sources
    sources = {}
    for split in SPLITS:
        path = os.path.join(args.data_dir, f"{split}_info.csv")
        df = pd.read_csv(path, encoding="latin1")
        df["answer"] = (
            df["answer"]
            .astype(str)
            .str.replace("−", "-", regex=False)
            .astype(float)
        )
        sources[split] = df
        print(f"[INFO] loaded {split} split with {len(df)} rows")

    rows = []

    for fname in os.listdir(args.results_dir):
        if not fname.endswith(".json"):
            continue

        agent, model, split = parse_result_filename(fname)
        print(f"[INFO] split={split} | agent={agent} | model={model}")

        outputs = load_outputs_auto(os.path.join(args.results_dir, fname))
        summary, df = evaluate(sources[split], outputs, split, agent, args=args)

        rows.append({
            "agent": agent,
            "model": model,
            "split": split,
            **summary,
        })

        split_dir = os.path.join(args.out_dir, "row_level", split)
        ensure_dir(split_dir)

        out_csv = os.path.join(
            split_dir,
            f"{agent}_{model}.csv".replace("/", "_"),
        )
        df.to_csv(out_csv, index=True)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        os.path.join(args.out_dir, "summaries", "per_run.csv"),
        index=False,
    )

    agg = (
        summary_df
        .groupby(["split", "agent", "model"])
        .mean(numeric_only=True)
        .reset_index()
    )

    agg.to_csv(
        os.path.join(args.out_dir, "summaries", "aggregated.csv"),
        index=False,
    )

    print("\n[OK] Aggregated results:")
    print(agg.sort_values(["split", "model", "softacc"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()