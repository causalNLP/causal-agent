# cais/components/dataset_cleaner.py

import os, io, json, traceback, contextlib, sys
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from langchain_core.messages import SystemMessage, HumanMessage
from cais.config import get_llm_client  # returns a LangChain chat model

PLANNER_SYSTEM = """You are “CausalPrep-Planner”, a senior data & methods engineer.

Goal: From (dataset_profile, causal_method, causal_query, variables), produce a SINGLE JSON
Transformation Spec that makes the dataframe METHOD-READY while avoiding target leakage.
Do not pattern-match to any given examples. Generalize beyond them.

Principles:
- Minimal, method-aware edits only. Prefer light-touch transforms.
- Idempotent: re-running produces no further net change.
- No outcome-conditioned decisions. Never peek at y to pick transforms.
- Keep original columns unless explicitly dropped.
- Bounded one-hot: cap category levels via `max_levels`.
- Short justifications (≤ 15 words each). No chain-of-thought.
- Return ONLY valid JSON. No prose.

Allowed operations (non-exhaustive, pick only what’s needed):
- row_filters: {valid_range, drop_missing_any, drop_missing_all, keep_values, drop_values}
- column_ops: {winsorize, clip, log1p, standardize, robust_scale, one_hot, fillna, astype, parse_datetime, sort}
- method_constructs:
  - OLS/ATE: standardize selected X; encode categoricals used in design
  - DiD/Event: build treated, post, event_time; ensure panel keys; align windows
  - RDD: construct treat via cutoff; center running var; add poly terms; select bandwidth
  - IV/2SLS: specify endog/exog/instruments sets; basic sanity flags
  - Matching/IPW: ensure binary T; prepare PS inputs; (weights spec only, not model fitting)
  - FE/Panel: unit_fe/time_fe flags or categorical codes
- weights: optional IPW spec (only the spec — no fitting here)
- fe: optional flags (unit_fe, time_fe)
- assumption_prechecks (metadata-only): missing-by-role, sparse-instrument, panel-completeness, pre-period-exists, bandwidth-rationale
- Do not include your response in ```json```

JSON schema to follow:
{
  "method": "<DiD|RDD|IV|OLS|...>",
  "roles": {
    "treatment": "<col?>",
    "outcome": "<col?>",
    "covariates": ["..."],
    "unit_id": "<col?>",
    "time_var": "<col?>",
    "running_var": {"name": "<col?>", "cutoff": <number?>},
    "instrument_var": ["..."],
    "mediator": ["..."],
    "cluster_id": "<col?>"
  },
  "row_filters": [ { "type": "...", "...": "..." , "why": "..." } ],
  "column_ops":  [ { "op": "...", "cols": ["..."], "...": "...", "why": "..." } ],
  "method_constructs": [ { "method": "...", "make": ["..."], "...": "...", "why": "..." } ],
  "weights": { "type": "ipw", "propensity_spec": "binlogit(covs)", "trim": [0.05,0.95] }?,
  "fe": { "unit_fe": true/false, "time_fe": true/false }?,
  "assumption_prechecks": ["..."],
  "notes": "1-3 sentences, general & method-aware, no example bias."
}
"""
CODEGEN_SYSTEM  = """You are “CausalPrep-Codegen”.

Input: (dataset_path, Transformation Spec JSON).
Output: A SINGLE Python script as text that:
- imports only: json, os, pandas as pd, numpy as np
- loads the dataset from dataset_path (infer CSV/Parquet by extension)
- applies ONLY what the Spec asks for (row_filters, column_ops, method_constructs, etc.)
- keeps all original columns unless Spec explicitly drops them
- creates any new columns explicitly; suffix where needed; no silent overwrite
- produces a dataframe named clean_df
- writes:
  - cleaned_df.csv (same directory as dataset_path)
  - preprocessing_manifest.json (the Spec actually executed)
  - derived_columns.json (list of new columns with one-line descriptions)
- prints a concise, human-readable summary report to stdout
- is idempotent, no randomness, no external I/O, no package installs
- NO chain-of-thought. Return ONLY the code block (no backticks).
- Do not include your response in ```json```

"""
FIXER_SYSTEM    = """You are “CausalPrep-Fixer”. The previous code failed to run.

Given: (failing_code, python_traceback, dataset_profile, head_preview, transformation_spec).
Task: Return a FULL corrected script that meets the Codegen contract exactly.

Rules:
- Keep the same outputs and filenames.
- Do not change Spec semantics unless necessary for compatibility (e.g., dtype coercions).
- Prefer safe coercions (to_datetime, to_numeric(errors="coerce"), astype with try/except).
- No new external libraries; only json, os, pandas, numpy.
- Idempotent & deterministic.
- Return ONLY the full corrected code (no backticks, no explanations).
- Do not include your response in ```json```

"""

# ---------- helpers ----------

def _profile_dataset(dataset_path: str, sample_rows: int = 5) -> Dict[str, Any]:
    def _read(dp):
        ext = os.path.splitext(dp)[1].lower()
        if ext in [".parquet", ".pq"]:
            return pd.read_parquet(dp)
        return pd.read_csv(dp)
    df = _read(dataset_path)

    prof = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": [],
        "numeric_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "missing_by_col": {}
    }
    for c in df.columns:
        s = df[c]
        colinfo = {
            "name": c,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "missing_pct": float(s.isna().mean())
        }
        prof["columns"].append(colinfo)
        prof["missing_by_col"][c] = {
            "count": int(s.isna().sum()),
            "pct": float(s.isna().mean())
        }
        if pd.api.types.is_numeric_dtype(s):
            prof["numeric_columns"].append(c)
        elif pd.api.types.is_datetime64_any_dtype(s):
            prof["datetime_columns"].append(c)
        else:
            prof["categorical_columns"].append(c)

    head_rows = df.head(sample_rows)
    head_preview = {
        "columns": list(head_rows.columns),
        "rows": head_rows.astype(object).where(pd.notna(head_rows), None).to_dict(orient="records")
    }
    return {"profile": prof, "head_preview": head_preview}


def _invoke_json(llm, system_prompt: str, human_payload: Dict[str, Any]) -> Dict[str, Any]:
    msgs = [SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(human_payload, ensure_ascii=False))]
    print(msgs)
    resp = llm.invoke(msgs)
    text = getattr(resp, "content", str(resp))
    print(text)
    # force JSON extraction (model already instructed to return only JSON)
    return json.loads(text)


def _invoke_text(llm, system_prompt: str, human_payload: Dict[str, Any]) -> str:
    msgs = [SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(human_payload, ensure_ascii=False))]
    resp = llm.invoke(msgs)
    return getattr(resp, "content", str(resp))


def _run_script_text(script: str, dataset_path: str) -> Tuple[str, str]:
    """
    Executes user code in a controlled globals dict.
    Captures stdout/stderr. No network or file sandboxing is applied here by default.
    """
    gbls = {
        "__name__": "__main__",
        "__file__": "<generated>",
    }
    lcls = {}
    stdout_io, stderr_io = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
            # Provide dataset_path as a global the script can read (it should anyway use the passed JSON)
            gbls["__DATASET_PATH__"] = dataset_path
            exec(script, gbls, lcls)
    except Exception as e:
        tb = traceback.format_exc()
        stderr_io.write("\n" + tb)
    return stdout_io.getvalue(), stderr_io.getvalue()


# ---------- main pipeline blocks ----------

def _plan_transformation_spec(llm, dataset_path: str, causal_method: str, causal_query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    prof = _profile_dataset(dataset_path)

    human = {
        "dataset_path": dataset_path,
        "dataset_profile": prof["profile"],
        "causal_method": causal_method,
        "causal_query": causal_query or "",
        "variables": variables
    }
    print(human)
    spec = _invoke_json(llm, PLANNER_SYSTEM, human)
    print(spec)
    # Persist the preview so Fixer can use it later
    spec["_runtime_previews"] = prof  # used only internally, not written to disk by LLM
    return spec


def _generate_code(llm, dataset_path: str, spec: Dict[str, Any]) -> str:
    human = {"dataset_path": dataset_path, "transformation_spec": spec}
    code = _invoke_text(llm, CODEGEN_SYSTEM, human)
    return code


def _self_repair(llm, dataset_path: str, spec: Dict[str, Any], failing_code: str, python_traceback: str) -> str:
    previews = spec.get("_runtime_previews", {})
    human = {
        "dataset_profile": previews.get("profile", {}),
        "head_preview": previews.get("head_preview", {}),
        "transformation_spec": spec,
        "failing_code": failing_code,
        "python_traceback": python_traceback
    }
    fixed = _invoke_text(llm, FIXER_SYSTEM, human)
    return fixed


# ---------- public entry (keep signature & outputs stable) ----------

def run_cleaning_stage(dataset_path: str,
                       variables: Dict[str, Any],
                       dataset_description: Optional[str] = None,
                       original_query: Optional[str] = None,
                       causal_method: Optional[str] = None,
                       max_repair_attempts: int = 2) -> Dict[str, Any]:
    """
    Returns (unchanged keys):
      - cleaned_dataset_path
      - cleaning_report_md
      - generated_code
      - stdout
      - stderr
    """
    llm = get_llm_client()

    # 1) PLAN
    
    method = causal_method or variables.get("method") or ""
    spec = _plan_transformation_spec(llm, dataset_path, method, original_query or "", variables)
    print(spec)

    # 2) CODEGEN
    code = _generate_code(llm, dataset_path, spec)
    print(code)

    # 3) EXECUTE
    stdout_all, stderr_all = _run_script_text(code, dataset_path)

    # 4) SELF-REPAIR (bounded)
    attempts = 0
    while attempts < max_repair_attempts and (("Traceback" in stderr_all) or ("Error" in stderr_all)):
        attempts += 1
        patched = _self_repair(llm, dataset_path, spec, code, stderr_all)
        code = patched
        out, err = _run_script_text(code, dataset_path)
        stdout_all += f"\n\n[repair_attempt_{attempts}_stdout]\n" + out
        stderr_all += f"\n\n[repair_attempt_{attempts}_stderr]\n" + err

    # Determine cleaned path (as required by contract)
    cleaned_path = os.path.join(os.path.dirname(os.path.abspath(dataset_path)) or ".", "cleaned_df.csv")
    report = []
    report.append(f"Method: {method}")
    report.append(f"Causal Query: {original_query or ''}")
    report.append(f"Rows/Cols before: see planner profile in memory.")
    report.append(f"Artifacts expected: cleaned_df.csv, preprocessing_manifest.json, derived_columns.json")
    if ("Traceback" in stderr_all) or ("Error" in stderr_all):
        report.append("\n⚠️ LLM pipeline produced errors. Check stderr; artifacts may be missing or partial.")

    return {
        "cleaned_dataset_path": cleaned_path,
        "cleaning_report_md": "\n".join(report),
        "generated_code": code,
        "stdout": stdout_all,
        "stderr": stderr_all
    }
