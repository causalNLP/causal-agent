# cais/components/dataset_cleaner.py

import os, io, json, traceback, contextlib, sys, ast
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from langchain_core.messages import SystemMessage, HumanMessage
from cais.config import get_llm_client  # returns a LangChain chat model

PLANNER_SYSTEM = """You are “CausalPrep-Planner”, a senior data engineer and analyst.

Goal: From (dataset_profile, causal_method, causal_query, variables), produce a SINGLE JSON
Transformation Spec that makes the dataframe METHOD-READY while avoiding target leakage.

Principles:
- STRICT: Transform ONLY treatment/outcome columns (or new columns derived solely to define them).
- STRICT: Do NOT transform covariates, instruments, controls, mediators, unit/time IDs, or other variables.
- STRICT: Only apply transformations explicitly requested in the causal_query or required to define treatment/outcome.
- STRICT: If the query names a transformed outcome/treatment (e.g., log, diff, rate), apply exactly that transform.
- Row filters/missing handling only when required to define treatment/outcome (e.g., drop missing y/t).
- For right-censored survival, preserve duration and event columns; never mean-impute duration or event.
- Missing treatment/event/duration rows are reported and excluded when required; covariate imputation must be fold-wise downstream.
- Do not infer a competing-risk event as ordinary right censoring.
- Minimal, method-aware edits only. Prefer light-touch transforms.
- Idempotent: re-running produces no further net change.
- No outcome-conditioned decisions. Never peek at y to pick transforms.
- Keep original columns unless explicitly dropped.
- Bounded one-hot: cap category levels via `max_levels`.
- STRICT: No scaling/normalization at this stage (e.g., standardize, robust_scale, min-max, z-score), even if asked.
- Short justifications (≤ 15 words each). No chain-of-thought.
- Return ONLY valid JSON. No prose.

Allowed operations (non-exhaustive, pick only what’s needed):
- row_filters: {valid_range, drop_missing_any, drop_missing_all, keep_values, drop_values}
- column_ops: {winsorize, clip, log1p, one_hot, fillna, astype, parse_datetime, sort}
- method_constructs:
  - OLS/ATE: no constructs beyond explicit treatment/outcome transforms
  - DiD/Event: build treated/post/event_time indicators as new columns; do not alter source columns
  - RDD: construct treatment via cutoff only; no centering or polynomial terms
  - IV/2SLS: specify endog/exog/instruments sets; no transformations
  - Matching/IPW: ensure binary treatment only; no covariate prep
  - FE/Panel: flags only (unit_fe/time_fe); no recoding
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
    "duration": "<follow-up time column for right-censored survival, or null>",
    "event": "<event indicator 1=event/0=censored, or null>",
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
- NEVER RE-DEFINE `__DATASET_PATH__` or `__CLEANED_PATH__`. They are already provided.
- loads the dataset from the path provided in global variable `__DATASET_PATH__`
- applies ONLY what the Spec asks for (row_filters, column_ops, method_constructs, etc.)
- keeps all original columns unless Spec explicitly drops them
- creates any new columns explicitly; suffix where needed; no silent overwrite
- produces a dataframe named clean_df
- writes:
  - clean_df.csv to the path provided in global variable `__CLEANED_PATH__`
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
    #print(msgs)
    resp = llm.invoke(msgs)
    text = getattr(resp, "content", str(resp))
    #print(text)
    # force JSON extraction (model already instructed to return only JSON)
    return json.loads(text)


def _invoke_text(llm, system_prompt: str, human_payload: Dict[str, Any]) -> str:
    msgs = [SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(human_payload, ensure_ascii=False))]
    resp = llm.invoke(msgs)
    return getattr(resp, "content", str(resp))


_ALLOWED_OPS = {
    "valid_range", "drop_missing_any", "drop_missing_all", "keep_values",
    "drop_values", "winsorize", "clip", "log1p", "one_hot", "fillna",
    "astype", "parse_datetime", "sort",
}
_ALLOWED_IMPORTS = {"json", "os", "pandas", "numpy"}
_FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "system", "popen",
    "run", "Popen", "remove", "unlink", "rmtree",
}


def _build_typed_data_contract(
    profile: Dict[str, Any], variables: Dict[str, Any], spec: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the auditable preprocessing contract consumed downstream."""
    columns = {item["name"]: item for item in profile.get("columns", [])}
    role_keys = {
        "treatment": "treatment_variable", "outcome": "outcome_variable",
        "unit": "group_variable", "time": "time_variable",
        "running_variable": "running_variable", "instrument": "instrument_variable",
        "mediator": "mediator_variable", "duration": "duration_variable",
        "event": "event_variable",
    }
    roles = {}
    for role, key in role_keys.items():
        value = variables.get(key)
        if value:
            roles[role] = value if isinstance(value, list) else [value]
    for role, value in (spec.get("roles") or {}).items():
        if value and role not in roles:
            roles[role] = value if isinstance(value, list) else [value]

    def column_type(name):
        info = columns.get(name, {})
        return {"dtype": info.get("dtype"),
                "missing_pct": info.get("missing_pct"),
                "n_unique": info.get("n_unique")}

    typed_variables = {
        name: {"roles": [role for role, names in roles.items() if name in names],
               "type": column_type(name)}
        for name in columns
    }
    missing = profile.get("missing_by_col", {})
    return {
        "variables": roles,
        "types": typed_variables,
        "design": {"method": variables.get("method") or spec.get("method"),
                    "time_variable": variables.get("time_variable"),
                    "unit_variable": variables.get("group_variable")},
        "missingness": missing,
        "transformations": spec.get("column_ops", []),
        "structural_prerequisites": spec.get("assumption_prechecks", []),
        "provenance": {
            "source": "dataset profile, causal query, variable contract and LLM proposal",
            "generated_spec": True,
            "unresolved_role_ambiguities": [],
        },
    }


def _validate_spec(spec: Dict[str, Any], profile: Dict[str, Any], variables: Dict[str, Any]) -> None:
    """Fail closed on malformed or unsafe transformation specifications."""
    if not isinstance(spec, dict):
        raise ValueError("Preprocessing specification must be a JSON object.")
    available = {item["name"] for item in profile.get("columns", [])}
    roles = spec.get("roles") or {}
    for role, value in roles.items():
        values = value if isinstance(value, list) else [value]
        for column in values:
            if column and column not in available:
                raise ValueError(f"Unknown column '{column}' in preprocessing role '{role}'.")
    protected = {
        variables.get(key) for key in (
            "instrument_variable", "mediator_variable", "group_variable",
            "time_variable", "running_variable", "duration_variable", "event_variable",
        ) if variables.get(key)
    }
    treatment_outcome = {variables.get("treatment_variable"), variables.get("outcome_variable")} - {None}
    for section in ("row_filters", "column_ops", "method_constructs"):
        if not isinstance(spec.get(section, []), list):
            raise ValueError(f"Preprocessing field '{section}' must be a list.")
    for operation in spec.get("column_ops", []):
        if not isinstance(operation, dict) or operation.get("op") not in _ALLOWED_OPS:
            raise ValueError(f"Unsupported preprocessing operation: {operation!r}")
        cols = operation.get("cols", [])
        if isinstance(cols, str):
            cols = [cols]
        if not isinstance(cols, list) or any(column not in available for column in cols):
            raise ValueError(f"Preprocessing operation references an unknown column: {cols!r}")
        if set(cols) & protected and operation.get("op") not in {"parse_datetime", "sort"}:
            raise ValueError("Design columns cannot be transformed during preprocessing.")
        # Treatment and outcome transformations must be explicitly requested
        # by the query; the generated spec alone cannot silently redefine them.
        if set(cols) & treatment_outcome and operation.get("op") in {"fillna", "winsorize", "clip", "one_hot"}:
            raise ValueError("Treatment/outcome missingness and support cannot be altered implicitly.")
        if operation.get("op") == "fillna":
            raise ValueError("Imputation is not enabled as a default preprocessing operation.")
    for construct in spec.get("method_constructs", []):
        if not isinstance(construct, dict):
            raise ValueError("Each method construct must be an object.")
        for name in construct.get("make", []) or []:
            if name in available:
                raise ValueError(f"Method construct would overwrite existing column '{name}'.")


def _validate_generated_code(script: str) -> None:
    """Reject generated code outside the small documented execution surface."""
    tree = ast.parse(script)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".")[0] for alias in node.names]
            if any(name not in _ALLOWED_IMPORTS for name in names):
                raise ValueError("Generated preprocessing code imports a disallowed module.")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                raise ValueError(f"Generated preprocessing code uses a forbidden call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_CALLS:
                raise ValueError(f"Generated preprocessing code uses a forbidden call: {node.func.attr}")


def _run_script_text(script: str, dataset_path: str, cleaned_path: str) -> Tuple[str, str]:
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
        _validate_generated_code(script)
        with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
            # Provide dataset_path as a global the script can read (it should anyway use the passed JSON)
            gbls["__DATASET_PATH__"] = dataset_path
            gbls["__CLEANED_PATH__"] = cleaned_path
            
            # We restore the replacements but use json.dumps for safe quoting on Windows
            # This handles models that hardcode the path despite instructions
            for placeholder in ["cleaned_df.csv", "clean_df.csv", "manifest.json", "derived_columns.json"]:
                if placeholder in script:
                    script = script.replace(f'"{placeholder}"', json.dumps(cleaned_path if "csv" in placeholder else placeholder))
                    script = script.replace(f"'{placeholder}'", json.dumps(cleaned_path if "csv" in placeholder else placeholder))
            
            exec(script, gbls, lcls)
    except Exception as e:
        tb = traceback.format_exc()
        stderr_io.write("\n" + tb)
    return stdout_io.getvalue(), stderr_io.getvalue()


# ---------- main pipeline blocks ----------

def _plan_transformation_spec(llm, dataset_path: str, causal_method: str, causal_query: str, variables: Dict[str, Any], dataset_description: Optional[str]) -> Dict[str, Any]:
    prof = _profile_dataset(dataset_path)

    human = {
        "dataset_path": dataset_path,
        "dataset_profile": prof["profile"],
        "dataset_description": dataset_description or "No dataset description provided",
        "causal_method": causal_method,
        "causal_query": causal_query or "",
        "variables": variables
    }
    #print(human)
    spec = _invoke_json(llm, PLANNER_SYSTEM, human)
    #print(spec)
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
                       max_repair_attempts: int = 2,
                       llm: Optional[Any] = None) -> Dict[str, Any]:
    """
    Returns (unchanged keys):
      - cleaned_dataset_path
      - cleaning_report_md
      - generated_code
      - stdout
      - stderr
      - typed_data_contract
      - typed_data_contract_path
    """
    if llm is None:
        llm = get_llm_client()

    dataset_path = dataset_path.replace("\\", "/")
    base_name = os.path.basename(dataset_path)
    file_stem = os.path.splitext(base_name)[0]
    cleaned_path = os.path.join(os.path.dirname(os.path.abspath(dataset_path)) or ".", f"{file_stem}_cleaned_{os.getpid()}.csv").replace("\\", "/")

    # 1) PLAN
    method = causal_method or variables.get("method") or ""
    spec = _plan_transformation_spec(llm, dataset_path, method, original_query or "", variables, dataset_description)
    profile = spec.get("_runtime_previews", {}).get("profile", {})
    _validate_spec(spec, profile, variables)
    typed_contract = _build_typed_data_contract(profile, variables, spec)
    #print(spec)

    # 2) CODEGEN
    code = _generate_code(llm, dataset_path, spec)
    #print(code)

    # 3) EXECUTE
    stdout_all, stderr_all = _run_script_text(code, dataset_path, cleaned_path)

    # 4) SELF-REPAIR (bounded)
    attempts = 0
    while attempts < max_repair_attempts and (("Traceback" in stderr_all) or ("Error" in stderr_all)):
        attempts += 1
        patched = _self_repair(llm, dataset_path, spec, code, stderr_all)
        code = patched
        out, err = _run_script_text(code, dataset_path, cleaned_path)
        stdout_all += f"\n\n[repair_attempt_{attempts}_stdout]\n" + out
        stderr_all += f"\n\n[repair_attempt_{attempts}_stderr]\n" + err

    # Determine cleaned path (as required by contract)
    contract_path = os.path.splitext(cleaned_path)[0] + "_typed_data_contract.json"
    with open(contract_path, "w", encoding="utf-8") as contract_file:
        json.dump(typed_contract, contract_file, indent=2, ensure_ascii=False)
    report = []
    report.append(f"Method: {method}")
    report.append(f"Causal Query: {original_query or ''}")
    report.append(f"Rows/Cols before: see planner profile in memory.")
    report.append(f"Artifacts expected: {cleaned_path}, preprocessing_manifest.json, derived_columns.json")
    if ("Traceback" in stderr_all) or ("Error" in stderr_all):
        report.append("\n⚠️ LLM pipeline produced errors. Check stderr; artifacts may be missing or partial.")

    if not os.path.exists(cleaned_path):
        report.append(f"\n⚠️ Cleaned file not found at expected path; falling back to original dataset.")
        cleaned_path = dataset_path
    else:
        # LLM-generated cleaning code must never discard design columns.  In
        # particular, DiD needs the unit and time identifiers even though they
        # are neither treatment, outcome, nor controls. Restore any such
        # columns when row alignment is unchanged, otherwise use the original
        # data rather than returning a dataset that cannot be estimated.
        structural_keys = (
            "treatment_variable",
            "outcome_variable",
            "time_variable",
            "group_variable",
            "instrument_variable",
            "running_variable",
            "mediator_variable",
        )
        structural_columns = [
            variables.get(key) for key in structural_keys if variables.get(key)
        ]
        try:
            original_df = pd.read_csv(dataset_path)
            cleaned_df = pd.read_csv(cleaned_path)
            missing_structural = [
                column
                for column in structural_columns
                if column in original_df.columns and column not in cleaned_df.columns
            ]
            if missing_structural:
                if len(cleaned_df) == len(original_df):
                    for column in missing_structural:
                        cleaned_df[column] = original_df[column].to_numpy()
                    cleaned_df.to_csv(cleaned_path, index=False)
                    report.append(
                        "Restored required design columns: "
                        + ", ".join(missing_structural)
                    )
                else:
                    report.append(
                        "\nCleaning removed required design columns after changing "
                        "row alignment, falling back to original dataset."
                    )
                    cleaned_path = dataset_path
            required_roles = [
                variables.get("treatment_variable"),
                variables.get("outcome_variable"),
            ]
            if any(column and column not in cleaned_df.columns for column in required_roles):
                report.append("\nCleaning removed a required treatment/outcome column; falling back to original dataset.")
                cleaned_path = dataset_path
        except Exception as validation_error:
            report.append(
                f"\nCould not validate cleaned design columns "
                f"({validation_error}); falling back to original dataset."
            )
            cleaned_path = dataset_path
    
    return {
        "cleaned_dataset_path": cleaned_path,
        "cleaning_report_md": "\n".join(report),
        "generated_code": code,
        "stdout": stdout_all,
        "stderr": stderr_all,
        "typed_data_contract": typed_contract,
        "typed_data_contract_path": contract_path,
        "survival_roles": {
            "duration_variable": (spec.get("roles") or {}).get("duration"),
            "event_variable": (spec.get("roles") or {}).get("event"),
        },
    }
