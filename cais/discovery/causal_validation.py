"""
Causal discovery validation pipeline.

Validates CAIS outputs by learning causal graphs from data using causal-learn
and checking whether the assumptions behind CAIS's chosen method are consistent
with the discovered graph structure.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges

from cais.discovery.data_preprocess import ClassificationResult, classify_columns, prepare_data

logger = logging.getLogger(__name__)

# Fraction of discrete columns above which we fall back to KCI
CATEGORICAL_FRACTION_THRESHOLD = 0.5

# Subsample large datasets to keep discovery tractable
MAX_ROWS_FOR_DISCOVERY = 5000

# Skip all discovery when column count exceeds this (PC/GES blow up too)
MAX_COLS_FOR_DISCOVERY = 50

# Skip FCI when column count exceeds this (FCI is exponential in depth)
MAX_COLS_FOR_FCI = 25

# Minimum rows-per-column ratio; below this CI tests are underpowered
MIN_ROWS_PER_COL = 10

# Per-algorithm timeout in seconds
ALGORITHM_TIMEOUT = 120

# Map CAIS method names to canonical families
_OLS_FAMILY = {"ols", "linear_regression", "backdoor_adjustment",}
_IV_FAMILY = {"iv", "instrumental_variable", "2sls", "tsls"}
_DID_FAMILY = {"did", "did_canonical", "difference_in_differences"}


# ──────────────────────────────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GraphResult:
    """Holds a learned causal graph and helper lookups."""
    graph_matrix: np.ndarray          # shape (p, p)
    node_names: list[str]
    algorithm: str                    # "pc", "ges", "fci"

    def _idx(self, name: str) -> int | None:
        try:
            return self.node_names.index(name)
        except ValueError:
            return None

    # ── edge queries ────────────────────────────────────────────────
    def has_directed_edge(self, src: str, dst: str) -> bool:
        """True if src -> dst (arrow at dst, tail at src)."""
        i, j = self._idx(src), self._idx(dst)
        if i is None or j is None:
            return False
        return self.graph_matrix[j, i] == -1 and self.graph_matrix[i, j] == 1

    def has_any_edge(self, a: str, b: str) -> bool:
        i, j = self._idx(a), self._idx(b)
        if i is None or j is None:
            return False
        return self.graph_matrix[i, j] != 0 or self.graph_matrix[j, i] != 0

    def has_bidirected_edge(self, a: str, b: str) -> bool:
        """True if a <-> b (arrow at both ends — FCI latent confounder)."""
        i, j = self._idx(a), self._idx(b)
        if i is None or j is None:
            return False
        return self.graph_matrix[i, j] == 1 and self.graph_matrix[j, i] == 1

    def parents(self, node: str) -> list[str]:
        """Return nodes with a directed edge into `node`."""
        j = self._idx(node)
        if j is None:
            return []
        out = []
        for i, name in enumerate(self.node_names):
            # arrow at j from i: graph[j, i] == -1 and graph[i, j] == 1
            if self.graph_matrix[j, i] == -1 and self.graph_matrix[i, j] == 1:
                out.append(name)
        return out

    def children(self, node: str) -> list[str]:
        """Return nodes that `node` has a directed edge to."""
        i = self._idx(node)
        if i is None:
            return []
        out = []
        for j, name in enumerate(self.node_names):
            if self.graph_matrix[j, i] == -1 and self.graph_matrix[i, j] == 1:
                out.append(name)
        return out

    def descendants(self, node: str) -> set[str]:
        """All descendants of `node` via directed edges (BFS)."""
        visited: set[str] = set()
        queue = self.children(node)
        while queue:
            cur = queue.pop(0)
            if cur not in visited:
                visited.add(cur)
                queue.extend(self.children(cur))
        return visited

    def adjacent(self, node: str) -> list[str]:
        """All nodes connected to `node` by any edge."""
        i = self._idx(node)
        if i is None:
            return []
        out = []
        for j, name in enumerate(self.node_names):
            if self.graph_matrix[i, j] != 0 or self.graph_matrix[j, i] != 0:
                out.append(name)
        return out

    def to_nx_digraph(self) -> nx.DiGraph:
        """Convert directed edges to a networkx DiGraph (ignores undirected/bidirected)."""
        G = nx.DiGraph()
        G.add_nodes_from(self.node_names)
        for i, src in enumerate(self.node_names):
            for j, dst in enumerate(self.node_names):
                # src -> dst: graph[j, i] == -1 and graph[i, j] == 1
                if self.graph_matrix[j, i] == -1 and self.graph_matrix[i, j] == 1:
                    G.add_edge(src, dst)
        return G


@dataclass
class ValidationResult:
    """Collects validation flags and messages for a single CAIS output."""
    query_index: str
    method_family: str
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    graphs_used: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_index": self.query_index,
            "method_family": self.method_family,
            "flags": self.flags,
            "warnings": self.warnings,
            "info": self.info,
            "graphs_used": self.graphs_used,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


# ──────────────────────────────────────────────────────────────────────
#  Step 1: preprocess
# ──────────────────────────────────────────────────────────────────────

def get_data_preprocess(dataset_path: str) -> tuple[pd.DataFrame, ClassificationResult]:
    """Load CSV, classify columns, and return cleaned data + classification."""
    df = pd.read_csv(dataset_path)
    classification = classify_columns(df)
    cleaned = prepare_data(df, classification)
    # Subsample if too large for discovery algorithms
    if len(cleaned) > MAX_ROWS_FOR_DISCOVERY:
        logger.info(
            "Subsampling %d -> %d rows for discovery", len(cleaned), MAX_ROWS_FOR_DISCOVERY
        )
        cleaned = cleaned.sample(n=MAX_ROWS_FOR_DISCOVERY, random_state=42).reset_index(drop=True)
    return cleaned, classification


# ──────────────────────────────────────────────────────────────────────
#  Step 2: learn graphs
# ──────────────────────────────────────────────────────────────────────

def _pick_ci_test(classification: ClassificationResult) -> str:
    """Choose conditional-independence test based on data types."""
    n_discovery = len(classification.cols_for_discovery)
    if n_discovery == 0:
        return "fisherz"
    n_discrete = len(classification.discrete_flags)
    frac = n_discrete / n_discovery
    if frac > CATEGORICAL_FRACTION_THRESHOLD:
        return "chisq"
    return "fisherz"


class _Timeout:
    """Context manager that raises TimeoutError after `seconds` using SIGALRM."""

    def __init__(self, seconds: int, label: str = ""):
        self.seconds = seconds
        self.label = label

    def _handler(self, signum, frame):
        raise TimeoutError(f"{self.label} timed out after {self.seconds}s")

    def __enter__(self):
        self._old = signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *args):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old)


def get_causal_learn_graphs(
    cleaned_df: pd.DataFrame,
    classification: ClassificationResult,
    alpha: float = 0.05,
) -> dict[str, GraphResult]:
    """
    Run PC, GES, and FCI on the cleaned data.  Returns a dict keyed by
    algorithm name.  Returns empty dict if the data is too wide or too
    underpowered for reliable structure learning.
    """
    data = cleaned_df.values.astype(np.float64)
    col_names = list(cleaned_df.columns)
    n_rows, n_cols = data.shape
    graphs: dict[str, GraphResult] = {}

    if n_cols > MAX_COLS_FOR_DISCOVERY:
        logger.info(
            "Skipping all discovery: %d columns exceeds limit of %d",
            n_cols, MAX_COLS_FOR_DISCOVERY,
        )
        return graphs

    rows_per_col = n_rows / n_cols if n_cols > 0 else 0
    if rows_per_col < MIN_ROWS_PER_COL:
        logger.info(
            "Skipping all discovery: rows/columns ratio %.1f < %d "
            "(n=%d, p=%d) — CI tests would be underpowered",
            rows_per_col, MIN_ROWS_PER_COL, n_rows, n_cols,
        )
        return graphs

    ci_test = _pick_ci_test(classification)
    logger.info("CI test selected: %s (shape: %d x %d)", ci_test, n_rows, n_cols)

    # --- PC ---
    logger.info("  Running PC...")
    t0 = time.time()
    try:
        with _Timeout(ALGORITHM_TIMEOUT, "PC"):
            cg = pc(data, alpha=alpha, indep_test=ci_test, show_progress=False)
        graphs["pc"] = GraphResult(
            graph_matrix=cg.G.graph.copy(),
            node_names=col_names,
            algorithm="pc",
        )
        logger.info("  PC finished in %.1fs", time.time() - t0)
    except Exception as e:
        logger.warning("  PC failed after %.1fs: %s", time.time() - t0, e)

    # --- GES ---
    logger.info("  Running GES...")
    t0 = time.time()
    try:
        with _Timeout(ALGORITHM_TIMEOUT, "GES"):
            record = ges(data, score_func="local_score_BIC", node_names=col_names)
        graphs["ges"] = GraphResult(
            graph_matrix=record["G"].graph.copy(),
            node_names=col_names,
            algorithm="ges",
        )
        logger.info("  GES finished in %.1fs", time.time() - t0)
    except Exception as e:
        logger.warning("  GES failed after %.1fs: %s", time.time() - t0, e)

    # --- FCI (skip if too many columns) ---
    if n_cols > MAX_COLS_FOR_FCI:
        logger.info("Skipping FCI: %d columns exceeds limit of %d", n_cols, MAX_COLS_FOR_FCI)
    else:
        logger.info("  Running FCI...")
        t0 = time.time()
        try:
            with _Timeout(ALGORITHM_TIMEOUT, "FCI"):
                g, _ = fci(data, independence_test_method=ci_test, alpha=alpha, show_progress=False)
            graphs["fci"] = GraphResult(
                graph_matrix=g.graph.copy(),
                node_names=col_names,
                algorithm="fci",
            )
            logger.info("  FCI finished in %.1fs", time.time() - t0)
        except Exception as e:
            logger.warning("  FCI failed after %.1fs: %s", time.time() - t0, e)

    return graphs


# ──────────────────────────────────────────────────────────────────────
#  Step 3: validation checks
# ──────────────────────────────────────────────────────────────────────

def _classify_method(method_str: str) -> str:
    """Return 'ols', 'iv', 'did', or 'other'."""
    m = method_str.lower().strip()
    if m in _OLS_FAMILY:
        return "ols"
    if m in _IV_FAMILY:
        return "iv"
    if m in _DID_FAMILY:
        return "did"
    return "other"


def _resolve_column_name(var_name: str, available: list[str]) -> str | None:
    """
    Try to find `var_name` in the available column list, accounting for
    one-hot encoding (e.g. "treatment" might become "treatment_Hawthorne").
    Returns exact match first, then prefix match, else None.
    """
    if var_name in available:
        return var_name
    # Check for one-hot encoded variants
    prefix = var_name + "_"
    matches = [c for c in available if c.startswith(prefix)]
    if matches:
        return matches[0]  # return first variant
    return None


def _validate_ols(
    final_result: dict,
    graphs: dict[str, GraphResult],
    vr: ValidationResult,
) -> None:
    """Checks for OLS / linear regression / backdoor methods."""
    treatment = final_result.get("treatment_variable")
    outcome = final_result.get("outcome_variable")
    covariates = final_result.get("covariates") or []

    if not treatment or not outcome:
        vr.warnings.append("Missing treatment or outcome variable — cannot validate.")
        return

    # --- PC + GES checks ---
    for alg_name in ("pc", "ges"):
        g = graphs.get(alg_name)
        if g is None:
            continue
        vr.graphs_used.append(alg_name)

        t_col = _resolve_column_name(treatment, g.node_names)
        y_col = _resolve_column_name(outcome, g.node_names)

        if t_col is None or y_col is None:
            vr.info.append(
                f"[{alg_name}] Treatment or outcome not found in graph columns "
                f"(T={treatment}, Y={outcome}). Possibly dropped during preprocessing."
            )
            continue

        # Check covariates: should be parents of T, Y, or both
        t_descendants = g.descendants(t_col)
        t_parents = set(g.parents(t_col))
        y_parents = set(g.parents(y_col))
        adjustment_set = t_parents | y_parents

        for cov in covariates:
            cov_col = _resolve_column_name(cov, g.node_names)
            if cov_col is None:
                continue
            if cov_col in t_descendants:
                vr.flags.append(
                    f"[{alg_name}] Covariate '{cov}' is a descendant of treatment "
                    f"'{treatment}' — adjusting for it may induce bias."
                )
            elif cov_col not in adjustment_set and not g.has_any_edge(cov_col, t_col) and not g.has_any_edge(cov_col, y_col):
                vr.info.append(
                    f"[{alg_name}] Covariate '{cov}' has no edge to treatment or outcome "
                    f"in the graph — may be unnecessary but not harmful."
                )

        # D-separation check: does the covariate set block all backdoor paths?
        try:
            nx_g = g.to_nx_digraph()
            if t_col in nx_g and y_col in nx_g:
                # Mutilated graph: remove all edges out of treatment
                mutilated = nx_g.copy()
                mutilated.remove_edges_from(list(nx_g.out_edges(t_col)))
                cov_set = set()
                for cov in covariates:
                    c = _resolve_column_name(cov, g.node_names)
                    if c is not None:
                        cov_set.add(c)
                if not nx.d_separated(mutilated, {t_col}, {y_col}, cov_set):
                    vr.flags.append(
                        f"[{alg_name}] Covariates do not block all backdoor paths from "
                        f"'{treatment}' to '{outcome}' — backdoor criterion may not be satisfied."
                    )
                else:
                    vr.info.append(
                        f"[{alg_name}] Covariates satisfy the backdoor criterion "
                        f"(d-separation holds in the mutilated graph)."
                    )
        except Exception as e:
            vr.warnings.append(f"[{alg_name}] D-separation check failed: {e}")

    # --- FCI check for latent confounding ---
    fci_g = graphs.get("fci")
    if fci_g is not None:
        vr.graphs_used.append("fci")
        t_col = _resolve_column_name(treatment, fci_g.node_names)
        y_col = _resolve_column_name(outcome, fci_g.node_names)

        if t_col and y_col and fci_g.has_bidirected_edge(t_col, y_col):
            vr.flags.append(
                f"[fci] Bidirected edge {treatment} <-> {outcome} detected — "
                f"suggests latent confounding. Conditional ignorability may be "
                f"violated and the OLS effect estimate may be biased."
            )
        elif t_col and y_col:
            vr.info.append(
                f"[fci] No bidirected edge between {treatment} and {outcome} — "
                f"no evidence of latent confounding from FCI."
            )


def _validate_iv(
    final_result: dict,
    graphs: dict[str, GraphResult],
    vr: ValidationResult,
) -> None:
    """Checks for instrumental variable methods."""
    treatment = final_result.get("treatment_variable")
    outcome = final_result.get("outcome_variable")
    instrument = final_result.get("instrument_variable")

    if not treatment or not outcome:
        vr.warnings.append("Missing treatment or outcome variable — cannot validate.")
        return
    if not instrument:
        vr.warnings.append("No instrument variable specified in CAIS output — cannot validate IV assumptions.")
        return

    # --- FCI is the primary tool for IV validation ---
    fci_g = graphs.get("fci")
    if fci_g is not None:
        vr.graphs_used.append("fci")
        t_col = _resolve_column_name(treatment, fci_g.node_names)
        y_col = _resolve_column_name(outcome, fci_g.node_names)
        z_col = _resolve_column_name(instrument, fci_g.node_names)

        if not all([t_col, y_col, z_col]):
            vr.info.append(
                f"[fci] One or more IV variables not found in graph columns "
                f"(Z={instrument}, T={treatment}, Y={outcome})."
            )
        else:
            # Exclusion restriction: Z should NOT directly cause Y
            if fci_g.has_directed_edge(z_col, y_col):
                vr.flags.append(
                    f"[fci] Directed edge {instrument} -> {outcome} — "
                    f"exclusion restriction may be violated."
                )
            # Reverse: Y causing Z means instrument is endogenous
            if fci_g.has_directed_edge(y_col, z_col):
                vr.flags.append(
                    f"[fci] Directed edge {outcome} -> {instrument} — "
                    f"instrument may be endogenous (caused by the outcome)."
                )
            # Bidirected Z <-> Y: latent common cause of instrument and outcome
            if fci_g.has_bidirected_edge(z_col, y_col):
                vr.flags.append(
                    f"[fci] Bidirected edge {instrument} <-> {outcome} — "
                    f"latent common cause of instrument and outcome, "
                    f"violating instrument exogeneity."
                )
            # Other edge types (undirected/circle)
            if (not fci_g.has_directed_edge(z_col, y_col)
                    and not fci_g.has_directed_edge(y_col, z_col)
                    and not fci_g.has_bidirected_edge(z_col, y_col)):
                if fci_g.has_any_edge(z_col, y_col):
                    vr.warnings.append(
                        f"[fci] Undirected/circle edge between {instrument} and "
                        f"{outcome} — exclusion restriction uncertain."
                    )
                else:
                    vr.info.append(
                        f"[fci] No direct edge from {instrument} to {outcome} — "
                        f"exclusion restriction appears satisfied."
                    )

            # Relevance: Z should affect T
            if fci_g.has_directed_edge(z_col, t_col):
                vr.info.append(
                    f"[fci] Directed edge {instrument} -> {treatment} — "
                    f"instrument relevance condition supported."
                )
            elif fci_g.has_any_edge(z_col, t_col):
                vr.info.append(
                    f"[fci] Edge between {instrument} and {treatment} — "
                    f"instrument relevance plausible but direction uncertain."
                )
            else:
                vr.flags.append(
                    f"[fci] No edge from {instrument} to {treatment} — "
                    f"instrument relevance condition may be violated."
                )

            # T <-> Y bidirected is actually expected for IV (unmeasured confounding)
            if fci_g.has_bidirected_edge(t_col, y_col):
                vr.info.append(
                    f"[fci] Bidirected edge {treatment} <-> {outcome} detected — "
                    f"consistent with IV rationale (unmeasured confounding present)."
                )
            else:
                vr.warnings.append(
                    f"[fci] No bidirected edge between {treatment} and {outcome} — "
                    f"if no latent confounding, IV may be unnecessary; "
                    f"OLS/backdoor adjustment might suffice."
                )

    # --- PC/GES cross-check ---
    for alg_name in ("pc", "ges"):
        g = graphs.get(alg_name)
        if g is None:
            continue
        vr.graphs_used.append(alg_name)

        t_col = _resolve_column_name(treatment, g.node_names)
        y_col = _resolve_column_name(outcome, g.node_names)
        z_col = _resolve_column_name(instrument, g.node_names)

        if not all([t_col, y_col, z_col]):
            continue

        # Exclusion restriction
        if g.has_directed_edge(z_col, y_col):
            vr.flags.append(
                f"[{alg_name}] Direct edge {instrument} -> {outcome} — "
                f"exclusion restriction may be violated."
            )

        # Relevance: Z should affect T
        if not g.has_directed_edge(z_col, t_col) and not g.has_any_edge(z_col, t_col):
            vr.flags.append(
                f"[{alg_name}] No edge from {instrument} to {treatment} — "
                f"instrument relevance condition may be violated."
            )


def _validate_did(
    final_result: dict,
    graphs: dict[str, GraphResult],
    vr: ValidationResult,
) -> None:
    """Checks for difference-in-differences methods."""
    treatment = final_result.get("treatment_variable")
    outcome = final_result.get("outcome_variable")
    covariates = final_result.get("covariates") or []

    if not treatment or not outcome:
        vr.warnings.append("Missing treatment or outcome variable — cannot validate.")
        return

    # --- PC/GES: check for post-treatment bias in covariates ---
    for alg_name in ("pc", "ges"):
        g = graphs.get(alg_name)
        if g is None:
            continue
        vr.graphs_used.append(alg_name)

        t_col = _resolve_column_name(treatment, g.node_names)
        if t_col is None:
            continue

        t_descendants = g.descendants(t_col)

        for cov in covariates:
            cov_col = _resolve_column_name(cov, g.node_names)
            if cov_col is None:
                continue
            if cov_col in t_descendants:
                vr.flags.append(
                    f"[{alg_name}] Covariate '{cov}' is a descendant of treatment "
                    f"'{treatment}' — conditioning on post-treatment variables "
                    f"may bias the DiD estimate."
                )

    # --- FCI: flag latent time-varying confounders ---
    fci_g = graphs.get("fci")
    if fci_g is not None:
        vr.graphs_used.append("fci")
        t_col = _resolve_column_name(treatment, fci_g.node_names)
        y_col = _resolve_column_name(outcome, fci_g.node_names)

        if t_col and y_col and fci_g.has_bidirected_edge(t_col, y_col):
            vr.warnings.append(
                f"[fci] Bidirected edge {treatment} <-> {outcome} — "
                f"latent confounders detected. Parallel trends assumption "
                f"may not handle time-varying unobserved confounding."
            )
        elif t_col and y_col:
            vr.info.append(
                f"[fci] No bidirected edge between {treatment} and {outcome} — "
                f"no evidence of latent confounders from FCI."
            )


def get_validation(
    cais_output: dict,
    graphs: dict[str, GraphResult],
    query_index: str,
) -> ValidationResult:
    """
    Validate a single CAIS output entry against discovered graphs.

    Parameters
    ----------
    cais_output : dict
        A single entry from the CAIS output JSON (contains 'final_result', etc.).
    graphs : dict[str, GraphResult]
        Graphs learned from the same dataset.
    query_index : str
        The key/index of this entry in the output file.
    """
    final_result = cais_output.get("final_result", {})
    method_raw = final_result.get("method", "unknown")
    method_family = _classify_method(method_raw)

    vr = ValidationResult(query_index=query_index, method_family=method_family)

    if not graphs:
        vr.skipped = True
        vr.skip_reason = "No graphs were learned (all algorithms failed)."
        return vr

    if method_family == "ols":
        _validate_ols(final_result, graphs, vr)
    elif method_family == "iv":
        _validate_iv(final_result, graphs, vr)
    elif method_family == "did":
        _validate_did(final_result, graphs, vr)
    else:
        vr.info.append(
            f"Method '{method_raw}' (family='{method_family}') — "
            f"no specific validation checks implemented yet."
        )

    # Deduplicate graphs_used
    vr.graphs_used = list(dict.fromkeys(vr.graphs_used))
    return vr


# ──────────────────────────────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────────────────────────────

def run_validation_pipeline(
    cais_outputs_dir: str,
    base_data_dir: str | None = None,
    output_path: str | None = None,
    alpha: float = 0.05,
    datasets: set[str] | None = None,
) -> dict[str, list[dict]]:
    """
    For every JSON file in `cais_outputs_dir`, load each query entry,
    preprocess the dataset, learn graphs, and validate.

    Parameters
    ----------
    cais_outputs_dir : str
        Directory containing CAIS output JSON files.
    base_data_dir : str or None
        If dataset_path in the JSON is relative, resolve it relative to this.
        Defaults to the project root (two levels up from this file).
    output_path : str or None
        If provided, write all results to this JSON file.
    alpha : float
        Significance level for CI tests.
    datasets : set of str or None
        If provided, only process entries whose dataset filename (without
        extension) is in this set.  E.g. {"women", "smoking2", "rct_data_0"}.

    Returns
    -------
    dict mapping filename -> list of ValidationResult dicts.
    """
    if base_data_dir is None:
        base_data_dir = str(Path(__file__).resolve().parents[2])

    outputs_dir = Path(cais_outputs_dir)
    all_results: dict[str, list[dict]] = {}

    # Cache preprocessed data + graphs per dataset path
    _graph_cache: dict[str, dict[str, GraphResult]] = {}
    _preprocess_cache: dict[str, tuple[pd.DataFrame, ClassificationResult]] = {}

    json_files = sorted(outputs_dir.glob("*.json"))
    logger.info("Found %d JSON files in %s", len(json_files), cais_outputs_dir)

    for json_path in json_files:
        filename = json_path.name
        logger.info("Processing %s", filename)
        print(f"\n{'='*60}")
        print(f"  {filename}")
        print(f"{'='*60}")

        with open(json_path) as f:
            data = json.load(f)

        file_results: list[dict] = []

        for idx, entry in data.items():
            if isinstance(entry, str):
                # Error entries are stored as plain strings
                file_results.append({
                    "query_index": idx,
                    "skipped": True,
                    "skip_reason": f"CAIS output was an error string: {entry[:200]}",
                })
                continue

            dataset_path_raw = entry.get("dataset_path", "")

            # Filter by dataset name if requested
            if datasets is not None:
                dataset_stem = Path(dataset_path_raw).stem
                if dataset_stem not in datasets:
                    continue

            dataset_path = dataset_path_raw
            if not os.path.isabs(dataset_path):
                dataset_path = os.path.join(base_data_dir, dataset_path)

            if not os.path.exists(dataset_path):
                file_results.append({
                    "query_index": idx,
                    "skipped": True,
                    "skip_reason": f"Dataset not found: {dataset_path}",
                })
                continue

            # Preprocess + learn graphs (cached per dataset)
            if dataset_path not in _graph_cache:
                dataset_stem = Path(dataset_path).stem
                try:
                    logger.info("Preprocessing dataset '%s'...", dataset_stem)
                    cleaned_df, classification = get_data_preprocess(dataset_path)
                    _preprocess_cache[dataset_path] = (cleaned_df, classification)
                    logger.info(
                        "Learning graphs for dataset '%s' (shape: %d x %d)",
                        dataset_stem, cleaned_df.shape[0], cleaned_df.shape[1],
                    )
                    graphs = get_causal_learn_graphs(cleaned_df, classification, alpha=alpha)
                    _graph_cache[dataset_path] = graphs
                    logger.info("Done with dataset '%s' — %d graphs learned", dataset_stem, len(graphs))
                except Exception as e:
                    logger.error("Failed to process dataset %s: %s", dataset_path, e)
                    _graph_cache[dataset_path] = {}

            graphs = _graph_cache[dataset_path]

            vr = get_validation(entry, graphs, query_index=idx)
            result_dict = vr.to_dict()
            result_dict["query"] = entry.get("query", "")
            result_dict["dataset_path"] = dataset_path_raw
            result_dict["cais_method"] = entry.get("final_result", {}).get("method", "unknown")
            file_results.append(result_dict)

            # Print summary
            status = "SKIP" if vr.skipped else ("FLAG" if vr.flags else "OK")
            print(f"  [{idx}] {status} | {vr.method_family} | flags={len(vr.flags)} warns={len(vr.warnings)}")
            for flag in vr.flags:
                print(f"       FLAG: {flag}")
            for warn in vr.warnings:
                print(f"       WARN: {warn}")

        all_results[filename] = file_results

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {output_path}")

    return all_results


# ──────────────────────────────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate CAIS outputs with causal discovery.")
    parser.add_argument(
        "--cais_outputs_dir",
        type=str,
        default="cais_outputs",
        help="Directory containing CAIS output JSON files.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default=None,
        help="Base directory for resolving relative dataset paths.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_results.json",
        help="Path to write validation results JSON.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for CI tests.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of dataset names (without extension) to validate. "
             "E.g. 'women,smoking2,rct_data_0'. If omitted, all datasets are processed.",
    )
    args = parser.parse_args()

    ds_filter = set(args.datasets.split(",")) if args.datasets else None

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run_validation_pipeline(
        cais_outputs_dir=args.cais_outputs_dir,
        base_data_dir=args.base_data_dir,
        output_path=args.output,
        alpha=args.alpha,
        datasets=ds_filter,
    )
