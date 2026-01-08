import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from typing import Dict, Any, Optional, List, Union
from rddensity import rddensity

# Try to import optimal_bandwidth from rdd package
try:
    from rdd.rdd import optimal_bandwidth
    _has_rdd_optimal_bandwidth = True
except ImportError:
    _has_rdd_optimal_bandwidth = False
    optimal_bandwidth = None

## ------ For observational methods relying on conditional ignorability ----------
# ----------- Internal helpers ------------

def _smd_from_groups(a: np.ndarray, b: np.ndarray) -> float:
    """SMD = (mu_t - mu_c) / sqrt((var_t + var_c)/2)."""

    mu_t, mu_c = np.nanmean(a), np.nanmean(b)
    var_t, var_c = np.nanvar(a, ddof=1), np.nanvar(b, ddof=1)
    denom = np.sqrt((var_t + var_c) / 2.0 + 1e-12)

    return float((mu_t - mu_c) / (denom if denom > 0 else 1.0))

def _one_hot_df(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """One-hot encode categorical cols; return expanded DF and mapping."""

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if c not in num_cols]
    mapping: Dict[str, List[str]] = {}
    parts = []
    if num_cols:
        parts.append(df[num_cols].astype(float))
        for c in num_cols: 
            mapping[c] = [c]
    if cat_cols:
        enc = OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore")
        arr = enc.fit_transform(df[cat_cols])
        oh_cols = enc.get_feature_names_out(cat_cols).tolist()
        parts.append(pd.DataFrame(arr, index=df.index, columns=oh_cols))
        for c in cat_cols:
            mapping[c] = [col for col in oh_cols if col.startswith(c + "_")]
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)

    return X, mapping

# ---------- Public diagnostics ----------

def compute_covariate_smds(df: pd.DataFrame, treat_col: str, covariates: List[str]) -> Dict[str, float]:
    """SMD for each covariate; categorical collapsed to max|SMD| across levels."""

    tmask = df[treat_col].values.astype(bool)
    X, mapping = _one_hot_df(df, covariates)
    smds: Dict[str, float] = {}
    for base, cols in mapping.items():
        vals_t = X[cols].values[tmask]
        vals_c = X[cols].values[~tmask]
        if len(cols) == 1:
            smd = _smd_from_groups(vals_t.ravel(), vals_c.ravel())
        else:
            smd = float(np.max([abs(_smd_from_groups(vals_t[:, i], vals_c[:, i])) for i in range(len(cols))]))
        smds[base] = smd

    return smds

def estimate_propensity_scores(df: pd.DataFrame, treat_col: str, covariates: List[str]) -> np.ndarray:
    """Logistic regression propensity scores on one-hot covariates."""

    y = df[treat_col].values.astype(int)
    X, _ = _one_hot_df(df, covariates)
    if X.shape[1] == 0:
        return np.full(len(df), fill_value=y.mean(), dtype=float)
    lr = LogisticRegression(max_iter=200, solver="lbfgs")
    lr.fit(X, y)

    return lr.predict_proba(X)[:, 1]

def compute_ps_smd(ps: np.ndarray, treat: np.ndarray) -> float:
    """SMD of the propensity score itself (single summary metric)."""

    tmask = treat.astype(bool)

    return _smd_from_groups(ps[tmask], ps[~tmask])

def summarize_ps_overlap(ps: np.ndarray, treat: np.ndarray) -> Dict[str, Any]:
    """Simple overlap summary (min/max & quantiles)."""

    tmask = treat.astype(bool)
    def stats(v):
        return {
            "min": float(np.min(v)),
            "quantile_25": float(np.quantile(v, 0.25)),
            "median": float(np.median(v)),
            "quantile_75": float(np.quantile(v, 0.75)),
            "max": float(np.max(v)),
        }
    treated_stats = stats(ps[tmask])
    control_stats = stats(ps[~tmask])
    overlap = not (treated_stats["min"] > control_stats["max"] or control_stats["min"] > treated_stats["max"])
    return {
        "treated": treated_stats,
        "control": control_stats,
        "range_overlap": overlap,
    }

def psm_diagnostics(df: pd.DataFrame, treatment: str, covariates: List[str]) -> Dict[str, Any]:
    """Bundle: covariate SMDs, PS array, PS SMD, and overlap summary."""

    smds = compute_covariate_smds(df, treatment, covariates)
    ps = estimate_propensity_scores(df, treatment, covariates)
    ps_smd = compute_ps_smd(ps, df[treatment].values.astype(int))
    overlap = summarize_ps_overlap(ps, df[treatment].values.astype(int))

    return {
        "covariate_SMDs": smds,
        "propensity_scores": ps,
        "propensity_SMD": ps_smd,
        "ps_overlap": overlap,
    }

## ----------------------------- For Difference-in-Differences --------------------------------

def _as_categorical_time(df: pd.DataFrame, time_col: str):
    # Ensure sortable time; create an integer index for regression

    tvals = pd.Categorical(df[time_col], ordered=True)
    if not tvals.ordered:
        # try to order by unique sorted values
        uniq = sorted(df[time_col].unique())
        tvals = pd.Categorical(df[time_col], categories=uniq, ordered=True)
    t_idx = tvals.codes  # -1 if NA

    return tvals, t_idx

## This will work only if time_col is non-binary
def infer_treatment_timing(df: pd.DataFrame, time_col: str, group_col: str, treat_col: str,
                           treated_value: Optional[int]=1) -> Dict[str, Any]:
    """
    Infer if the time variable encodes treatment timing:
    - Finds treated group (assumes binary group_col with values {0,1} or two categories)
    - Finds the first time period where treated group's treatment share jumps.
    """
    out: Dict[str, Any] = {"ok": False, "treated_group": None, "adoption_time": None,
                           "pre_periods": [], "post_periods": [], "notes": ""}

    # Identify treated group as the group whose mean treatment eventually becomes high
    grp_values = df[group_col].dropna().unique()
    if len(grp_values) != 2:
        out["notes"] = "group_variable not binary; cannot infer clear treated/control groups."
        return out

    # compute by group & time
    mean_t = df.groupby([group_col, time_col], as_index=False)[treat_col].mean()
    # Rename the treatment column to avoid conflicts
    mean_t = mean_t.rename(columns={treat_col: 'mean_treatment'})
    #print(mean_t)
    # choose group with larger max mean treatment as treated_group
    max_by_group = mean_t.groupby(group_col)['mean_treatment'].max()
    treated_group = max_by_group.idxmax()
    control_group = [g for g in grp_values if g != treated_group][0]

    # adoption time: first time where treated_group mean_t >= 0.5 and strictly above its pre-history
    tg = mean_t[mean_t[group_col] == treated_group].sort_values(time_col)
    # any time periods?
    if tg.empty:
        out["notes"] = "No time variation for treated group."
        return out
    # pick earliest period with high treatment share
    adopt_rows = tg.loc[tg['mean_treatment'] >= 0.5]
    if adopt_rows.empty:
        out["notes"] = "Treated group never shows high treatment intensity."
        return out
    adoption_time = adopt_rows.iloc[0][time_col]

    # pre/post sets
    all_times = sorted(df[time_col].dropna().unique().tolist())
    pre = [t for t in all_times if t < adoption_time]
    post = [t for t in all_times if t >= adoption_time]

    # sanity: control should remain mostly untreated overall
    ctrl_max = mean_t.loc[mean_t[group_col] == control_group, 'mean_treatment'].max()
    ok_ctrl_untreated = ctrl_max < 0.5

    # sanity: treated pre periods should be mostly untreated
    tg_pre = tg[tg[time_col].isin(pre)]['mean_treatment'] if pre else pd.Series([], dtype=float)
    ok_no_anticipation = True
    if len(tg_pre) > 0:
        ok_no_anticipation = float(tg_pre.max()) < 0.5

    out.update({
        "ok": bool(ok_ctrl_untreated),
        "treated_group": treated_group,
        "control_group": control_group,
        "adoption_time": adoption_time,
        "pre_periods": pre,
        "post_periods": post,
        "ok_no_anticipation": bool(ok_no_anticipation),
        "notes": out["notes"]
    })
    return out

def pretrend_parallel_test(df: pd.DataFrame, time_col: str, group_col: str, outcome_col: str,
                           treated_group) -> Dict[str, Any]:
    """
    Parallel trends visual proxy using pre-periods only:
    Regress outcome on time_index * treated_group (interaction) with group and time fixed effects disabled
    (simple slopes model over aggregated means). Reports interaction slope diff and p-value.
    If < 3 pre periods, mark as insufficient and skip test.
    """
    # Build aggregated mean outcome by group-time for stability
    agg = df.groupby([group_col, time_col])[outcome_col].mean().reset_index()

    # Determine ordering/index for time
    tcat, t_idx = _as_categorical_time(agg, time_col)
    agg = agg.assign(t_idx=t_idx)

    # pre-periods: strictly before treated group's adoption (caller should subset already if desired)
    # Here we select all periods that exist before any observed treatment jump; caller provides pre list if needed.
    # To align with the "visual slopes" idea, we only need >= 3 distinct pre periods.
    # We'll detect number of unique times where treated group's mean treatment is low; caller can pass pre list.
    # For simplicity, we infer pre as the first K unique times (K >= 3) without treatment info;
    # but better: caller supplies pre periods from infer_treatment_timing. We support both.
    return {"ok": None, "pval": None, "slope_diff": None, "n_pre_periods": None, "insufficient_pre_periods": True}

def pretrend_parallel_test_with_periods(df: pd.DataFrame, time_col: str, group_col: str, outcome_col: str,
                                        pre_periods: list, treated_group) -> Dict[str, Any]:
    """
    Preferred version: use explicit pre_periods (from infer_treatment_timing). 
    Fit outcome ~ t_idx * I[group==treated] on pre-periods and test the interaction coefficient.
    """
    out = {"ok": None, "pval": None, "slope_diff": None, "n_pre_periods": len(set(pre_periods)),
           "insufficient_pre_periods": False}
    if len(set(pre_periods)) < 3:
        out["insufficient_pre_periods"] = True
        return out

    sub = df[df[time_col].isin(pre_periods)].copy()
    if sub.empty:
        out["insufficient_pre_periods"] = True
        return out

    # aggregate means by group-time for stability (visual-style slopes)
    agg = sub.groupby([group_col, time_col])[outcome_col].mean().reset_index()
    tcat, t_idx = _as_categorical_time(agg, time_col)
    agg["t_idx"] = t_idx
    agg["treated_grp_flag"] = (agg[group_col] == treated_group).astype(int)

    X = sm.add_constant(agg[["t_idx", "treated_grp_flag"]])
    X["t_idx_x_treated"] = agg["t_idx"] * agg["treated_grp_flag"]
    y = agg[outcome_col].values

    ols = sm.OLS(y, X).fit(cov_type="HC1")
    coef_name = "t_idx_x_treated"
    slope_diff = float(ols.params.get(coef_name, np.nan))
    pval = float(ols.pvalues.get(coef_name, np.nan))

    out.update({
        "ok": bool(pval >= 0.10),  # fail to reject difference in pre-trends at 10%
        "pval": pval,
        "slope_diff": slope_diff
    })
    return out


## ----------------------------- For Instrumental Variables --------------------------------

def iv_first_stage_relevance(df: pd.DataFrame,
                             instrument: Union[str, List[str]],
                             treatment: str,
                             controls: List[str] | None = None) -> Dict[str, Any]:
    """
    Compute first-stage F-statistic for IV relevance: regress D on Z (+ X).
    - If multiple instruments, returns the joint F for all Z.
    - Uses HC1 robust covariance.
    """
    Z = [instrument] if isinstance(instrument, str) else list(instrument)
    X = pd.DataFrame(index=df.index)
    X[Z] = df[Z]
    if controls:
        X[controls] = df[controls]
    X = sm.add_constant(X, has_constant="add")

    y = df[treatment]
    ols = sm.OLS(y, X).fit(cov_type="HC1")

    # Joint F-test that all instrument coefficients == 0
    z_idx = [X.columns.get_loc(z) for z in Z]
    R = np.zeros((len(Z), len(X.columns)))
    for i, j in enumerate(z_idx):
        R[i, j] = 1.0
    ftest = ols.f_test(R)

    fval = float(ftest.fvalue) if np.ndim(ftest.fvalue) == 0 else float(ftest.fvalue.item())
    pval = float(ftest.pvalue)
    return {
        "first_stage_F": fval,
        "first_stage_F_p": pval,
        "n_instruments": len(Z),
        "weak_iv_flag": bool(fval < 10.0)  # Stock–Yogo rule-of-thumb
    }

## ----------------------------- For Regression Discontinuity Design --------------------------------

def rdd_design_compliance(df: pd.DataFrame, running: str, treatment: str, cutoff: float, 
                          thresh_comply: float=0.8) -> Dict[str, Any]:
    """
    Check if treatment is (approximately) assigned by cutoff: T ≈ 1{running >= cutoff}.
    Returns compliance rate and misclassification share.
    """
    if running not in df.columns or treatment not in df.columns:
        return {"ok": False, "error": "Missing running/treatment column."}
    


    assign = (df[running] >= cutoff).astype(int)
    t = df[treatment].astype(int)
    if t.nunique() != 2:
        return {"ok": False, "error": "Treatment variable not binary."}
    
    agree = (assign == t)
    rate = float(agree.mean())
    return {
        "ok": bool(rate >= thresh_comply),          # loose gate; tune as needed
        "compliance_rate": rate,
        "misclassified_share": float(1.0 - rate),
        "n": int(len(df))}

def rdd_window_summary(df: pd.DataFrame, running: str, outcome: str, cutoff: float,
                       h: Optional[float] = None) -> Dict[str, Any]:
    """
    Summarize outcome means just around the cutoff for visual inspection.
    To determine h i.e. the bandwidth we use the Imbens-Kalyanaraman optimal bandwidth selector, in the 
    rdd package.
    """

    x = df[running].astype(float)
    y = df[outcome].astype(float)
    mean_running = float(np.nanmean(x))
    ## data is already centered
    if abs(mean_running) < 0.01:
        cutoff = 0.0  # snap to zero if close
    if h is None:
        #sd = float(np.nanstd(x))
        #h = 0.2 * sd if sd > 0 else np.nanmax(np.abs(x - cutoff)) * 0.2
        h = optimal_bandwidth(y.values, x.values, cutoff)

    mask = (x >= cutoff - h) & (x <= cutoff + h)
    subx, suby = x[mask], y[mask]
    left = suby[subx < cutoff]
    right = suby[subx >= cutoff]

    return {
        "window_h": float(h),
        "n_left": int(left.size),
        "n_right": int(right.size),
        "mean_left": float(np.nanmean(left)) if left.size else None,
        "mean_right": float(np.nanmean(right)) if right.size else None,
        "jump_right_minus_left": (float(np.nanmean(right) - np.nanmean(left))
                                  if left.size and right.size else None)}

def rdd_bins_for_plot(df: pd.DataFrame, running: str, outcome: str, cutoff: float,
                      h: float, bins_per_side: int = 10) -> Dict[str, Any]:
    """
    Prepare simple binned averages for a binscatter near cutoff (purely for plotting later).
    """

    x = df[running].astype(float)
    y = df[outcome].astype(float)
    mask = (x >= cutoff - h) & (x <= cutoff + h)
    sub = df.loc[mask, [running, outcome]].copy()
    left = sub[sub[running] < cutoff]
    right = sub[sub[running] >= cutoff]

    def binside(side_df, start, end, n):
        if side_df.empty: return []
        edges = np.linspace(start, end, n + 1)
        idx = np.digitize(side_df[running], edges, right=False) - 1
        out = []
        for b in range(n):
            sel = side_df[idx == b]
            if sel.empty: continue
            out.append({
                "x_center": float((edges[b] + edges[b+1]) / 2),
                "y_mean": float(sel[outcome].mean()),
                "n": int(len(sel))
            })
        return out

    left_bins = binside(left, cutoff - h, cutoff, bins_per_side)
    right_bins = binside(right, cutoff, cutoff + h, bins_per_side)

    return {"cutoff": float(cutoff), "h": float(h), "left_bins": left_bins, "right_bins": right_bins}


def mccrary_test(df, running_var, cutoff):
    """
    Performs the McCrary density test

    Args:
        df (pd.DataFrame): DataFrame containing the running variable 
        running_var (str): Column name of the running variable 
        cutoff (float): cutoff value for the running variable 
    Returns:
         (float) the p-value of the test 
    """

    if running_var not in df.columns:
        return np.nan 

    running_vals = df[running_var].values
    test = rddensity(running_vals, c=cutoff)
    p_val = float(test.test['p_jk'])

    return p_val