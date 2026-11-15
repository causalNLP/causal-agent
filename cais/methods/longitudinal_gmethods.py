"""Restricted longitudinal g-methods based on Robins' g-formula/MSM.

The interfaces make the data contract explicit: one row per person and time,
binary treatment, and a final observed outcome.  The g-formula implementation
is the two-time-point parametric formula; the MSM implementation estimates a
user-specified marginal model after stabilized treatment weighting.
"""
from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression


def _check_long(df, id_col, time_col, treatment_col):
    req = [id_col, time_col, treatment_col]
    missing = [c for c in req if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    dat = df.dropna(subset=req).copy()
    if not set(dat[treatment_col].astype(int).unique()).issubset({0,1}):
        raise ValueError("Longitudinal treatment must be binary coded 0/1.")
    return dat.sort_values([id_col, time_col])


def estimate_msm(df: pd.DataFrame, id_col: str, time_col: str, treatment_col: str,
                 outcome_col: str, confounders: Sequence[str],
                 msm_terms: Sequence[str] = ("treatment",), **_: Any) -> Dict[str, Any]:
    """Fit a stabilized-IPTW marginal structural model for a final outcome.

    ``msm_terms`` may contain ``treatment`` and ``time``; the returned
    coefficients are the parameters of that explicitly requested marginal
    model, not an automatically selected causal estimand.
    """
    dat = _check_long(df, id_col, time_col, treatment_col)
    dat = dat.copy()
    dat["_A_history"] = dat.groupby(id_col)[treatment_col].shift(1).fillna(0)
    if outcome_col not in dat or any(c not in dat for c in confounders):
        raise ValueError("Outcome and confounder columns are required.")
    weights = pd.Series(1.0, index=dat.index, dtype=float)
    for _, idx in dat.groupby(time_col, sort=True).groups.items():
        block = dat.loc[idx]
        history = block["_A_history"].to_numpy()
        # A conservative per-time model: numerator uses past treatment only;
        # denominator also uses supplied time-varying confounders.
        Xn = sm.add_constant(np.asarray(history).reshape(-1,1), has_constant="add")
        Xd = sm.add_constant(pd.concat([pd.Series(history, index=block.index, name="A_history"), block[list(confounders)]], axis=1), has_constant="add")
        a = block[treatment_col].astype(int).to_numpy()
        if len(np.unique(a)) < 2: raise ValueError("Each treatment time must contain both arms for stabilized weights.")
        pn = np.clip(LogisticRegression(max_iter=1000).fit(Xn, a).predict_proba(Xn)[:,1], .01, .99)
        pdn = np.clip(LogisticRegression(max_iter=1000).fit(Xd, a).predict_proba(Xd)[:,1], .01, .99)
        weights.loc[block.index] *= np.where(a == 1, pn/pdn, (1-pn)/(1-pdn))
    final = dat.groupby(id_col, sort=False).tail(1).copy()
    final["_weight"] = weights.loc[final.index]
    X = pd.DataFrame(index=final.index)
    if "treatment" in msm_terms: X["treatment"] = final[treatment_col].astype(float)
    if "time" in msm_terms: X["time"] = final[time_col].astype(float)
    if X.shape[1] == 0: raise ValueError("msm_terms must include treatment and/or time.")
    X = sm.add_constant(X, has_constant="add")
    model = sm.WLS(final[outcome_col].astype(float), X, weights=final["_weight"]).fit(cov_type="HC1")
    return {"effect_estimate": float(model.params.get("treatment", np.nan)),
            "standard_error": float(model.bse.get("treatment", np.nan)),
            "confidence_interval": list(map(float, model.conf_int().loc["treatment"])),
            "method_used": "stabilized-IPTW marginal structural model",
            "coefficients": {str(k): float(v) for k,v in model.params.items()},
            "diagnostics": {"n_subjects": int(final[id_col].nunique()), "n_rows": len(dat),
                            "weight_min": float(final["_weight"].min()), "weight_max": float(final["_weight"].max()),
                            "estimand": "coefficient of requested MSM", "warnings": ["Sequential exchangeability and model specification are advisory assumptions."]}}


def estimate_gformula_two_time(df: pd.DataFrame, id_col: str, time_col: str,
                               treatment_col: str, outcome_col: str,
                               intermediate_covariate: str,
                               baseline_covariates: Sequence[str],
                               regime_a0: int = 0, regime_a1: int = 0,
                               compare_a0: int = 0, compare_a1: int = 0,
                               **_: Any) -> Dict[str, Any]:
    """Parametric g-formula for two static treatment regimes (A0,A1)."""
    dat = _check_long(df, id_col, time_col, treatment_col)
    times = sorted(dat[time_col].unique())
    if len(times) != 2: raise ValueError("This g-formula route requires exactly two time points.")
    if intermediate_covariate not in dat or outcome_col not in dat:
        raise ValueError("Intermediate covariate and final outcome are required.")
    wide = dat.pivot(index=id_col, columns=time_col)
    a0, a1 = wide[treatment_col].iloc[:,0].astype(float), wide[treatment_col].iloc[:,1].astype(float)
    l1 = wide[intermediate_covariate].iloc[:,1].astype(float)
    y = wide[outcome_col].iloc[:,1].astype(float)
    base = pd.DataFrame({c: wide[c].iloc[:,0].astype(float) for c in baseline_covariates}, index=wide.index)
    Xl = sm.add_constant(pd.concat([base, a0.rename("A0")], axis=1), has_constant="add")
    lm = sm.OLS(l1, Xl).fit()
    def regime_mean(r0, r1):
        lpred = lm.predict(sm.add_constant(pd.concat([base, pd.Series(r0,index=base.index,name="A0")], axis=1), has_constant="add"))
        Xy = sm.add_constant(pd.concat([base, pd.Series(r0,index=base.index,name="A0"), pd.Series(r1,index=base.index,name="A1"), lpred.rename("L1")], axis=1), has_constant="add")
        ym = sm.OLS(y, sm.add_constant(pd.concat([base, a0.rename("A0"), a1.rename("A1"), l1.rename("L1")], axis=1), has_constant="add")).fit()
        return float(np.mean(ym.predict(Xy)))
    m = regime_mean(regime_a0, regime_a1)
    m0 = regime_mean(compare_a0, compare_a1)
    return {"effect_estimate": m - m0, "regime_mean": m, "comparison_regime_mean": m0,
            "method_used": "parametric longitudinal g-computation",
            "diagnostics": {"n_subjects": int(len(wide)), "regime": [int(regime_a0), int(regime_a1)],
                            "comparison_regime": [int(compare_a0), int(compare_a1)],
                            "estimand": "E[Y^{a0,a1}] - E[Y^{a0*,a1*}]", "warnings": ["Sequential exchangeability, consistency, positivity and both parametric models are advisory assumptions."]}}


def estimate_msm_route(df, treatment, outcome, covariates=None, group_variable=None,
                       time_variable=None, **kwargs):
    """METHOD_MAPPING adapter; longitudinal columns must be supplied explicitly."""
    if not group_variable or not time_variable:
        raise ValueError("MSM routing requires group_variable and time_variable.")
    return estimate_msm(df, id_col=group_variable, time_col=time_variable,
                        treatment_col=treatment, outcome_col=outcome,
                        confounders=list(covariates or []), **kwargs)


def estimate_gformula_route(df, treatment, outcome, covariates=None,
                            group_variable=None, time_variable=None, **kwargs):
    if not group_variable or not time_variable:
        raise ValueError("Longitudinal g-computation requires group_variable and time_variable.")
    intermediate = kwargs.pop("intermediate_covariate", None)
    if not intermediate:
        raise ValueError("Longitudinal g-computation requires intermediate_covariate.")
    return estimate_gformula_two_time(
        df, id_col=group_variable, time_col=time_variable,
        treatment_col=treatment, outcome_col=outcome,
        intermediate_covariate=intermediate,
        baseline_covariates=list(kwargs.pop("baseline_covariates", covariates or [])), **kwargs)
