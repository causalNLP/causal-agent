"""Sharp bounds from Manski (1990) and Balke--Pearl (1997).

The functions return intervals, not point estimates.  Outcome bounds are
required explicitly for Manski bounds.  Balke--Pearl is restricted to binary
Z, A and Y and solves the paper's sharp linear program under random
assignment/exogeneity, exclusion and consistency; monotonicity is not added
silently.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def manski_bounds(df: pd.DataFrame, treatment: str, outcome: str,
                  outcome_lower: float, outcome_upper: float, **_: Any) -> Dict[str, Any]:
    if outcome_lower >= outcome_upper:
        raise ValueError("outcome_lower must be smaller than outcome_upper.")
    dat = df[[treatment, outcome]].dropna()
    if len(dat) == 0:
        raise ValueError("No complete observations.")
    a = dat[treatment].astype(int).to_numpy(); y = dat[outcome].astype(float).to_numpy()
    if set(a) != {0, 1}:
        raise ValueError("Treatment must be binary and coded 0/1.")
    if np.any((y < outcome_lower) | (y > outcome_upper)):
        raise ValueError("Observed outcomes must lie within the declared support bounds.")
    p1 = float(np.mean(a == 1)); p0 = 1-p1
    mu1 = float(np.mean(y[a == 1])); mu0 = float(np.mean(y[a == 0]))
    lower = p1*mu1 + p0*outcome_lower - (p0*mu0 + p1*outcome_upper)
    upper = p1*mu1 + p0*outcome_upper - (p0*mu0 + p1*outcome_lower)
    return {"identified_set": [float(lower), float(upper)], "effect_estimate": None,
            "method_used": "Manski worst-case bounds",
            "diagnostics": {"n_complete": len(dat), "estimand": "ATE", "outcome_support": [outcome_lower, outcome_upper],
                            "warnings": ["The interval is a sharp identified set, not a point estimate."]}}


def balke_pearl_bounds(df: pd.DataFrame, instrument: str, treatment: str,
                       outcome: str, monotonicity: bool = False, **_: Any) -> Dict[str, Any]:
    dat = df[[instrument, treatment, outcome]].dropna()
    z = dat[instrument].astype(int).to_numpy(); a = dat[treatment].astype(int).to_numpy(); y = dat[outcome].astype(int).to_numpy()
    if set(z) != {0, 1} or set(a) != {0, 1} or not set(y).issubset({0, 1}):
        raise ValueError("Balke--Pearl requires binary Z, treatment and outcome coded 0/1.")
    # latent type: (Y(0),Y(1),A(0),A(1)); exogeneity makes its distribution
    # independent of Z.  Observed cells impose linear equality constraints.
    types = [(y0, y1, a0, a1) for y0 in (0,1) for y1 in (0,1)
             for a0 in (0,1) for a1 in (0,1)]
    Aeq = [np.ones(16)]; beq = [1.0]
    for zz in (0, 1):
        for aa in (0, 1):
            for yy in (0, 1):
                row = np.array([float((t[2+zz] == aa) and (t[t[2+zz]] == yy)) for t in types])
                Aeq.append(row); beq.append(float(np.mean((z == zz) & (a == aa) & (y == yy)) / np.mean(z == zz)))
    if monotonicity:
        # A(1) >= A(0): remove defier types.
        bounds = [(0, 0) if a1 < a0 else (0, None) for _, _, a0, a1 in types]
    else:
        bounds = [(0, None)] * 16
    objective = np.array([y1-y0 for y0, y1, _, _ in types], dtype=float)
    lo = linprog(objective, A_eq=np.asarray(Aeq), b_eq=np.asarray(beq), bounds=bounds, method="highs")
    hi = linprog(-objective, A_eq=np.asarray(Aeq), b_eq=np.asarray(beq), bounds=bounds, method="highs")
    if not lo.success or not hi.success:
        raise ValueError("Observed distribution is incompatible with the Balke--Pearl IV model.")
    return {"identified_set": [float(lo.fun), float(-hi.fun)], "effect_estimate": None,
            "method_used": "Balke--Pearl sharp IV bounds",
            "diagnostics": {"n_complete": len(dat), "estimand": "ATE", "monotonicity": bool(monotonicity),
                            "warnings": ["Exogeneity, exclusion and consistency are substantive assumptions; they are not tested by this LP."]}}


def estimate_manski_route(df, treatment, outcome, covariates=None, **kwargs):
    lower = kwargs.pop("outcome_lower", None)
    upper = kwargs.pop("outcome_upper", None)
    if lower is None or upper is None:
        raise ValueError("Manski bounds require outcome_lower and outcome_upper.")
    return manski_bounds(df, treatment, outcome, lower, upper)


def estimate_balke_pearl_route(df, treatment, outcome, instrument_variable=None,
                               covariates=None, **kwargs):
    if not instrument_variable:
        raise ValueError("Balke--Pearl bounds require instrument_variable.")
    return balke_pearl_bounds(df, instrument_variable, treatment, outcome,
                              monotonicity=bool(kwargs.pop("monotonicity", False)))
