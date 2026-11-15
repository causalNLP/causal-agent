"""Cross-fitted AIPW and DML estimators for binary treatments.

The module deliberately implements a narrow, explicit contract: a binary
treatment, a numeric outcome, and a pre-treatment adjustment set.  It does
not claim to solve longitudinal treatment, survival, or continuous-treatment
problems.  Identification assumptions (exchangeability, consistency and
positivity) are reported as advisory; the empirical checks below only assess
data prerequisites and overlap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from cais.methods.causal_method import CausalMethod


def _design(data: pd.DataFrame, covariates: List[str]) -> np.ndarray:
    if not covariates:
        raise ValueError("AIPW/DML require at least one pre-treatment covariate.")
    x = pd.get_dummies(data[covariates], drop_first=False, dtype=float)
    matrix = x.to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Covariates must form a finite numeric design matrix.")
    return matrix


def _prepare(df: pd.DataFrame, treatment: str, outcome: str,
             covariates: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    required = [treatment, outcome, *covariates]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = df[required].dropna().copy()
    if len(data) < 20:
        raise ValueError("Cross-fitted AIPW/DML require at least 20 complete observations.")
    values = set(data[treatment].unique())
    if not values.issubset({0, 1, False, True}) or len(values) != 2:
        raise ValueError("Treatment must be binary and coded 0/1.")
    if not pd.api.types.is_numeric_dtype(data[outcome]):
        raise ValueError("The outcome must be numeric.")
    w = data[treatment].astype(int).to_numpy()
    y = data[outcome].astype(float).to_numpy()
    return data, _design(data, covariates), w, y


def _crossfit_nuisance(x, w, y, n_splits, random_state, clip,
                       outcome_model, propensity_model):
    counts = np.bincount(w, minlength=2)
    folds = min(int(n_splits), int(counts.min()))
    if folds < 2:
        raise ValueError("At least two observations per treatment arm are required for cross-fitting.")
    splitter = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    m0 = np.empty(len(y)); m1 = np.empty(len(y)); e = np.empty(len(y))
    for train, test in splitter.split(x, w):
        pmodel = clone(propensity_model).fit(x[train], w[train])
        e[test] = np.clip(pmodel.predict_proba(x[test])[:, 1], clip, 1 - clip)
        for arm, target in ((0, m0), (1, m1)):
            arm_train = train[w[train] == arm]
            if len(arm_train) < 2:
                raise ValueError("Each cross-fitting training fold needs observations in both arms.")
            model = clone(outcome_model).fit(x[arm_train], y[arm_train])
            target[test] = model.predict(x[test])
    return m0, m1, e, int(folds)


def _summary(effect, influence, method, data, w, e, warnings, parameters):
    influence = np.asarray(influence, dtype=float)
    se = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
    ci = [float(effect - 1.96 * se), float(effect + 1.96 * se)]
    return {
        "effect_estimate": float(effect),
        "standard_error": se,
        "confidence_interval": ci,
        "p_value": float(2 * stats.norm.sf(abs(effect / se))) if se > 0 else None,
        "method_used": method,
        "diagnostics": {
            "n_complete": int(len(data)),
            "treated": int(w.sum()), "control": int((1 - w).sum()),
            "propensity_min": float(e.min()), "propensity_max": float(e.max()),
            "propensity_at_clip_share": float(np.mean((e <= e.min() + 1e-12) | (e >= e.max() - 1e-12))),
            "effective_sample_size": float(np.sum(e * (1 - e)) ** 2 / np.sum(np.square(e * (1 - e)))),
            "warnings": warnings,
        },
        "parameters": parameters,
    }


def estimate_aipw(df: pd.DataFrame, treatment: str, outcome: str,
                  covariates: List[str], target: str = "ATE", n_splits: int = 5,
                  random_state: int = 0, propensity_clip: float = 0.01,
                  outcome_model: Any = None, propensity_model: Any = None,
                  **_: Any) -> Dict[str, Any]:
    """Estimate ATE, ATT, or ATC with cross-fitted augmented IPW.

    ``target`` changes the estimand; it is never inferred from the data.
    Clipping protects numerical stability but does not silently trim units.
    """
    target = str(target).upper()
    if target not in {"ATE", "ATT", "ATC"}:
        raise ValueError("target must be ATE, ATT, or ATC.")
    if not 0 < float(propensity_clip) < 0.5:
        raise ValueError("propensity_clip must lie strictly between 0 and 0.5.")
    data, x, w, y = _prepare(df, treatment, outcome, list(covariates or []))
    outcome_model = outcome_model or RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                                            random_state=random_state, n_jobs=1)
    propensity_model = propensity_model or LogisticRegression(max_iter=1000, random_state=random_state)
    m0, m1, e, folds = _crossfit_nuisance(x, w, y, n_splits, random_state,
                                          propensity_clip, outcome_model, propensity_model)
    if target == "ATE":
        influence = m1 - m0 + w * (y - m1) / e - (1 - w) * (y - m0) / (1 - e)
        effect = float(np.mean(influence))
    elif target == "ATT":
        treated = w == 1
        influence = m1[treated] - m0[treated] + (1 - e[treated]) / e[treated] * (y[treated] - m1[treated])
        effect = float(np.mean(influence))
    else:
        control = w == 0
        influence = m1[control] - m0[control] - e[control] / (1 - e[control]) * (y[control] - m0[control])
        effect = float(np.mean(influence))
    warnings = [
        "Exchangeability, consistency and correct covariate timing are substantive advisory assumptions.",
        "The confidence interval uses the empirical influence function and assumes independent observations.",
    ]
    if np.mean((e <= propensity_clip + 1e-12) | (e >= 1 - propensity_clip - 1e-12)) > 0.05:
        warnings.append("More than 5% of propensities hit clipping bounds; positivity is weak.")
    result = _summary(effect, influence - effect, f"cross-fitted AIPW ({target})", data, w, e,
                      warnings, {"target": target, "n_splits": folds, "propensity_clip": propensity_clip,
                                 "covariates": list(covariates)})
    result["diagnostics"]["estimand"] = target
    return result


def estimate_dml(df: pd.DataFrame, treatment: str, outcome: str, covariates: List[str],
                 target: str = "ATE", **kwargs: Any) -> Dict[str, Any]:
    """Cross-fitted orthogonal-score DML for the binary-treatment ATE.

    DML is intentionally restricted to ATE here: ATT/ATC require a different
    target-specific score and are rejected rather than mislabeled.
    """
    if str(target).upper() != "ATE":
        raise ValueError("This DML contract currently supports target='ATE' only.")
    result = estimate_aipw(df, treatment, outcome, covariates, target="ATE", **kwargs)
    result["method_used"] = "cross-fitted DML (orthogonal ATE score)"
    result["diagnostics"]["estimand"] = "ATE"
    result["diagnostics"]["warnings"].append(
        "DML uses cross-fitted nuisance models and an orthogonal score; it does not prove identification assumptions."
    )
    return result


class AIPWEstimator(CausalMethod):
    name = "Augmented Inverse Probability Weighting"
    description = "Cross-fitted doubly robust AIPW for binary-treatment ATE, ATT, or ATC."
    assumptions = ["Binary treatment and numeric outcome", "Conditional exchangeability", "Consistency", "Positivity"]

    def validate_assumptions(self, df, variables):
        required = [variables.treatment_variable, variables.outcome_variable,
                    *(getattr(variables, "confounders", None) or getattr(variables, "covariates", None) or [])]
        missing = sorted(self.check_missing(df, [x for x in required if x]))
        return {"assumptions_valid": not missing, "failed_assumptions": missing,
                "warnings": ["Identification assumptions are advisory and are not proven by this check."],
                "suggestions": []}

    def estimate_effect(self, df, variables, query=None, **kwargs):
        return estimate_aipw(df, variables.treatment_variable, variables.outcome_variable,
                             list(getattr(variables, "confounders", None) or getattr(variables, "covariates", None) or []),
                             **kwargs)


class DMLEstimator(AIPWEstimator):
    name = "Double Machine Learning"
    description = "Cross-fitted orthogonal-score DML for the binary-treatment ATE."

    def estimate_effect(self, df, variables, query=None, **kwargs):
        return estimate_dml(df, variables.treatment_variable, variables.outcome_variable,
                            list(getattr(variables, "confounders", None) or getattr(variables, "covariates", None) or []),
                            **kwargs)
