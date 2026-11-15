"""A narrow, canonical TMLE for a point-treatment binary outcome.

This is the binary-outcome ATE construction in the targeted-learning paper:
an initial outcome regression and propensity score are updated through the
clever covariate, and the plug-in contrast is returned with its influence
curve.  Continuous, survival and longitudinal TMLE are intentionally not
claimed by this module.
"""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy import stats, optimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from cais.methods.causal_method import CausalMethod


def estimate_tmle(df: pd.DataFrame, treatment: str, outcome: str,
                  covariates: List[str], n_splits: int = 5,
                  propensity_clip: float = .01, **_: Any) -> Dict[str, Any]:
    required = [treatment, outcome, *covariates]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    dat = df[required].dropna().copy()
    if len(dat) < 20:
        raise ValueError("TMLE requires at least 20 complete observations.")
    w = dat[treatment].astype(int).to_numpy()
    y = dat[outcome].astype(int).to_numpy()
    if set(w) != {0, 1} or not set(y).issubset({0, 1}):
        raise ValueError("This TMLE route requires binary treatment and binary outcome coded 0/1.")
    x = pd.get_dummies(dat[covariates], dtype=float).to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("Covariates must be finite after encoding.")
    folds = min(int(n_splits), int(np.bincount(w, minlength=2).min()))
    if folds < 2:
        raise ValueError("At least two observations per arm are required for cross-fitting.")
    q0 = np.empty(len(dat)); q1 = np.empty(len(dat)); g = np.empty(len(dat))
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=0).split(x, w):
        gm = LogisticRegression(max_iter=1000).fit(x[train], w[train])
        g[test] = np.clip(gm.predict_proba(x[test])[:, 1], propensity_clip, 1-propensity_clip)
        for arm, target in ((0, q0), (1, q1)):
            idx = train[w[train] == arm]
            qm = LogisticRegression(max_iter=1000).fit(x[idx], y[idx])
            target[test] = np.clip(qm.predict_proba(x[test])[:, 1], 1e-6, 1-1e-6)
    # Estimate the one-dimensional fluctuation parameter with a binomial GLM
    # and offset logit(Q).  The score is the canonical TMLE clever covariate.
    h = w / g - (1-w) / (1-g)
    q_obs = np.where(w == 1, q1, q0)
    def score(e):
        return float(np.sum(h * (y - expit(logit(q_obs) + e*h))))
    if score(-20.0) * score(20.0) <= 0:
        epsilon = float(optimize.brentq(score, -20.0, 20.0))
    else:
        epsilon = float(optimize.minimize_scalar(lambda e: score(e) ** 2,
                                                 bounds=(-20, 20), method="bounded").x)
    q0_star = expit(logit(q0) + epsilon * (-1/(1-g)))
    q1_star = expit(logit(q1) + epsilon * (1/g))
    psi_i = q1_star - q0_star + w*(y-q1_star)/g - (1-w)*(y-q0_star)/(1-g)
    effect = float(np.mean(q1_star - q0_star))
    ic = psi_i - effect
    se = float(np.std(ic, ddof=1) / np.sqrt(len(ic)))
    return {
        "effect_estimate": effect, "standard_error": se,
        "confidence_interval": [effect-1.96*se, effect+1.96*se],
        "p_value": float(2*stats.norm.sf(abs(effect/se))) if se else None,
        "method_used": "cross-fitted TMLE (binary ATE)",
        "diagnostics": {"n_complete": len(dat), "epsilon": epsilon,
                        "propensity_min": float(g.min()), "propensity_max": float(g.max()),
                        "estimand": "ATE", "outcome_support": [0, 1],
                        "warnings": ["Exchangeability, consistency and positivity remain advisory assumptions."]},
        "parameters": {"n_splits": folds, "propensity_clip": propensity_clip,
                       "covariates": list(covariates)},
    }


class TMLEEstimator(CausalMethod):
    name = "Targeted Maximum Likelihood Estimation"
    description = "Cross-fitted canonical TMLE for a binary-treatment, binary-outcome ATE."
    assumptions = ["Binary treatment and outcome", "Conditional exchangeability", "Consistency", "Positivity"]
    def validate_assumptions(self, df, variables):
        req = [variables.treatment_variable, variables.outcome_variable,
               *(getattr(variables, "confounders", None) or getattr(variables, "covariates", None) or [])]
        missing = sorted(self.check_missing(df, [x for x in req if x]))
        return {"assumptions_valid": not missing, "failed_assumptions": missing,
                "warnings": ["Substantive identification assumptions are advisory."], "suggestions": []}
    def estimate_effect(self, df, variables, query=None, **kwargs):
        return estimate_tmle(df, variables.treatment_variable, variables.outcome_variable,
                             list(getattr(variables, "confounders", None) or getattr(variables, "covariates", None) or []), **kwargs)
