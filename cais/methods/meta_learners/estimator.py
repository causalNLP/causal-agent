"""Cross-fitted meta-learners for heterogeneous treatment effects.

These estimators target the conditional average treatment effect
``tau(x) = E[Y(1)-Y(0) | X=x]`` under conditional exchangeability,
consistency and positivity.  Per-row outputs are predictions of that CATE,
not identified realizations of the individual causal contrast
``Y_i(1)-Y_i(0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from cais.methods.causal_method import CausalMethod


SUPPORTED_LEARNERS = {"S", "T", "X", "R"}


def _default_regressor(random_state: int):
    return RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=1,
    )


def _prepare_data(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if not covariates:
        raise ValueError("Meta-learners require at least one pre-treatment covariate.")
    required = [treatment, outcome, *covariates]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Meta-learner columns are missing: {missing}")

    data = df[required].dropna().copy()
    if len(data) < 20:
        raise ValueError("Meta-learners require at least 20 complete observations.")
    treatment_values = set(data[treatment].unique())
    if not treatment_values.issubset({0, 1, False, True}) or len(treatment_values) != 2:
        raise ValueError("Meta-learners require binary treatment coded 0/1 with both arms present.")
    if not pd.api.types.is_numeric_dtype(data[outcome]):
        raise ValueError("Meta-learners currently require a numeric outcome.")

    design = pd.get_dummies(data[covariates], drop_first=False, dtype=float)
    if design.shape[1] == 0 or not np.isfinite(design.to_numpy()).all():
        raise ValueError("Covariates could not be converted to a finite design matrix.")
    return (
        data,
        design.to_numpy(dtype=float),
        data[treatment].to_numpy(dtype=int),
        data[outcome].to_numpy(dtype=float),
    )


def _folds(w: np.ndarray, n_splits: int, random_state: int):
    smallest_arm = int(np.bincount(w, minlength=2).min())
    folds = min(int(n_splits), smallest_arm)
    if folds < 2:
        raise ValueError("Cross-fitting requires at least two observations in each treatment arm.")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)


def _propensity_predictions(model, x_train, w_train, x_test, clip: float):
    fitted = clone(model).fit(x_train, w_train)
    raw = fitted.predict_proba(x_test)[:, 1]
    return np.clip(raw, clip, 1.0 - clip)


def _s_learner(train_x, train_w, train_y, test_x, outcome_model):
    fitted = clone(outcome_model).fit(np.column_stack([train_x, train_w]), train_y)
    mu1 = fitted.predict(np.column_stack([test_x, np.ones(len(test_x))]))
    mu0 = fitted.predict(np.column_stack([test_x, np.zeros(len(test_x))]))
    return mu1 - mu0, mu0, mu1, None


def _t_learner(train_x, train_w, train_y, test_x, outcome_model):
    model0 = clone(outcome_model).fit(train_x[train_w == 0], train_y[train_w == 0])
    model1 = clone(outcome_model).fit(train_x[train_w == 1], train_y[train_w == 1])
    mu0 = model0.predict(test_x)
    mu1 = model1.predict(test_x)
    return mu1 - mu0, mu0, mu1, None


def _x_learner(
    train_x, train_w, train_y, test_x, outcome_model, effect_model,
    propensity_model, propensity_clip,
):
    model0 = clone(outcome_model).fit(train_x[train_w == 0], train_y[train_w == 0])
    model1 = clone(outcome_model).fit(train_x[train_w == 1], train_y[train_w == 1])
    imputed_treated = train_y[train_w == 1] - model0.predict(train_x[train_w == 1])
    imputed_control = model1.predict(train_x[train_w == 0]) - train_y[train_w == 0]
    tau1 = clone(effect_model).fit(train_x[train_w == 1], imputed_treated)
    tau0 = clone(effect_model).fit(train_x[train_w == 0], imputed_control)
    propensity = _propensity_predictions(
        propensity_model, train_x, train_w, test_x, propensity_clip
    )
    # With high treatment propensity, tau0 uses the relatively abundant treated
    # response model; with low propensity, tau1 uses the abundant controls.
    cate = propensity * tau0.predict(test_x) + (1.0 - propensity) * tau1.predict(test_x)
    mu0 = model0.predict(test_x)
    mu1 = model1.predict(test_x)
    return cate, mu0, mu1, propensity


def _r_learner(
    train_x, train_w, train_y, test_x, outcome_model, effect_model,
    propensity_model, propensity_clip, random_state,
):
    # Cross-fit both nuisance functions inside the outer training fold. This
    # prevents the R-loss pseudo-outcome from reusing in-sample nuisance fits.
    m_hat = np.empty(len(train_y), dtype=float)
    propensity_train = np.empty(len(train_y), dtype=float)
    nuisance_folds = _folds(train_w, 5, random_state)
    for nuisance_train, nuisance_test in nuisance_folds.split(train_x, train_w):
        outcome_nuisance = clone(outcome_model).fit(
            train_x[nuisance_train], train_y[nuisance_train]
        )
        m_hat[nuisance_test] = outcome_nuisance.predict(train_x[nuisance_test])
        propensity_train[nuisance_test] = _propensity_predictions(
            propensity_model,
            train_x[nuisance_train],
            train_w[nuisance_train],
            train_x[nuisance_test],
            propensity_clip,
        )
    treatment_residual = train_w - propensity_train
    usable = np.abs(treatment_residual) > 1e-6
    pseudo_outcome = (train_y[usable] - m_hat[usable]) / treatment_residual[usable]
    weights = np.square(treatment_residual[usable])
    tau = clone(effect_model)
    try:
        tau.fit(train_x[usable], pseudo_outcome, sample_weight=weights)
    except TypeError as exc:
        raise ValueError("R-learner effect_model must support sample_weight.") from exc
    cate = tau.predict(test_x)
    propensity = _propensity_predictions(
        propensity_model, train_x, train_w, test_x, propensity_clip
    )
    # R-loss identifies tau directly. Potential-outcome predictions below are
    # a coherent decomposition around the marginal outcome regression, useful
    # for reporting but not separate nuisance fits.
    outcome_fit = clone(outcome_model).fit(train_x, train_y)
    marginal = outcome_fit.predict(test_x)
    mu0 = marginal - propensity * cate
    mu1 = marginal + (1.0 - propensity) * cate
    return cate, mu0, mu1, propensity


def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str],
    learner_type: str = "T",
    n_splits: int = 5,
    random_state: int = 0,
    propensity_clip: float = 0.01,
    outcome_model: Any = None,
    effect_model: Any = None,
    propensity_model: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Estimate cross-fitted CATE predictions with an S/T/X/R learner."""
    learner = str(learner_type).upper().replace("-LEARNER", "")
    if learner not in SUPPORTED_LEARNERS:
        raise ValueError(f"learner_type must be one of {sorted(SUPPORTED_LEARNERS)}.")
    if not 0 < float(propensity_clip) < 0.5:
        raise ValueError("propensity_clip must lie strictly between 0 and 0.5.")

    data, x, w, y = _prepare_data(df, treatment, outcome, list(covariates or []))
    if outcome_model is None:
        outcome_model = _default_regressor(random_state)
    if effect_model is None:
        effect_model = _default_regressor(random_state + 1)
    if propensity_model is None:
        propensity_model = LogisticRegression(max_iter=1000, random_state=random_state)

    cate = np.empty(len(data), dtype=float)
    mu0 = np.empty(len(data), dtype=float)
    mu1 = np.empty(len(data), dtype=float)
    propensity = np.full(len(data), np.nan, dtype=float)
    splitter = _folds(w, n_splits, random_state)
    for train, test in splitter.split(x, w):
        args = (x[train], w[train], y[train], x[test])
        if learner == "S":
            fold_cate, fold_mu0, fold_mu1, fold_ps = _s_learner(*args, outcome_model)
        elif learner == "T":
            fold_cate, fold_mu0, fold_mu1, fold_ps = _t_learner(*args, outcome_model)
        elif learner == "X":
            fold_cate, fold_mu0, fold_mu1, fold_ps = _x_learner(
                *args, outcome_model, effect_model, propensity_model, propensity_clip
            )
        else:
            fold_cate, fold_mu0, fold_mu1, fold_ps = _r_learner(
                *args,
                outcome_model,
                effect_model,
                propensity_model,
                propensity_clip,
                random_state,
            )
        cate[test], mu0[test], mu1[test] = fold_cate, fold_mu0, fold_mu1
        if fold_ps is not None:
            propensity[test] = fold_ps

    if not np.isfinite(cate).all():
        raise ValueError("Meta-learner produced non-finite CATE predictions.")
    if np.isnan(propensity).all():
        propensity = _propensity_predictions(propensity_model, x, w, x, propensity_clip)

    overlap_warning = bool(
        np.mean((propensity <= propensity_clip + 1e-12) | (propensity >= 1 - propensity_clip - 1e-12)) > 0.05
    )
    warnings = [
        "Per-row effects are cross-fitted CATE predictions, not identified realized ITEs.",
        "Conditional exchangeability, consistency and correct covariate timing are substantive advisory assumptions.",
    ]
    if overlap_warning:
        warnings.append("More than 5% of propensity predictions hit clipping bounds; overlap is weak.")

    predictions = [
        {
            "row_index": str(index),
            "cate_prediction": float(tau),
            "predicted_y0": float(y0),
            "predicted_y1": float(y1),
        }
        for index, tau, y0, y1 in zip(data.index, cate, mu0, mu1)
    ]
    return {
        "effect_estimate": float(np.mean(cate)),
        "standard_error": None,
        "confidence_interval": [None, None],
        "p_value": None,
        "method_used": f"{learner}-learner",
        "individual_effect_predictions": predictions,
        "diagnostics": {
            "estimand": "CATE tau(x); row-level values are individualized CATE predictions",
            "not_identified_as": "realized individual treatment effect Y_i(1)-Y_i(0)",
            "n_complete": int(len(data)),
            "n_splits": int(min(n_splits, np.bincount(w, minlength=2).min())),
            "treated_fraction": float(w.mean()),
            "cate_mean": float(np.mean(cate)),
            "cate_std": float(np.std(cate, ddof=1)),
            "cate_quantiles": {
                str(q): float(np.quantile(cate, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
            },
            "propensity_min": float(np.min(propensity)),
            "propensity_max": float(np.max(propensity)),
            "warnings": warnings,
        },
        "parameters": {
            "learner_type": learner,
            "covariates": list(covariates),
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "propensity_clip": float(propensity_clip),
        },
    }


def estimate_s_learner(*args, **kwargs):
    return estimate_effect(*args, learner_type="S", **kwargs)


def estimate_t_learner(*args, **kwargs):
    return estimate_effect(*args, learner_type="T", **kwargs)


def estimate_x_learner(*args, **kwargs):
    return estimate_effect(*args, learner_type="X", **kwargs)


def estimate_r_learner(*args, **kwargs):
    return estimate_effect(*args, learner_type="R", **kwargs)


@dataclass
class MetaLearner(CausalMethod):
    learner_type: str = "T"

    name = "Meta-Learner CATE"
    description = "Cross-fitted S/T/X/R meta-learner for conditional average treatment effects."
    assumptions = [
        "Binary treatment and numeric outcome",
        "Conditional exchangeability given pre-treatment covariates",
        "Consistency and no relevant interference",
        "Positivity over the covariate profiles where CATE is predicted",
    ]

    def __post_init__(self):
        super().__init__()
        self.learner_type = str(self.learner_type).upper()
        if self.learner_type not in SUPPORTED_LEARNERS:
            raise ValueError(f"Unsupported meta-learner {self.learner_type!r}.")

    def validate_assumptions(self, df, variables) -> Dict[str, Any]:
        covariates = list(getattr(variables, "confounders", None) or [])
        missing = [
            column for column in [
                getattr(variables, "treatment_variable", None),
                getattr(variables, "outcome_variable", None),
                *covariates,
            ]
            if not column or column not in df.columns
        ]
        return {
            "assumptions_valid": not missing,
            "failed_assumptions": missing,
            "warnings": [
                "Exchangeability, consistency and covariate timing are argued/advisory; "
                "they are not proven by predictive performance."
            ],
            "suggestions": [],
        }

    def estimate_effect(self, df, variables, query: Optional[str] = None, **kwargs):
        return estimate_effect(
            df=df,
            treatment=variables.treatment_variable,
            outcome=variables.outcome_variable,
            covariates=list(getattr(variables, "confounders", None) or []),
            learner_type=self.learner_type,
            **kwargs,
        )


class SLearner(MetaLearner):
    name = "S-Learner CATE"

    def __init__(self):
        super().__init__("S")


class TLearner(MetaLearner):
    name = "T-Learner CATE"

    def __init__(self):
        super().__init__("T")


class XLearner(MetaLearner):
    name = "X-Learner CATE"

    def __init__(self):
        super().__init__("X")


class RLearner(MetaLearner):
    name = "R-Learner CATE"

    def __init__(self):
        super().__init__("R")
