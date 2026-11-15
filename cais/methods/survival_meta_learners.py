"""Meta-learners for RMST effects with right-censored outcomes.

The target is the conditional restricted mean survival-time contrast

    tau_RMST(x; tau) = E[min(T(1), tau) - min(T(0), tau) | X=x].

This is not a realized individual treatment effect.  The implementation uses
cross-fitted inverse-censoring-probability weighted (IPCW) pseudo-outcomes for
the default forest learner.  Cox-PH and scikit-survival random-survival-forest
learners are optional because their dependencies are not part of the minimal
CAIS installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

from cais.methods.causal_method import CausalMethod


SUPPORTED_SURVIVAL_LEARNERS = {"S", "T", "X"}
SUPPORTED_BASE_LEARNERS = {"ipcw_rf", "random_survival_forest", "cox_ph"}


def _diagnostic(status: str, detail: str, **details: Any) -> Dict[str, Any]:
    return {"status": status, "detail": detail, **details}


def validate_survival_assumptions(
    df: pd.DataFrame,
    treatment: str,
    duration: str,
    event: str,
    covariates: List[str],
    horizon: Optional[float] = None,
    base_learner: str = "ipcw_rf",
    llm: Any = None,
    dataset_description: Optional[str] = None,
) -> Dict[str, Any]:
    """Separate computable survival diagnostics from LLM-only advisories.

    Diagnostics can falsify structural/data conditions but do not prove
    exchangeability or independent censoring.  LLM advisories are explicitly
    non-blocking and return ``insufficient_evidence`` when context is lacking.
    """
    testable: Dict[str, Any] = {}
    advisory: Dict[str, Any] = {}
    required = [treatment, duration, event, *covariates]
    missing = [column for column in required if not column or column not in df.columns]
    if missing:
        testable["required_columns"] = _diagnostic(
            "evidence_against", f"Missing survival columns: {missing}", missing=missing
        )
        return {"testable_diagnostics": testable, "advisory_assessments": advisory,
                "blocking_failures": ["required_columns"], "advisory_only": True}

    data = df[required].dropna()
    if data.empty:
        testable["complete_cases"] = _diagnostic("evidence_against", "No complete survival rows remain.")
        return {"testable_diagnostics": testable, "advisory_assessments": advisory,
                "blocking_failures": ["complete_cases"], "advisory_only": True}

    w = data[treatment].to_numpy()
    d = data[duration].to_numpy(dtype=float)
    e = data[event].to_numpy()
    testable["treatment_binary"] = _diagnostic(
        "supported" if set(pd.unique(w)).issubset({0, 1}) and len(pd.unique(w)) == 2 else "evidence_against",
        "Treatment has two observed levels coded 0/1.", levels=[str(v) for v in pd.unique(w)]
    )
    testable["event_binary"] = _diagnostic(
        "supported" if set(pd.unique(e)).issubset({0, 1}) else "evidence_against",
        "Event indicator is coded 0/1.", levels=[str(v) for v in pd.unique(e)]
    )
    tau = float(horizon) if horizon is not None else float(np.quantile(d, 0.9))
    testable["rmst_horizon_support"] = _diagnostic(
        "supported" if tau > 0 and tau <= float(np.max(d)) else "evidence_against",
        "RMST horizon lies within observed follow-up support.", horizon=tau, max_followup=float(np.max(d))
    )
    counts = {str(level): int((w == level).sum()) for level in [0, 1]}
    testable["treatment_arm_counts"] = _diagnostic(
        "supported" if min(counts.values()) >= 10 else "evidence_against",
        "Each treatment arm has at least 10 complete observations.", counts=counts
    )
    censor_g = _censoring_survival_at(np.asarray([tau]), d, e)[0]
    testable["censoring_support"] = _diagnostic(
        "supported" if censor_g >= 0.05 else "evidence_against",
        "Estimated marginal censoring survival remains above 0.05 at the horizon.",
        censoring_survival_at_horizon=float(censor_g),
    )
    testable["event_counts"] = _diagnostic(
        "supported" if int(e.sum()) >= 10 else "evidence_against",
        "At least 10 events are observed overall.", events=int(e.sum()), censored=int((1 - e).sum())
    )

    if base_learner == "cox_ph":
        try:
            from lifelines import CoxPHFitter
            from lifelines.statistics import proportional_hazard_test
            x_frame = pd.get_dummies(data[covariates], drop_first=False, dtype=float)
            x_frame["__duration__"] = d
            x_frame["__event__"] = e
            x_frame[treatment] = w
            cox = CoxPHFitter().fit(x_frame, "__duration__", "__event__")
            ph = proportional_hazard_test(cox, x_frame, time_transform="rank")
            p_value = float(ph.summary["p"].min())
            testable["proportional_hazards"] = _diagnostic(
                "supported" if p_value >= 0.05 else "evidence_against",
                "Schoenfeld-residual proportional-hazards diagnostic.", p_value=p_value,
                note="A non-significant test does not prove proportional hazards."
            )
        except ImportError:
            testable["proportional_hazards"] = _diagnostic(
                "insufficient_evidence", "lifelines is not installed; PH diagnostic unavailable."
            )
        except Exception as exc:
            testable["proportional_hazards"] = _diagnostic(
                "insufficient_evidence", f"PH diagnostic could not be computed: {exc}"
            )

    # Reuse CAIS's existing structured LLM argumentation mechanism.  These
    # assumptions are not converted into hard failures by this function.
    try:
        from cais.models import AssumptionVariables
        from cais.methods.pre_model_assumption_utils import _llm_argue_assumption
        vars = AssumptionVariables(
            df=data, treatment=treatment, outcome=duration, covariates=list(covariates),
            dataset_description=dataset_description,
            variables_summary={"duration": duration, "event": event, "horizon": tau},
        )
        for name, definition in {
            "Consistency and treatment version": "The observed treatment corresponds to a well-defined intervention version.",
            "Conditional exchangeability": "Given the measured pre-treatment covariates, treatment assignment is independent of both potential survival outcomes.",
            "Conditional independent censoring": "Conditional on treatment and measured covariates, censoring is independent of the counterfactual event process.",
            "No competing risks": "The event indicator represents a single target event rather than conflating competing event types with ordinary censoring.",
            "Time origin and RMST horizon": "The time origin and the selected RMST horizon have a substantive study-design interpretation.",
        }.items():
            result = _llm_argue_assumption(name, definition, vars, llm)
            status = "supported" if result.passed is True else "evidence_against" if result.passed is False else "insufficient_evidence"
            advisory[name] = {"status": status, "reasoning": result.reasoning, "details": result.details}
    except Exception as exc:
        advisory["llm_assessment"] = {
            "status": "insufficient_evidence", "reasoning": f"LLM advisory unavailable: {exc}"
        }

    blocking = [name for name, result in testable.items() if result["status"] == "evidence_against"]
    return {
        "testable_diagnostics": testable,
        "advisory_assessments": advisory,
        "blocking_failures": blocking,
        "advisory_only": True,
    }


def _design(df: pd.DataFrame, covariates: List[str]) -> np.ndarray:
    if not covariates:
        raise ValueError("Survival meta-learners require pre-treatment covariates.")
    matrix = pd.get_dummies(df[covariates], drop_first=False, dtype=float)
    values = matrix.to_numpy(dtype=float)
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Survival covariates must form a finite design matrix.")
    return values


def _prepare_data(
    df: pd.DataFrame,
    treatment: str,
    duration: str,
    event: str,
    covariates: List[str],
    horizon: Optional[float],
):
    required = [treatment, duration, event, *covariates]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Survival columns are missing: {missing}")
    data = df[required].dropna().copy()
    if len(data) < 30:
        raise ValueError("Survival meta-learners require at least 30 complete observations.")
    w = data[treatment].to_numpy(dtype=int)
    if not np.isin(w, [0, 1]).all() or np.unique(w).size != 2:
        raise ValueError("Survival meta-learners require a binary treatment coded 0/1.")
    durations = data[duration].to_numpy(dtype=float)
    events = data[event].to_numpy(dtype=int)
    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("Survival durations must be finite and non-negative.")
    if not np.isin(events, [0, 1]).all():
        raise ValueError("The event indicator must be binary: 1=event, 0=right censored.")
    tau = float(horizon) if horizon is not None else float(np.quantile(durations, 0.9))
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("RMST horizon must be finite and strictly positive.")
    if tau > float(np.max(durations)):
        raise ValueError("RMST horizon cannot exceed the largest observed follow-up time.")
    return data, _design(data, covariates), w, durations, events, tau


def _censoring_survival_at(times: np.ndarray, durations: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Kaplan--Meier estimate of G(t)=P(C>=t), fitted on the training fold."""
    order = np.argsort(durations)
    sorted_t = durations[order]
    censor = (1 - events)[order]
    unique = np.unique(sorted_t)
    at_risk = len(sorted_t)
    survival = 1.0
    values = []
    for t in unique:
        at_time = sorted_t == t
        censor_events = int(censor[at_time].sum())
        values.append((float(t), float(survival)))
        if at_risk:
            survival *= max(0.0, 1.0 - censor_events / at_risk)
        at_risk -= int(at_time.sum())
    if not values:
        return np.ones(len(times))
    knot_t = np.asarray([item[0] for item in values])
    knot_g = np.asarray([item[1] for item in values])
    indices = np.searchsorted(knot_t, times, side="right") - 1
    return np.clip(np.where(indices < 0, 1.0, knot_g[np.maximum(indices, 0)]), 1e-3, 1.0)


def _ipcw_rmst_pseudo_outcome(
    durations: np.ndarray, events: np.ndarray, horizon: float, censor_durations=None, censor_events=None
) -> np.ndarray:
    """Approximate integral of I(U>=t)/G(t) over [0,tau]."""
    censor_durations = durations if censor_durations is None else censor_durations
    censor_events = events if censor_events is None else censor_events
    grid = np.linspace(0.0, horizon, 101)
    dt = grid[1] - grid[0]
    g = _censoring_survival_at(grid, censor_durations, censor_events)
    at_risk = durations[:, None] >= grid[None, :]
    return np.sum(at_risk * (dt / g[None, :]), axis=1)


class IPCWRMSTLearner:
    """Random-forest regression on a cross-fitted IPCW RMST pseudo-outcome."""

    def __init__(self, random_state=0, n_estimators=200, min_samples_leaf=5):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            random_state=random_state, n_jobs=1,
        )

    def fit(self, x, pseudo_outcome, **kwargs):
        self.model.fit(x, pseudo_outcome)
        return self

    def predict_rmst(self, x):
        return self.model.predict(x)


class RandomSurvivalForestRMSTLearner:
    """Optional scikit-survival RSF adapter, integrated on the RMST scale."""

    def __init__(self, random_state=0, n_estimators=200, min_samples_leaf=5):
        try:
            from sksurv.ensemble import RandomSurvivalForest
        except ImportError as exc:
            raise ImportError(
                "base_learner='random_survival_forest' requires scikit-survival."
            ) from exc
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            random_state=random_state, n_jobs=1,
        )
        self.horizon = None

    def fit(self, x, pseudo_outcome=None, durations=None, events=None, horizon=None, **kwargs):
        from sksurv.util import Surv
        self.horizon = float(horizon)
        self.model.fit(x, Surv.from_arrays(events.astype(bool), durations))
        return self

    def predict_rmst(self, x):
        curves = self.model.predict_survival_function(x, return_array=False)
        grid = np.linspace(0.0, self.horizon, 101)
        return np.asarray([
            np.trapz(np.clip(curve(grid), 0.0, 1.0), grid) for curve in curves
        ])


class CoxPHRMSTLearner:
    """Optional lifelines Cox proportional-hazards adapter on the RMST scale."""

    def __init__(self):
        try:
            from lifelines import CoxPHFitter
        except ImportError as exc:
            raise ImportError("base_learner='cox_ph' requires lifelines.") from exc
        self.model = CoxPHFitter()
        self.horizon = None
        self.columns = None

    def fit(self, x, pseudo_outcome=None, durations=None, events=None, horizon=None, **kwargs):
        self.horizon = float(horizon)
        self.columns = [f"x{i}" for i in range(x.shape[1])]
        frame = pd.DataFrame(x, columns=self.columns)
        frame["__duration__"] = durations
        frame["__event__"] = events
        self.model.fit(frame, duration_col="__duration__", event_col="__event__")
        return self

    def predict_rmst(self, x):
        frame = pd.DataFrame(x, columns=self.columns)
        grid = np.linspace(0.0, self.horizon, 101)
        curves = self.model.predict_survival_function(frame, times=grid)
        return np.asarray([
            np.trapz(np.clip(curves.iloc[:, i].to_numpy(), 0.0, 1.0), grid)
            for i in range(curves.shape[1])
        ])


def _factory(kind: str, random_state: int):
    if kind == "ipcw_rf":
        return IPCWRMSTLearner(random_state=random_state)
    if kind == "random_survival_forest":
        return RandomSurvivalForestRMSTLearner(random_state=random_state)
    if kind == "cox_ph":
        return CoxPHRMSTLearner()
    raise ValueError(f"Unknown survival base learner {kind!r}; choose {sorted(SUPPORTED_BASE_LEARNERS)}.")


def _fit_predict(base_kind, x_train, w_train, duration_train, event_train,
                 x_test, horizon, learner_type, random_state):
    pseudo = _ipcw_rmst_pseudo_outcome(duration_train, event_train, horizon)

    # Shared pseudo-outcomes are valid for the IPCW forest. Survival-native
    # learners use observed durations/events and estimate survival directly.
    def fit_model(x_fit, d_fit, e_fit, pseudo_fit, seed):
        model = _factory(base_kind, seed)
        if base_kind == "ipcw_rf":
            model.fit(x_fit, pseudo_fit)
        else:
            model.fit(x_fit, durations=d_fit, events=e_fit, horizon=horizon)
        return model

    if learner_type == "S":
        model = fit_model(
            np.column_stack([x_train, w_train]), duration_train, event_train,
            pseudo, random_state,
        )
        x1 = np.column_stack([x_test, np.ones(len(x_test))])
        x0 = np.column_stack([x_test, np.zeros(len(x_test))])
        return model.predict_rmst(x1), model.predict_rmst(x0)

    mask0, mask1 = w_train == 0, w_train == 1
    p0 = pseudo[mask0]
    p1 = pseudo[mask1]
    m0 = fit_model(x_train[mask0], duration_train[mask0], event_train[mask0], p0, random_state)
    m1 = fit_model(x_train[mask1], duration_train[mask1], event_train[mask1], p1, random_state + 1)
    mu0 = m0.predict_rmst(x_test)
    mu1 = m1.predict_rmst(x_test)
    if learner_type == "T":
        return mu1, mu0

    # X-learner on the RMST scale: impute treatment contrasts from the two
    # arm-specific response surfaces and blend using empirical propensity.
    prop = np.clip(np.mean(w_train), 0.01, 0.99)
    tau1 = p1 - m0.predict_rmst(x_train[mask1])
    tau0 = m1.predict_rmst(x_train[mask0]) - p0
    effect0 = IPCWRMSTLearner(random_state=random_state + 2)
    effect1 = IPCWRMSTLearner(random_state=random_state + 3)
    effect0.fit(x_train[mask0], tau0)
    effect1.fit(x_train[mask1], tau1)
    tau = (1 - prop) * effect1.predict_rmst(x_test) + prop * effect0.predict_rmst(x_test)
    return mu0 + tau, mu0


def estimate_survival_effect(
    df: pd.DataFrame,
    treatment: str,
    duration: str,
    event: str,
    covariates: List[str],
    learner_type: str = "T",
    base_learner: str = "ipcw_rf",
    horizon: Optional[float] = None,
    n_splits: int = 5,
    random_state: int = 0,
    llm: Any = None,
    dataset_description: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    learner = str(learner_type).upper().replace("-LEARNER", "")
    base_learner = str(base_learner).lower()
    if learner not in SUPPORTED_SURVIVAL_LEARNERS:
        raise ValueError(f"learner_type must be one of {sorted(SUPPORTED_SURVIVAL_LEARNERS)}.")
    if base_learner not in SUPPORTED_BASE_LEARNERS:
        raise ValueError(f"base_learner must be one of {sorted(SUPPORTED_BASE_LEARNERS)}.")
    data, x, w, durations, events, tau = _prepare_data(
        df, treatment, duration, event, list(covariates or []), horizon
    )
    assumption_report = validate_survival_assumptions(
        df=df, treatment=treatment, duration=duration, event=event,
        covariates=list(covariates or []), horizon=tau,
        base_learner=base_learner, llm=llm,
        dataset_description=dataset_description,
    )
    folds = min(int(n_splits), int(np.bincount(w, minlength=2).min()))
    if folds < 2:
        raise ValueError("Survival cross-fitting requires at least two observations per treatment arm.")
    mu0 = np.empty(len(data), dtype=float)
    mu1 = np.empty(len(data), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    for train, test in splitter.split(x, w):
        fold_mu1, fold_mu0 = _fit_predict(
            base_learner, x[train], w[train], durations[train], events[train],
            x[test], tau, learner, random_state,
        )
        mu0[test], mu1[test] = fold_mu0, fold_mu1
    cate = mu1 - mu0
    if not np.isfinite(cate).all():
        raise ValueError("Survival meta-learner produced non-finite RMST predictions.")
    predictions = [
        {
            "row_index": str(index),
            "rmst_cate_prediction": float(effect),
            "predicted_rmst_0": float(y0),
            "predicted_rmst_1": float(y1),
        }
        for index, effect, y0, y1 in zip(data.index, cate, mu0, mu1)
    ]
    warnings = [
        "Row-level values are conditional RMST predictions, not realized ITEs.",
        "Exchangeability, consistency, positivity and conditional independent censoring are substantive advisories.",
        "Competing risks are not supported by this route; use a competing-risks-specific estimand and estimator.",
    ]
    if base_learner == "cox_ph":
        warnings.append("Cox-PH requires proportional hazards; this assumption is not testable from the RMST contrast alone.")
    return {
        "effect_estimate": float(np.mean(cate)),
        "standard_error": None,
        "confidence_interval": [None, None],
        "p_value": None,
        "method_used": f"{learner}-learner RMST ({base_learner})",
        "individual_effect_predictions": predictions,
        "diagnostics": {
            "estimand": "conditional RMST contrast E[min(T(1),tau)-min(T(0),tau)|X=x]",
            "not_identified_as": "realized individual effect Y_i(1)-Y_i(0)",
            "horizon": tau,
            "n_complete": len(data),
            "event_fraction": float(events.mean()),
            "censoring_fraction": float(1 - events.mean()),
            "n_splits": folds,
            "base_learner": base_learner,
            "warnings": warnings,
            "assumption_report": assumption_report,
        },
    }


def route_survival_base_learner(df: pd.DataFrame, preferred: str = "auto") -> str:
    """Select a base learner from available dependencies and observed schema."""
    if preferred != "auto":
        if preferred not in SUPPORTED_BASE_LEARNERS:
            raise ValueError(f"Unknown survival base learner: {preferred}")
        return preferred
    try:
        import sksurv  # noqa: F401
        return "random_survival_forest"
    except ImportError:
        return "ipcw_rf"


def _route_kwargs(treatment, covariates, duration_variable, event_variable,
                  rmst_horizon, learner_type, **kwargs):
    return estimate_survival_effect(
        treatment=treatment, duration=duration_variable, event=event_variable,
        covariates=covariates, horizon=rmst_horizon,
        learner_type=learner_type, **kwargs,
    )


def estimate_s_learner_rmst(df, treatment, outcome, covariates,
                            duration_variable=None, event_variable=None,
                            rmst_horizon=None, **kwargs):
    return _route_kwargs(
        treatment, covariates, duration_variable, event_variable,
        rmst_horizon, "S", df=df, **kwargs
    )


def estimate_t_learner_rmst(df, treatment, outcome, covariates,
                            duration_variable=None, event_variable=None,
                            rmst_horizon=None, **kwargs):
    return _route_kwargs(
        treatment, covariates, duration_variable, event_variable,
        rmst_horizon, "T", df=df, **kwargs
    )


def estimate_x_learner_rmst(df, treatment, outcome, covariates,
                            duration_variable=None, event_variable=None,
                            rmst_horizon=None, **kwargs):
    return _route_kwargs(
        treatment, covariates, duration_variable, event_variable,
        rmst_horizon, "X", df=df, **kwargs
    )


@dataclass
class SurvivalMetaLearner(CausalMethod):
    learner_type: str = "T"
    base_learner: str = "ipcw_rf"

    name = "Survival Meta-Learner RMST"
    description = "Cross-fitted S/T/X learner for conditional restricted mean survival-time effects."
    assumptions = [
        "Binary treatment and right-censored single-event outcome",
        "Consistency and no interference",
        "Conditional exchangeability given pre-treatment covariates",
        "Positivity in treatment and censoring support",
        "Conditional independent censoring given covariates and treatment",
        "Pre-specified RMST horizon within observed follow-up support",
    ]

    def validate_assumptions(self, df, variables) -> Dict[str, Any]:
        required = [
            getattr(variables, "treatment_variable", None),
            getattr(variables, "duration_variable", None),
            getattr(variables, "event_variable", None),
            *(getattr(variables, "confounders", None) or []),
        ]
        missing = [column for column in required if not column or column not in df.columns]
        return {
            "assumptions_valid": not missing,
            "failed_assumptions": missing,
            "warnings": [
                "Censoring independence and causal exchangeability are advisory assumptions; "
                "predictive performance does not prove them."
            ],
            "suggestions": [],
        }

    def estimate_effect(self, df, variables, query=None, **kwargs):
        return estimate_survival_effect(
            df=df,
            treatment=variables.treatment_variable,
            duration=variables.duration_variable,
            event=variables.event_variable,
            covariates=list(getattr(variables, "confounders", None) or []),
            learner_type=self.learner_type,
            base_learner=kwargs.pop("base_learner", self.base_learner),
            horizon=kwargs.pop("horizon", getattr(variables, "rmst_horizon", None)),
            **kwargs,
        )


class SurvivalSLearner(SurvivalMetaLearner):
    name = "S-Learner RMST"
    def __init__(self, base_learner="ipcw_rf"): super().__init__("S", base_learner)


class SurvivalTLearner(SurvivalMetaLearner):
    name = "T-Learner RMST"
    def __init__(self, base_learner="ipcw_rf"): super().__init__("T", base_learner)


class SurvivalXLearner(SurvivalMetaLearner):
    name = "X-Learner RMST"
    def __init__(self, base_learner="ipcw_rf"): super().__init__("X", base_learner)
