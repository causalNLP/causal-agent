"""CausalMethod adapters for function-based estimators."""

from typing import Any, Dict, Optional

import pandas as pd

from cais.methods.causal_method import CausalMethod
from cais.models import Variables


class _FunctionEstimatorAdapter(CausalMethod):
    estimator_id: str
    estimator_function: Any

    def validate_assumptions(self, df: pd.DataFrame, variables: Variables) -> Dict[str, Any]:
        treatment = getattr(variables, "treatment_variable", None)
        outcome = getattr(variables, "outcome_variable", None)
        covariates = list(getattr(variables, "confounders", None)
                          or getattr(variables, "covariates", None) or [])
        required = [value for value in [treatment, outcome, *covariates] if value]
        missing = sorted(self.check_missing(df, required))
        return {
            "assumptions_valid": not missing,
            "failed_assumptions": [f"Missing required columns: {missing}"] if missing else [],
            "warnings": [
                "Causal identification assumptions are substantive and advisory; "
                "this check does not prove them."
            ],
            "suggestions": [],
        }

    def estimate_effect(self, df: pd.DataFrame, variables: Variables,
                        query: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        covariates = list(getattr(variables, "confounders", None)
                          or getattr(variables, "covariates", None) or [])
        return self.estimator_function(
            df=df,
            treatment=variables.treatment_variable,
            outcome=variables.outcome_variable,
            covariates=covariates,
            query=query,
            **kwargs,
        )


class IPWEstimator(_FunctionEstimatorAdapter):
    estimator_id = "propensity_score_weighting"
    name = "Inverse Probability Weighting"
    description = "Propensity-score weighting for binary-treatment ATE, ATT, or ATC targets."
    assumptions = ["Binary treatment", "Conditional exchangeability", "Positivity and adequate propensity model"]

    def __init__(self):
        from cais.methods.propensity_score.weighting import estimate_effect
        self.estimator_function = estimate_effect


class GeneralizedPropensityScoreEstimator(_FunctionEstimatorAdapter):
    estimator_id = "generalized_propensity_score"
    name = "Generalized Propensity Score"
    description = "GPS dose-response estimation for a continuous treatment."
    assumptions = ["Continuous treatment support", "Conditional exchangeability", "Correct GPS and outcome models"]

    def __init__(self):
        from cais.methods.generalized_propensity_score.estimator import estimate_effect_gps
        self.estimator_function = estimate_effect_gps


class BackdoorAdjustmentEstimator(_FunctionEstimatorAdapter):
    estimator_id = "backdoor_adjustment"
    name = "Backdoor Adjustment"
    description = "Outcome regression adjusting for a supplied backdoor adjustment set."
    assumptions = ["Covariates satisfy the backdoor criterion", "No unmeasured confounding", "Correct outcome model"]

    def __init__(self):
        from cais.methods.backdoor_adjustment.estimator import estimate_effect
        self.estimator_function = estimate_effect


class DifferenceInMeansEstimator(_FunctionEstimatorAdapter):
    estimator_id = "diff_in_means"
    name = "Difference in Means"
    description = "Unadjusted treated-versus-control mean difference."
    assumptions = ["Binary treatment", "No confounding or randomized assignment", "Well-defined outcome"]

    def __init__(self):
        from cais.methods.diff_in_means.estimator import estimate_effect
        self.estimator_function = estimate_effect


class TMLEEstimator(_FunctionEstimatorAdapter):
    estimator_id = "tmle"
    name = "Targeted Maximum Likelihood Estimation"
    description = "Canonical cross-fitted TMLE for binary-treatment, binary-outcome ATE."
    assumptions = ["Binary treatment and outcome", "Conditional exchangeability", "Consistency", "Positivity"]

    def __init__(self):
        from cais.methods.tmle import estimate_tmle
        self.estimator_function = estimate_tmle


__all__ = ["IPWEstimator", "GeneralizedPropensityScoreEstimator",
           "BackdoorAdjustmentEstimator", "DifferenceInMeansEstimator", "TMLEEstimator"]
