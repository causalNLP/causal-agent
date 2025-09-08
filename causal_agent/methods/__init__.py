"""
Causal inference methods for the causal_agent module.

This package contains implementations of various causal inference methods
that can be selected and applied by the causal_agent pipeline.
"""

from .causal_method import CausalMethod
from .propensity_score.matching import estimate_effect as psm_estimate_effect
from .propensity_score.weighting import estimate_effect as psw_estimate_effect
from .instrumental_variable.estimator import estimate_effect as iv_estimate_effect
from .difference_in_differences.estimator import estimate_effect as did_estimate_effect
from .diff_in_means.estimator import estimate_effect as dim_estimate_effect
from .linear_regression.estimator import estimate_effect as lr_estimate_effect
from .backdoor_adjustment.estimator import estimate_effect as ba_estimate_effect
from .regression_discontinuity.estimator import estimate_effect as rdd_estimate_effect
from .generalized_propensity_score.estimator import estimate_effect_gps

# Mapping of method names to their implementation functions
METHOD_MAPPING = {
    "propensity_score_matching": psm_estimate_effect,
    "propensity_score_weighting": psw_estimate_effect,
    "instrumental_variable": iv_estimate_effect,
    "difference_in_differences": did_estimate_effect,
    "regression_discontinuity_design": rdd_estimate_effect,
    "backdoor_adjustment": ba_estimate_effect,
    "linear_regression": lr_estimate_effect,
    "diff_in_means": dim_estimate_effect,
    "generalized_propensity_score": estimate_effect_gps,
}

__all__ = [
    "CausalMethod",
    "psm_estimate_effect",
    "psw_estimate_effect",
    "iv_estimate_effect",
    "did_estimate_effect",
    "rdd_estimate_effect",
    "dim_estimate_effect",
    "lr_estimate_effect",
    "ba_estimate_effect",
    "METHOD_MAPPING",
    "estimate_effect_gps",
]
