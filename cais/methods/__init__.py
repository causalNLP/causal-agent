"""
Causal inference methods for the cais module.

This package contains implementations of various causal inference methods
that can be selected and applied by the cais pipeline.
"""

from .causal_method import CausalMethod
from .propensity_score.matching import estimate_effect as psm_estimate_effect
from .propensity_score.weighting import estimate_effect as psw_estimate_effect
from .propensity_score.weighting import estimate_effect_trimmed_ipw
from .frontdoor_adjustment.estimator import estimate_effect as frontdoor_estimate_effect
from .instrumental_variable.estimator import estimate_effect as iv_estimate_effect
from .difference_in_differences.estimator import estimate_effect as did_estimate_effect
from .difference_in_differences.time_window import estimate_effect_time_window as did_time_window_estimate_effect
from .diff_in_means.estimator import estimate_effect as dim_estimate_effect
from .linear_regression.estimator import estimate_effect as lr_estimate_effect
from .backdoor_adjustment.estimator import estimate_effect as ba_estimate_effect
from .regression_discontinuity.estimator import estimate_effect as rdd_estimate_effect
from .regression_discontinuity.donut_estimator import estimate_effect as donut_rdd_estimate_effect
from .generalized_propensity_score.estimator import estimate_effect_gps
from .frontdoor_adjustment.estimator import estimate_effect as frontdoor_estimate_effect
from .synthetic_control.estimator import estimate_effect as synthetic_control_estimate_effect
from .fuzzy_regression_discontinuity.estimator import estimate_effect as fuzzy_rdd_estimate_effect
from .meta_learners.estimator import (
    estimate_s_learner,
    estimate_t_learner,
    estimate_x_learner,
    estimate_r_learner,
)
from .registry_adapters import (
    IPWEstimator,
    GeneralizedPropensityScoreEstimator,
    BackdoorAdjustmentEstimator,
    DifferenceInMeansEstimator,
)
from .survival_meta_learners import (
    estimate_s_learner_rmst,
    estimate_t_learner_rmst,
    estimate_x_learner_rmst,
)

# Mapping of method names to their implementation functions
METHOD_MAPPING = {
    "propensity_score_matching": psm_estimate_effect,
    "propensity_score_weighting": psw_estimate_effect,
    "trimmed_ipw": estimate_effect_trimmed_ipw,
    "frontdoor_adjustment": frontdoor_estimate_effect,
    "instrumental_variable": iv_estimate_effect,
    "difference_in_differences": did_estimate_effect,
    "difference_in_differences_time_window": did_time_window_estimate_effect,
    "regression_discontinuity_design": rdd_estimate_effect,
    "donut_rdd": donut_rdd_estimate_effect,
    "backdoor_adjustment": ba_estimate_effect,
    "linear_regression": lr_estimate_effect,
    "diff_in_means": dim_estimate_effect,
    "generalized_propensity_score": estimate_effect_gps,
    "frontdoor_adjustment": frontdoor_estimate_effect,
    "synthetic_control": synthetic_control_estimate_effect,
    "fuzzy_regression_discontinuity_design": fuzzy_rdd_estimate_effect,
    "s_learner": estimate_s_learner,
    "t_learner": estimate_t_learner,
    "x_learner": estimate_x_learner,
    "r_learner": estimate_r_learner,
    "s_learner_rmst": estimate_s_learner_rmst,
    "t_learner_rmst": estimate_t_learner_rmst,
    "x_learner_rmst": estimate_x_learner_rmst,
}

__all__ = [
    "CausalMethod",
    "psm_estimate_effect",
    "psw_estimate_effect",
    "estimate_effect_trimmed_ipw",
    "frontdoor_estimate_effect",
    "iv_estimate_effect",
    "did_estimate_effect",
    "did_time_window_estimate_effect",
    "rdd_estimate_effect",
    "donut_rdd_estimate_effect",
    "dim_estimate_effect",
    "lr_estimate_effect",
    "ba_estimate_effect",
    "METHOD_MAPPING",
    "estimate_effect_gps",
    "frontdoor_estimate_effect",
    "synthetic_control_estimate_effect",
    "fuzzy_rdd_estimate_effect",
    "estimate_s_learner",
    "estimate_t_learner",
    "estimate_x_learner",
    "estimate_r_learner",
    "IPWEstimator",
    "GeneralizedPropensityScoreEstimator",
    "BackdoorAdjustmentEstimator",
    "DifferenceInMeansEstimator",
    "EXECUTABLE_METHOD_IDS",
    "estimate_s_learner_rmst",
    "estimate_t_learner_rmst",
    "estimate_x_learner_rmst",
]
