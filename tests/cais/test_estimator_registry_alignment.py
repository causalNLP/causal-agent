from cais.estimator_lib import Estimators
from cais.methods import METHOD_MAPPING


def test_every_executor_route_has_a_registered_estimator():
    estimators = Estimators()
    assert len(METHOD_MAPPING) == len(estimators)
    registered_names = estimators.names()
    assert "Inverse Probability Weighting" in registered_names
    assert "Generalized Propensity Score" in registered_names
    assert "Backdoor Adjustment" in registered_names
    assert "Difference in Means" in registered_names


def test_function_based_routes_are_not_orphans():
    estimators = Estimators()
    expected = {
        "propensity_score_weighting": "Inverse Probability Weighting",
        "generalized_propensity_score": "Generalized Propensity Score",
        "backdoor_adjustment": "Backdoor Adjustment",
        "diff_in_means": "Difference in Means",
    }
    assert set(expected).issubset(METHOD_MAPPING)
    assert set(expected.values()).issubset(estimators.names())
