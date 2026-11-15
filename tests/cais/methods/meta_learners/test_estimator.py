import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from cais.estimator_lib import Estimators
from cais.methods import METHOD_MAPPING
from cais.methods.meta_learners.estimator import (
    RLearner,
    SLearner,
    TLearner,
    XLearner,
    estimate_effect,
)
from cais.models import Variables


def heterogeneous_data(n=500, seed=41):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    propensity = 1 / (1 + np.exp(-0.5 * x1 + 0.3 * x2))
    treatment = rng.binomial(1, propensity)
    true_cate = 1.0 + 1.5 * x1
    outcome = 2.0 + x1 + x2 + treatment * true_cate + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "w": treatment, "y": outcome}), true_cate


@pytest.mark.parametrize("learner", ["S", "T", "X", "R"])
def test_meta_learners_return_cross_fitted_cate_predictions(learner):
    data, true_cate = heterogeneous_data()
    model = RandomForestRegressor(
        n_estimators=60, min_samples_leaf=8, random_state=9, n_jobs=1
    )
    result = estimate_effect(
        data,
        treatment="w",
        outcome="y",
        covariates=["x1", "x2"],
        learner_type=learner,
        n_splits=3,
        random_state=9,
        outcome_model=model,
        effect_model=model,
    )

    predictions = np.array(
        [row["cate_prediction"] for row in result["individual_effect_predictions"]]
    )
    assert len(predictions) == len(data)
    assert np.isfinite(predictions).all()
    assert result["effect_estimate"] == pytest.approx(predictions.mean())
    assert np.corrcoef(predictions, true_cate)[0, 1] > 0.55
    assert "not identified realized ITEs" in result["diagnostics"]["warnings"][0]
    assert result["confidence_interval"] == [None, None]


def test_meta_learner_rejects_nonbinary_treatment_and_missing_covariates():
    data, _ = heterogeneous_data(n=100)
    data["w"] = np.arange(len(data)) % 3
    with pytest.raises(ValueError, match="binary treatment"):
        estimate_effect(data, "w", "y", ["x1"], learner_type="T")
    with pytest.raises(ValueError, match="at least one"):
        estimate_effect(data.assign(w=np.arange(len(data)) % 2), "w", "y", [], learner_type="T")


@pytest.mark.parametrize(
    ("estimator_class", "mapping_name"),
    [(SLearner, "s_learner"), (TLearner, "t_learner"), (XLearner, "x_learner"), (RLearner, "r_learner")],
)
def test_meta_learners_are_registered(estimator_class, mapping_name):
    assert mapping_name in METHOD_MAPPING
    assert estimator_class.name in Estimators().names()


def test_substantive_assumptions_are_advisory():
    data, _ = heterogeneous_data(n=100)
    variables = Variables(
        treatment_variable="w", outcome_variable="y", confounders=["x1", "x2"]
    )
    validation = TLearner().validate_assumptions(data, variables)
    assert validation["assumptions_valid"] is True
    assert "argued/advisory" in validation["warnings"][0]
