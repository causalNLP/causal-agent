import numpy as np
import pandas as pd
import pytest

from cais.methods import METHOD_MAPPING
from cais.methods.survival_meta_learners import (
    estimate_survival_effect,
    route_survival_base_learner,
)


def survival_data(n=240, seed=17):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    w = rng.binomial(1, 1 / (1 + np.exp(-0.4 * x[:, 0])))
    hazard = np.exp(-1.0 - 0.25 * x[:, 0] - 0.2 * w)
    event_time = rng.exponential(1 / hazard)
    censor_time = rng.exponential(4.0, size=n)
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    return pd.DataFrame({"w": w, "duration": duration, "event": event,
                         "x1": x[:, 0], "x2": x[:, 1]})


@pytest.mark.parametrize("learner", ["S", "T", "X"])
def test_survival_meta_learners_return_rmst_cate(learner):
    df = survival_data()
    result = estimate_survival_effect(
        df, "w", "duration", "event", ["x1", "x2"],
        learner_type=learner, base_learner="ipcw_rf", horizon=2.0,
        n_splits=3, random_state=3,
    )
    assert np.isfinite(result["effect_estimate"])
    assert len(result["individual_effect_predictions"]) == len(df)
    assert result["diagnostics"]["horizon"] == 2.0
    assert "not realized ITEs" in result["diagnostics"]["warnings"][0]
    report = result["diagnostics"]["assumption_report"]
    assert "treatment_binary" in report["testable_diagnostics"]
    assert "Conditional exchangeability" in report["advisory_assessments"]
    assert report["advisory_only"] is True


def test_survival_routes_and_auto_base_learner():
    assert {"s_learner_rmst", "t_learner_rmst", "x_learner_rmst"}.issubset(METHOD_MAPPING)
    assert route_survival_base_learner(survival_data()) == "ipcw_rf"


def test_survival_rejects_nonbinary_event_and_missing_roles():
    df = survival_data()
    df["event"] = 2
    with pytest.raises(ValueError, match="event indicator"):
        estimate_survival_effect(df, "w", "duration", "event", ["x1"], horizon=2.0)
    with pytest.raises(ValueError, match="missing"):
        estimate_survival_effect(df, "w", "missing_duration", "event", ["x1"], horizon=2.0)
