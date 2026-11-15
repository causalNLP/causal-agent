import numpy as np
import pandas as pd
import pytest

from cais.methods.doubly_robust.estimator import estimate_aipw, estimate_dml


def _data(seed=4, n=600):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-0.5 * x))
    a = rng.binomial(1, p)
    y = 2.0 * a + 0.7 * x + rng.normal(size=n)
    return pd.DataFrame({"a": a, "y": y, "x": x})


def test_aipw_ate_is_close_to_known_effect():
    result = estimate_aipw(_data(), "a", "y", ["x"], n_splits=3, random_state=1)
    assert result["method_used"].startswith("cross-fitted AIPW")
    assert abs(result["effect_estimate"] - 2.0) < 0.35
    assert result["standard_error"] >= 0


def test_aipw_supports_target_specific_contracts():
    data = _data()
    att = estimate_aipw(data, "a", "y", ["x"], target="ATT", n_splits=3)
    atc = estimate_aipw(data, "a", "y", ["x"], target="ATC", n_splits=3)
    assert att["diagnostics"]["estimand"] == "ATT"
    assert atc["diagnostics"]["estimand"] == "ATC"


def test_dml_rejects_wrong_estimand_instead_of_mislabelling():
    with pytest.raises(ValueError, match="ATE"):
        estimate_dml(_data(), "a", "y", ["x"], target="ATT")
