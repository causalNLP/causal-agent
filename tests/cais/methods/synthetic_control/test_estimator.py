import numpy as np
import pandas as pd
import pytest

from cais.estimator_lib import Estimators
from cais.methods import METHOD_MAPPING
from cais.methods.synthetic_control.estimator import SyntheticControl, estimate_effect


def make_panel(two_treated=False):
    rows = []
    donor_loadings = {"d1": (1.0, 0.2), "d2": (3.0, -0.1), "d3": (2.0, 0.4)}
    units = dict(donor_loadings)
    units["treated"] = (0.25 * 1.0 + 0.75 * 3.0, 0.25 * 0.2 + 0.75 * -0.1)
    if two_treated:
        units["treated2"] = units["treated"]
    for unit, (level, slope) in units.items():
        for year in range(8):
            treated = unit.startswith("treated") and year >= 5
            untreated_y = level + slope * year + 0.1 * year**2
            rows.append({"unit": unit, "year": year, "treated": int(treated), "y": untreated_y + (4.0 if treated else 0.0), "x": level})
    return pd.DataFrame(rows)


def test_recovers_effect_and_convex_weights():
    result = estimate_effect(
        make_panel(), treatment="treated", outcome="y", covariates=["x"],
        time_variable="year", group_variable="unit", run_placebos=False,
    )
    assert result["effect_estimate"] == pytest.approx(4.0, abs=1e-5)
    weights = result["diagnostics"]["donor_weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(weight >= 0 for weight in weights.values())
    assert result["diagnostics"]["pre_mspe"] < 1e-10


def test_aggregates_a_small_number_of_treated_units():
    result = estimate_effect(
        make_panel(two_treated=True), treatment="treated", outcome="y", covariates=[],
        time_variable="year", group_variable="unit", run_placebos=False,
    )
    assert result["effect_estimate"] == pytest.approx(4.0, abs=1e-5)
    assert len(result["diagnostics"]["treated_units"]) == 2


def test_placebo_inference_and_registration():
    panel = make_panel()
    # Avoid a degenerate exact pre-fit, for which the paper's MSPE filter can
    # legitimately exclude every placebo.
    panel.loc[(panel.unit == "treated") & (panel.year == 2), "y"] += 0.2
    result = estimate_effect(
        panel, treatment="treated", outcome="y", covariates=[],
        time_variable="year", group_variable="unit", placebo_mspe_multiplier=1e9,
    )
    assert result["p_value"] is not None
    assert result["diagnostics"]["n_placebos"] > 0
    assert METHOD_MAPPING["synthetic_control"] is estimate_effect
    assert SyntheticControl.name in Estimators().names()


def test_rejects_staggered_adoption():
    panel = make_panel(two_treated=True)
    panel.loc[(panel.unit == "treated2") & (panel.year == 5), "treated"] = 0
    with pytest.raises(ValueError, match="common intervention time"):
        estimate_effect(panel, treatment="treated", outcome="y", time_variable="year", group_variable="unit")
