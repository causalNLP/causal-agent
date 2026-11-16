import pytest

from cais.components.dataset_cleaner import (
    _build_typed_data_contract,
    _validate_generated_code,
    _validate_spec,
)
from cais.methods.utils import preprocess_data
import pandas as pd


PROFILE = {
    "columns": [
        {"name": "a", "dtype": "int64", "n_unique": 2, "missing_pct": 0.0},
        {"name": "y", "dtype": "float64", "n_unique": 4, "missing_pct": 0.1},
        {"name": "x", "dtype": "float64", "n_unique": 5, "missing_pct": 0.0},
        {"name": "unit", "dtype": "object", "n_unique": 2, "missing_pct": 0.0},
    ],
    "missing_by_col": {"a": {"count": 0, "pct": 0.0}, "y": {"count": 1, "pct": 0.1}},
}


def test_typed_contract_records_roles_types_and_missingness():
    contract = _build_typed_data_contract(
        PROFILE,
        {"treatment_variable": "a", "outcome_variable": "y", "group_variable": "unit"},
        {"method": "OLS", "column_ops": [], "assumption_prechecks": []},
    )
    assert contract["variables"]["treatment"] == ["a"]
    assert contract["variables"]["unit"] == ["unit"]
    assert contract["types"]["y"]["type"]["dtype"] == "float64"
    assert contract["missingness"]["y"]["count"] == 1
    assert contract["provenance"]["generated_spec"] is True


def test_preprocessing_rejects_implicit_imputation_and_design_mutation():
    variables = {"treatment_variable": "a", "outcome_variable": "y", "group_variable": "unit"}
    with pytest.raises(ValueError, match="Imputation"):
        _validate_spec({"column_ops": [{"op": "fillna", "cols": ["x"]}]}, PROFILE, variables)
    with pytest.raises(ValueError, match="Design columns"):
        _validate_spec({"column_ops": [{"op": "log1p", "cols": ["unit"]}]}, PROFILE, variables)


def test_generated_code_allowlist_is_fail_closed():
    _validate_generated_code("import pandas as pd\nclean_df = pd.DataFrame()")
    with pytest.raises(ValueError, match="disallowed module"):
        _validate_generated_code("import subprocess\nclean_df = None")
    with pytest.raises(ValueError, match="forbidden call"):
        _validate_generated_code("clean_df = open('x')")


def test_generic_preprocessing_does_not_impute_causal_roles():
    df = pd.DataFrame({"a": [0, 1, None], "y": [1.0, None, 3.0], "x": [2.0, 4.0, 6.0]})
    cleaned, _, _, _, mapping = preprocess_data(df, "a", "y", ["x"], verbose=False)
    assert len(cleaned) == 1
    assert cleaned["a"].tolist() == [0.0]
    assert mapping["_missingness"]["imputation"] == "none"
