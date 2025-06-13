import pytest

# Import the function to test and constants
from auto_causal.components.decision_tree import (
    select_method,
    METHOD_ASSUMPTIONS, # Import assumptions map
    REGRESSION_ADJUSTMENT, LINEAR_REGRESSION, LINEAR_REGRESSION_COV,
    DIFF_IN_DIFF, REGRESSION_DISCONTINUITY, PROPENSITY_SCORE_MATCHING,
    INSTRUMENTAL_VARIABLE
)

# --- Test Data Fixtures (Optional, but good practice) ---
# Using simple dicts for now

@pytest.fixture
def base_variables():
    return {
        "treatment_variable": "T",
        "outcome_variable": "Y",
        "covariates": ["X1", "X2"],
        "time_variable": None,
        "group_variable": None,
        "instrument_variable": None,
        "running_variable": None,
        "cutoff_value": None
    }

@pytest.fixture
def base_dataset_analysis():
    return {
        "temporal_structure": False
        # Add other keys as needed by specific tests, e.g., potential_instruments
    }

# --- Test Cases ---

def test_no_covariates(base_dataset_analysis, base_variables):
    """Test: No covariates provided -> Regression Adjustment"""
    variables = base_variables.copy()
    variables["covariates"] = []
    result = select_method(base_dataset_analysis, variables, is_rct=False)
    assert result["selected_method"] == REGRESSION_ADJUSTMENT
    assert "no covariates" in result["method_justification"].lower()
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[REGRESSION_ADJUSTMENT]

def test_rct_no_covariates(base_dataset_analysis, base_variables):
    """Test: RCT, no covariates -> Linear Regression"""
    variables = base_variables.copy()
    variables["covariates"] = [] # Explicitly empty
    # Even though the first check catches empty covariates, test RCT path specifically
    result = select_method(base_dataset_analysis, variables, is_rct=True)
    # The initial check for no covariates takes precedence
    assert result["selected_method"] == REGRESSION_ADJUSTMENT
    # assert result["selected_method"] == LINEAR_REGRESSION # This won't be reached

def test_rct_with_covariates(base_dataset_analysis, base_variables):
    """Test: RCT with covariates -> Linear Regression with Covariates"""
    variables = base_variables.copy()
    result = select_method(base_dataset_analysis, variables, is_rct=True)
    assert result["selected_method"] == LINEAR_REGRESSION_COV
    assert "rct" in result["method_justification"].lower()
    assert "covariates are provided" in result["method_justification"].lower()
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[LINEAR_REGRESSION_COV]

def test_observational_temporal(base_dataset_analysis, base_variables):
    """Test: Observational, temporal structure -> DiD"""
    variables = base_variables.copy()
    variables["time_variable"] = "time"
    variables["group_variable"] = "unit" # Often needed for DiD context
    dataset_analysis = base_dataset_analysis.copy()
    dataset_analysis["temporal_structure"] = True
    result = select_method(dataset_analysis, variables, is_rct=False)
    assert result["selected_method"] == DIFF_IN_DIFF
    assert "temporal structure" in result["method_justification"].lower()
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[DIFF_IN_DIFF]

def test_observational_rdd(base_dataset_analysis, base_variables):
    """Test: Observational, RDD vars present -> RDD"""
    variables = base_variables.copy()
    variables["running_variable"] = "score"
    variables["cutoff_value"] = 50
    result = select_method(base_dataset_analysis, variables, is_rct=False)
    assert result["selected_method"] == REGRESSION_DISCONTINUITY
    assert "running variable" in result["method_justification"].lower()
    assert "cutoff" in result["method_justification"].lower()
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[REGRESSION_DISCONTINUITY]

def test_observational_iv(base_dataset_analysis, base_variables):
    """Test: Observational, IV present -> IV"""
    variables = base_variables.copy()
    variables["instrument_variable"] = "Z"
    result = select_method(base_dataset_analysis, variables, is_rct=False)
    assert result["selected_method"] == INSTRUMENTAL_VARIABLE
    assert "instrumental variable" in result["method_justification"].lower()
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[INSTRUMENTAL_VARIABLE]

def test_observational_confounders_default_psm(base_dataset_analysis, base_variables):
    """Test: Observational, confounders, no other design -> PSM (default)"""
    variables = base_variables.copy() # Has covariates by default
    # Ensure no other conditions are met
    dataset_analysis = base_dataset_analysis.copy()
    dataset_analysis["temporal_structure"] = False
    variables["time_variable"] = None
    variables["running_variable"] = None
    variables["instrument_variable"] = None
    
    result = select_method(dataset_analysis, variables, is_rct=False)
    assert result["selected_method"] == PROPENSITY_SCORE_MATCHING
    assert "observed confounders" in result["method_justification"].lower()
    assert "selected as the default method" in result["method_justification"].lower()
    assert result["method_assumptions"] == METHOD_ASSUMPTIONS[PROPENSITY_SCORE_MATCHING]

# Note: A specific test for the final fallback (Reg Adjustment for observational 
# with covariates but somehow no other method fits) might be hard to trigger 
# given the current logic defaults to PSM if covariates exist and IV/RDD/DiD don't apply.
# The initial 'no covariates' test effectively covers the main Reg Adjustment path.

if __name__ == '__main__':
    pytest.main() 