import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from causalscientist.auto_causal.methods.backdoor_adjustment.llm_assist import (
    identify_backdoor_set,
    interpret_backdoor_results
)

# Patch target for the helper function where it's used
LLM_ASSIST_MODULE = "causalscientist.auto_causal.methods.backdoor_adjustment.llm_assist"

@pytest.fixture
def mock_llm():
    """Fixture for a basic mock LLM object."""
    return MagicMock()

@pytest.fixture
def mock_ols_results():
    """Creates a mock statsmodels OLS results object."""
    results = MagicMock()
    treatment_var = 'treatment'
    covs = ['confounder1', 'confounder2']
    results.params = pd.Series({'const': 10.0, treatment_var: 3.1, **{c: i*0.5 for i, c in enumerate(covs)}})
    results.pvalues = pd.Series({'const': 0.1, treatment_var: 0.02, **{c: 0.1 for c in covs}})
    conf_int_df = pd.DataFrame([[2.1, 4.1]], index=[treatment_var], columns=[0, 1])
    results.conf_int.return_value = conf_int_df
    results.rsquared = 0.6
    results.rsquared_adj = 0.55
    return results

@pytest.fixture
def mock_backdoor_diagnostics():
     """Creates a mock diagnostics dictionary."""
     return {
        "status": "Success",
        "details": {
            'r_squared': 0.6, 
            'residuals_normality_status': "Non-Normal",
            'homoscedasticity_status': "Homoscedastic",
            'multicollinearity_status': "Low"
        }
     }

# --- Tests for identify_backdoor_set --- 

@patch(f"{LLM_ASSIST_MODULE}.call_llm_with_json_output")
def test_identify_backdoor_set_llm_success(mock_call_json, mock_llm):
    """Test successful identification of backdoor set via LLM."""
    mock_call_json.return_value = {"suggested_backdoor_set": ["W1", "W2"]}
    df_cols = ['Y', 'T', 'W1', 'W2', 'X']
    treatment = 'T'
    outcome = 'Y'
    query = "Effect of T on Y?"
    
    result = identify_backdoor_set(df_cols, treatment, outcome, query=query, llm=mock_llm)
    
    assert result == ["W1", "W2"]
    mock_call_json.assert_called_once()
    # Check prompt content
    call_args, _ = mock_call_json.call_args
    prompt = call_args[1]
    assert "'T' on 'Y'" in prompt
    assert "Available variables" in prompt
    assert "'W1', 'W2', 'X'" in prompt # Check potential confounders list
    assert "Return ONLY a valid JSON" in prompt
    assert "suggested_backdoor_set" in prompt

@patch(f"{LLM_ASSIST_MODULE}.call_llm_with_json_output")
def test_identify_backdoor_set_llm_combines_existing(mock_call_json, mock_llm):
    """Test that LLM suggestions are combined with existing covariates."""
    mock_call_json.return_value = {"suggested_backdoor_set": ["W2", "W3"]}
    df_cols = ['Y', 'T', 'W1', 'W2', 'W3']
    existing = ["W1", "W2"] # User provided W1, W2
    
    result = identify_backdoor_set(df_cols, 'T', 'Y', existing_covariates=existing, llm=mock_llm)
    
    # Order should be existing + suggested, with duplicates removed
    assert result == ["W1", "W2", "W3"]
    mock_call_json.assert_called_once()

def test_identify_backdoor_set_no_llm():
    """Test behavior when no LLM is provided."""
    df_cols = ['Y', 'T', 'W1']
    existing = ["W1"]
    result_with = identify_backdoor_set(df_cols, 'T', 'Y', existing_covariates=existing, llm=None)
    assert result_with == ["W1"] # Returns only existing
    
    result_without = identify_backdoor_set(df_cols, 'T', 'Y', llm=None)
    assert result_without == [] # Returns empty list

@patch(f"{LLM_ASSIST_MODULE}.call_llm_with_json_output")
def test_identify_backdoor_set_llm_fail(mock_call_json, mock_llm):
    """Test behavior when LLM call fails or returns bad format."""
    mock_call_json.return_value = None
    df_cols = ['Y', 'T', 'W1']
    existing = ["W1"]
    result = identify_backdoor_set(df_cols, 'T', 'Y', existing_covariates=existing, llm=mock_llm)
    assert result == ["W1"] # Should return only existing on failure
    mock_call_json.assert_called_once()

# --- Tests for interpret_backdoor_results --- 

@patch(f"{LLM_ASSIST_MODULE}.call_llm_with_json_output")
def test_interpret_backdoor_results_implementation(mock_call_json, mock_llm, mock_ols_results, mock_backdoor_diagnostics):
    """Test the implemented backdoor results interpretation function."""
    treatment_var = 'treatment'
    covariates = ['confounder1', 'confounder2']
    mock_interpretation_text = "After adjusting, treatment has a significant positive effect, assuming confounders are controlled."
    mock_call_json.return_value = {"interpretation": mock_interpretation_text}
    
    # --- Test with LLM --- 
    interp_with_llm = interpret_backdoor_results(
        mock_ols_results, 
        mock_backdoor_diagnostics, 
        treatment_var,
        covariates,
        llm=mock_llm
    )
    
    assert interp_with_llm == mock_interpretation_text
    mock_call_json.assert_called_once() 
    # Basic check on the prompt structure passed to the helper
    call_args, _ = mock_call_json.call_args
    prompt = call_args[1]
    assert "Backdoor Adjustment (Regression) results" in prompt
    assert "Results Summary:" in prompt
    assert "Diagnostics Summary" in prompt
    assert "Treatment Effect Estimate': '3.100" in prompt
    assert "Adjustment Set (Covariates Used)': ['confounder1', 'confounder2']" in prompt
    assert "relies heavily on the assumption" in prompt # Check assumption emphasis
    assert "confounder1', 'confounder2" in prompt
    assert "Residuals Normality Status': 'Non-Normal'" in prompt
    assert "Return ONLY a valid JSON" in prompt

    # --- Test LLM Call Failure --- 
    mock_call_json.reset_mock()
    mock_call_json.return_value = None
    interp_fail = interpret_backdoor_results(mock_ols_results, mock_backdoor_diagnostics, treatment_var, covariates, llm=mock_llm)
    assert "LLM interpretation not available for Backdoor Adjustment" in interp_fail
    mock_call_json.assert_called_once()
    
    # --- Test without LLM --- 
    mock_call_json.reset_mock()
    interp_no_llm = interpret_backdoor_results(mock_ols_results, mock_backdoor_diagnostics, treatment_var, covariates, llm=None)
    assert "LLM interpretation not available for Backdoor Adjustment" in interp_no_llm
    mock_call_json.assert_not_called()
