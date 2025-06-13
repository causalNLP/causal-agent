import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from auto_causal.methods.linear_regression.llm_assist import (
    suggest_lr_covariates,
    interpret_lr_results
)

# Patch target for the helper function where it's used
LLM_ASSIST_MODULE = "auto_causal.methods.linear_regression.llm_assist"

@pytest.fixture
def mock_llm():
    """Fixture for a basic mock LLM object."""
    return MagicMock()

@pytest.fixture
def mock_ols_results():
    """Creates a mock statsmodels OLS results object with necessary attributes."""
    results = MagicMock()
    treatment_var = 'treatment'
    results.params = pd.Series({'const': 1.0, treatment_var: 2.5, 'cov1': 0.5})
    results.pvalues = pd.Series({'const': 0.5, treatment_var: 0.01, 'cov1': 0.1})
    # Mock conf_int() to return a DataFrame-like object accessible by .loc
    conf_int_df = pd.DataFrame([[2.0, 3.0]], index=[treatment_var], columns=[0, 1])
    results.conf_int.return_value = conf_int_df
    results.rsquared = 0.75
    results.rsquared_adj = 0.70
    return results

@pytest.fixture
def mock_diagnostics_success():
     """Creates a mock diagnostics dictionary for successful checks."""
     return {
        "status": "Success",
        "details": {
            'residuals_normality_jb_p_value': 0.6,
            'homoscedasticity_bp_lm_p_value': 0.5,
            'homoscedasticity_status': "Homoscedastic",
            'residuals_normality_status': "Normal"
        }
     }

def test_suggest_lr_covariates_placeholder(mock_llm):
    """Test the placeholder covariate suggestion function."""
    df_cols = ['a', 'b', 't', 'y']
    treatment = 't'
    outcome = 'y'
    query = "What is the effect of t on y?"
    
    # Test without LLM
    suggested_no_llm = suggest_lr_covariates(df_cols, treatment, outcome, query, llm=None)
    assert suggested_no_llm == []
    
    # Test with LLM (should still return empty list for placeholder)
    suggested_with_llm = suggest_lr_covariates(df_cols, treatment, outcome, query, llm=mock_llm)
    assert suggested_with_llm == []
    # Ensure mock LLM wasn't actually called by the placeholder
    mock_llm.assert_not_called()

@patch(f"{LLM_ASSIST_MODULE}.call_llm_with_json_output")
def test_interpret_lr_results_implementation(mock_call_json, mock_llm, mock_ols_results, mock_diagnostics_success):
    """Test the implemented results interpretation function."""
    treatment_var = 'treatment'
    mock_interpretation_text = "The treatment had a positive and significant effect."
    mock_call_json.return_value = {"interpretation": mock_interpretation_text}
    
    # --- Test with LLM --- 
    interp_with_llm = interpret_lr_results(
        mock_ols_results, 
        mock_diagnostics_success, 
        treatment_var,
        llm=mock_llm
    )
    
    assert interp_with_llm == mock_interpretation_text
    mock_call_json.assert_called_once() 
    # Basic check on the prompt structure passed to the helper
    call_args, call_kwargs = mock_call_json.call_args
    prompt = call_args[1] # Second argument is the prompt string
    assert "Linear Regression (OLS) results" in prompt
    assert "Model Results Summary:" in prompt
    assert "Model Diagnostics Summary:" in prompt
    assert treatment_var in prompt
    assert "Treatment Effect Estimate': '2.500" in prompt # Check formatting
    assert "Homoscedasticity Status': 'Homoscedastic" in prompt # Check diagnostics inclusion
    assert "Return ONLY a valid JSON" in prompt

    # --- Test LLM Call Failure --- 
    mock_call_json.reset_mock()
    mock_call_json.return_value = None # Simulate LLM helper failure
    interp_fail = interpret_lr_results(mock_ols_results, mock_diagnostics_success, treatment_var, llm=mock_llm)
    assert "LLM interpretation not available" in interp_fail
    mock_call_json.assert_called_once() # Ensure it was still called
    
    # --- Test without LLM --- 
    mock_call_json.reset_mock()
    interp_no_llm = interpret_lr_results(mock_ols_results, mock_diagnostics_success, treatment_var, llm=None)
    assert isinstance(interp_no_llm, str)
    assert "LLM interpretation not available" in interp_no_llm
    mock_call_json.assert_not_called() # Ensure helper wasn't called

# Test edge case where treatment var isn't in results (though estimator should prevent this)
def test_interpret_lr_results_treatment_not_found(mock_llm):
    """Test interpretation when treatment var is unexpectedly missing from results."""
    mock_res = MagicMock()
    mock_res.params = pd.Series({'const': 1.0})
    mock_res.pvalues = pd.Series({'const': 0.5})
    mock_res.rsquared = 0.1
    mock_res.rsquared_adj = 0.05
    # Mock conf_int to avoid error even if treatment isn't there
    mock_res.conf_int.return_value = pd.DataFrame([[0.0, 2.0]], index=['const'], columns=[0, 1])
    
    mock_diag = {"status": "Success", "details": {}} 
    
    # With LLM (should default gracefully)
    interp = interpret_lr_results(mock_res, mock_diag, "missing_treatment", llm=mock_llm)
    assert "LLM interpretation not available" in interp # Should hit default as LLM call won't work well
    
    # Without LLM
    interp_no_llm = interpret_lr_results(mock_res, mock_diag, "missing_treatment", llm=None)
    assert "LLM interpretation not available" in interp_no_llm

