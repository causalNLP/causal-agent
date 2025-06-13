import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from auto_causal.methods.diff_in_means.llm_assist import interpret_dim_results

# Patch target for the helper function where it's used
LLM_ASSIST_MODULE = "auto_causal.methods.diff_in_means.llm_assist"

@pytest.fixture
def mock_llm():
    """Fixture for a basic mock LLM object."""
    return MagicMock()

@pytest.fixture
def mock_dim_ols_results():
    """Creates a mock statsmodels OLS results object for DiM."""
    results = MagicMock()
    treatment_var = 'treatment'
    results.params = pd.Series({'const': 20.0, treatment_var: 5.1})
    results.pvalues = pd.Series({'const': 0.001, treatment_var: 0.03})
    # Mock conf_int() to return a DataFrame-like object accessible by .loc
    conf_int_df = pd.DataFrame([[0.5, 9.7]], index=[treatment_var], columns=[0, 1])
    results.conf_int.return_value = conf_int_df
    return results

@pytest.fixture
def mock_dim_diagnostics():
     """Creates a mock diagnostics dictionary for DiM."""
     return {
        "status": "Success",
        "details": {
            'control_group_stats': {'mean': 20.1, 'std': 2.0, 'count': 75},
            'treated_group_stats': {'mean': 25.2, 'std': 2.2, 'count': 75},
            'variance_homogeneity_status': "Likely Similar"
        }
     }

@patch(f"{LLM_ASSIST_MODULE}.call_llm_with_json_output")
def test_interpret_dim_results_implementation(mock_call_json, mock_llm, mock_dim_ols_results, mock_dim_diagnostics):
    """Test the implemented DiM results interpretation function."""
    treatment_var = 'treatment'
    mock_interpretation_text = "The treatment group had a significantly higher average outcome."
    mock_call_json.return_value = {"interpretation": mock_interpretation_text}
    
    # --- Test with LLM --- 
    interp_with_llm = interpret_dim_results(
        mock_dim_ols_results, 
        mock_dim_diagnostics, 
        treatment_var,
        llm=mock_llm
    )
    
    assert interp_with_llm == mock_interpretation_text
    mock_call_json.assert_called_once() 
    # Basic check on the prompt structure passed to the helper
    call_args, call_kwargs = mock_call_json.call_args
    prompt = call_args[1] # Second argument is the prompt string
    assert "Difference in Means results" in prompt
    assert "Results Summary:" in prompt
    assert "Effect Estimate (Difference in Means)': '5.100" in prompt # Check formatting
    assert "Control Group Mean Outcome': '20.100" in prompt
    assert "Treated Group Mean Outcome': '25.200" in prompt
    assert "Return ONLY a valid JSON" in prompt

    # --- Test LLM Call Failure --- 
    mock_call_json.reset_mock()
    mock_call_json.return_value = None # Simulate LLM helper failure
    interp_fail = interpret_dim_results(mock_dim_ols_results, mock_dim_diagnostics, treatment_var, llm=mock_llm)
    assert "LLM interpretation not available for Difference in Means" in interp_fail
    mock_call_json.assert_called_once() # Ensure it was still called
    
    # --- Test without LLM --- 
    mock_call_json.reset_mock()
    interp_no_llm = interpret_dim_results(mock_dim_ols_results, mock_dim_diagnostics, treatment_var, llm=None)
    assert isinstance(interp_no_llm, str)
    assert "LLM interpretation not available for Difference in Means" in interp_no_llm
    mock_call_json.assert_not_called() # Ensure helper wasn't called
