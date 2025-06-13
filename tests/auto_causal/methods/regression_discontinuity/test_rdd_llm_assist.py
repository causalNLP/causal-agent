import pytest
from unittest.mock import MagicMock, patch
from auto_causal.methods.regression_discontinuity.llm_assist import (
    suggest_rdd_parameters,
    interpret_rdd_results
)

# Patch target for the helper function where it's used
LLM_ASSIST_MODULE = "auto_causal.methods.regression_discontinuity.llm_assist"

@pytest.fixture
def mock_llm():
    """Fixture for a basic mock LLM object."""
    return MagicMock()

@pytest.fixture
def mock_rdd_results():
    """Creates a mock RDD results dictionary."""
    return {
        'effect_estimate': 10.5,
        'p_value': 0.01,
        'confidence_interval': [8.0, 13.0],
        'standard_error': 1.25,
        'method_used': 'DoWhy RDD' # Or Fallback RDD
    }

@pytest.fixture
def mock_rdd_diagnostics_success():
     """Creates a mock RDD diagnostics dictionary for successful checks."""
     return {
        "status": "Success (Partial Implementation)",
        "details": {
            'covariate_balance': {
                'cov1': {'p_value': 0.6, 'balanced': 'Yes'},
                'cov2': {'p_value': 0.02, 'balanced': 'No (p <= 0.05)'}
            },
            'continuity_density_test': 'Not Implemented',
            'visual_inspection': 'Recommended'
        }
     }

def test_suggest_rdd_parameters_placeholder(mock_llm):
    """Test the placeholder RDD parameter suggestion function."""
    df_cols = ['score', 'age', 'outcome']
    query = "Effect of passing score (50) on outcome?"
    
    # Test without LLM
    suggested_no_llm = suggest_rdd_parameters(df_cols, query, llm=None)
    assert suggested_no_llm == {}
    
    # Test with LLM (should still return empty dict for placeholder)
    suggested_with_llm = suggest_rdd_parameters(df_cols, query, llm=mock_llm)
    assert suggested_with_llm == {}
    mock_llm.assert_not_called()

@patch(f"{LLM_ASSIST_MODULE}.call_llm_with_json_output")
def test_interpret_rdd_results_implementation(mock_call_json, mock_llm, mock_rdd_results, mock_rdd_diagnostics_success):
    """Test the implemented RDD results interpretation function."""
    mock_interpretation_text = "RDD shows a significant positive effect at the cutoff."
    mock_call_json.return_value = {"interpretation": mock_interpretation_text}
    
    # --- Test with LLM --- 
    interp_with_llm = interpret_rdd_results(
        mock_rdd_results, 
        mock_rdd_diagnostics_success, 
        llm=mock_llm
    )
    
    assert interp_with_llm == mock_interpretation_text
    mock_call_json.assert_called_once() 
    # Basic check on the prompt structure passed to the helper
    call_args, call_kwargs = mock_call_json.call_args
    prompt = call_args[1] # Second argument is the prompt string
    assert "Regression Discontinuity Design (RDD) results" in prompt
    assert "Estimation Results Summary:" in prompt
    assert "Diagnostics Summary:" in prompt
    assert "Effect Estimate': '10.500" in prompt # Check formatting
    assert "Number of Unbalanced Covariates (p<=0.05)': 1" in prompt # Check diagnostics summary
    assert "visual inspection of the running variable vs outcome is recommended" in prompt
    assert "Return ONLY a valid JSON" in prompt

    # --- Test LLM Call Failure --- 
    mock_call_json.reset_mock()
    mock_call_json.return_value = None # Simulate LLM helper failure
    interp_fail = interpret_rdd_results(mock_rdd_results, mock_rdd_diagnostics_success, llm=mock_llm)
    assert "LLM interpretation not available for RDD" in interp_fail
    mock_call_json.assert_called_once() # Ensure it was still called
    
    # --- Test without LLM --- 
    mock_call_json.reset_mock()
    interp_no_llm = interpret_rdd_results(mock_rdd_results, mock_rdd_diagnostics_success, llm=None)
    assert isinstance(interp_no_llm, str)
    assert "LLM interpretation not available for RDD" in interp_no_llm
    mock_call_json.assert_not_called() # Ensure helper wasn't called

# Test interpretation with failed diagnostics
def test_interpret_rdd_results_failed_diagnostics(mock_llm):
    """Test interpretation when diagnostics failed."""
    mock_res = {'effect_estimate': 5.0, 'p_value': 0.04}
    mock_diag = {"status": "Failed", "error": "Something broke"}
    
    # Patch the call to LLM helper for this specific test case
    with patch(f"{LLM_ASSIST_MODULE}.call_llm_with_json_output") as mock_call_json_fail:
        mock_call_json_fail.return_value = {"interpretation": "Interpreted despite failed diagnostics"}
        
        interp = interpret_rdd_results(mock_res, mock_diag, llm=mock_llm)
        
        assert interp == "Interpreted despite failed diagnostics"
        mock_call_json_fail.assert_called_once()
        call_args, call_kwargs = mock_call_json_fail.call_args
        prompt = call_args[1]
        assert "Diagnostics Summary:" in prompt
        assert "Status': 'Failed" in prompt # Check failed status is in prompt
        assert "Error': 'Something broke" in prompt

