import pytest
from unittest.mock import patch, MagicMock

# Import functions to test
from auto_causal.methods.instrumental_variable.llm_assist import (
    identify_instrument_variable,
    validate_instrument_assumptions_qualitative,
    interpret_iv_results
)
# Assume BaseChatModel is importable for type hinting if needed elsewhere,
# but for mocking we don't strictly need it here.
# from langchain.chat_models.base import BaseChatModel

# Assume shared helpers are in this location
# LLM_HELPERS_PATH = "causalscientist.auto_causal.utils.llm_helpers"
# Correct patch target is where the function is *used*
LLM_ASSIST_PATH = "auto_causal.methods.instrumental_variable.llm_assist"

@pytest.fixture
def mock_llm():
    """Fixture to create a mock LLM object."""
    return MagicMock() # Basic mock, can be configured in tests

# --- Tests for identify_instrument_variable --- #

@patch(f"{LLM_ASSIST_PATH}.call_llm_with_json_output")
def test_identify_instrument_variable_success(mock_call_json, mock_llm):
    """Test successful identification of instruments."""
    mock_call_json.return_value = {"potential_instruments": ["Z1", "Z2"]}
    df_cols = ["Y", "T", "Z1", "Z2", "W"]
    query = "What is the effect of T on Y using Z1 and Z2 as instruments?"

    result = identify_instrument_variable(df_cols, query, llm=mock_llm)

    assert result == ["Z1", "Z2"]
    mock_call_json.assert_called_once()
    # TODO: Optionally add assertion on the prompt passed to mock_call_json

@patch(f"{LLM_ASSIST_PATH}.call_llm_with_json_output")
def test_identify_instrument_variable_llm_fail(mock_call_json, mock_llm):
    """Test when LLM call fails or returns bad format."""
    mock_call_json.return_value = None # Simulate failure
    df_cols = ["Y", "T", "Z1", "Z2", "W"]
    query = "What is the effect of T on Y using Z1 and Z2 as instruments?"

    result = identify_instrument_variable(df_cols, query, llm=mock_llm)

    assert result == [] # Expect empty list on failure
    mock_call_json.assert_called_once()

@patch(f"{LLM_ASSIST_PATH}.call_llm_with_json_output")
def test_identify_instrument_variable_no_llm(mock_call_json):
    """Test behavior when no LLM is provided."""
    df_cols = ["Y", "T", "Z1", "Z2", "W"]
    query = "What is the effect of T on Y using Z1 and Z2 as instruments?"

    result = identify_instrument_variable(df_cols, query, llm=None)

    assert result == []
    mock_call_json.assert_not_called() # LLM helper should not be called

# --- Tests for validate_instrument_assumptions_qualitative --- #

@patch(f"{LLM_ASSIST_PATH}.call_llm_with_json_output")
def test_validate_assumptions_success(mock_call_json, mock_llm):
    """Test successful qualitative validation."""
    mock_response = {"exclusion_assessment": "Plausible", "exogeneity_assessment": "Likely holds"}
    mock_call_json.return_value = mock_response

    result = validate_instrument_assumptions_qualitative(
        treatment='T', outcome='Y', instrument=['Z1'], covariates=['W'], query="Test query", llm=mock_llm
    )

    assert result == mock_response
    mock_call_json.assert_called_once()

@patch(f"{LLM_ASSIST_PATH}.call_llm_with_json_output")
def test_validate_assumptions_llm_fail(mock_call_json, mock_llm):
    """Test qualitative validation when LLM fails."""
    mock_call_json.return_value = None # Simulate failure

    result = validate_instrument_assumptions_qualitative(
        treatment='T', outcome='Y', instrument=['Z1'], covariates=['W'], query="Test query", llm=mock_llm
    )

    assert result == {"exclusion_assessment": "LLM Check Failed", "exogeneity_assessment": "LLM Check Failed"}
    mock_call_json.assert_called_once()

@patch(f"{LLM_ASSIST_PATH}.call_llm_with_json_output")
def test_validate_assumptions_no_llm(mock_call_json):
    """Test qualitative validation when no LLM is provided."""
    result = validate_instrument_assumptions_qualitative(
        treatment='T', outcome='Y', instrument=['Z1'], covariates=['W'], query="Test query", llm=None
    )

    assert result == {"exclusion_assessment": "LLM Not Provided", "exogeneity_assessment": "LLM Not Provided"}
    mock_call_json.assert_not_called()

# --- Tests for interpret_iv_results --- #

@patch(f"{LLM_ASSIST_PATH}.call_llm_with_json_output")
def test_interpret_results_success(mock_call_json, mock_llm):
    """Test successful interpretation generation."""
    mock_interpretation_text = "The effect is positive and significant."
    # Simulate the JSON helper returning a dict containing the text
    mock_call_json.return_value = {"interpretation": mock_interpretation_text}
    sample_results = {'effect_estimate': 2.5, 'p_value': 0.01, 'confidence_interval': [1.0, 4.0], 'treatment_variable': 'T', 'outcome_variable': 'Y', 'method_used': 'dowhy'}
    sample_diagnostics = {'first_stage_f_statistic': 50.0, 'weak_instrument_test_status': 'Strong', 'overid_test_applicable': False}

    result = interpret_iv_results(sample_results, sample_diagnostics, llm=mock_llm)

    assert result == mock_interpretation_text
    mock_call_json.assert_called_once()
    # TODO: Optionally add assertion on the prompt passed to mock_call_json

@patch(f"{LLM_ASSIST_PATH}.call_llm_with_json_output")
def test_interpret_results_llm_fail(mock_call_json, mock_llm):
    """Test interpretation generation when LLM fails."""
    mock_call_json.return_value = None # Simulate failure
    sample_results = {'effect_estimate': 2.5}
    sample_diagnostics = {}

    result = interpret_iv_results(sample_results, sample_diagnostics, llm=mock_llm)

    assert "LLM interpretation could not be generated" in result
    mock_call_json.assert_called_once()

@patch(f"{LLM_ASSIST_PATH}.call_llm_with_json_output")
def test_interpret_results_no_llm(mock_call_json):
    """Test interpretation generation when no LLM is provided."""
    sample_results = {'effect_estimate': 2.5}
    sample_diagnostics = {}

    result = interpret_iv_results(sample_results, sample_diagnostics, llm=None)

    assert "LLM was not available" in result
    mock_call_json.assert_not_called() 