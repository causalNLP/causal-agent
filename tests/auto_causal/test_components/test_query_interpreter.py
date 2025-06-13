import pytest
from unittest.mock import patch, MagicMock

from causalscientist.auto_causal.components.query_interpreter import interpret_query
from causalscientist.auto_causal.models import LLMTreatmentReferenceLevel

# Basic mock data setup
MOCK_QUERY_INFO_REF_LEVEL = {
    "query_text": "What is the effect of different fertilizers (Nitro, Phos, Control) on crop_yield, using Control as the baseline?",
    "potential_treatments": ["fertilizer_type"],
    "outcome_hints": ["crop_yield"],
    "covariates_hints": ["soil_ph", "rainfall"]
}

MOCK_DATASET_ANALYSIS_REF_LEVEL = {
    "columns": ["fertilizer_type", "crop_yield", "soil_ph", "rainfall"],
    "column_categories": {
        "fertilizer_type": "categorical_multi", # Assuming a category type for multi-level
        "crop_yield": "continuous_numeric",
        "soil_ph": "continuous_numeric",
        "rainfall": "continuous_numeric"
    },
    "potential_treatments": ["fertilizer_type"],
    "potential_outcomes": ["crop_yield"],
    "value_counts": { # Added for providing unique values to the prompt
        "fertilizer_type": {
            "values": ["Nitro", "Phos", "Control"]
        }
    },
    "columns_data_preview": { # Fallback if value_counts isn't structured as expected
        "fertilizer_type": ["Nitro", "Phos", "Control", "Nitro", "Control"]
    }
    # Add other necessary fields from DatasetAnalysis model if interpret_query uses them
}

MOCK_DATASET_DESCRIPTION_REF_LEVEL = "A dataset from an agricultural experiment."

def test_interpret_query_identifies_treatment_reference_level():
    """
    Test that interpret_query correctly identifies and returns the treatment_reference_level
    when the LLM simulation provides one.
    """
    # Mock the LLM client and its structured output
    mock_llm_instance = MagicMock()
    mock_structured_llm = MagicMock()

    # This will be the mock for the call related to TREATMENT_REFERENCE_IDENTIFICATION_PROMPT_TEMPLATE
    # The other LLM calls (for T, O, C, IV, RDD, RCT) also need to be considered
    # or made to return benign defaults for this specific test.

    # Simulate LLM responses for different calls within interpret_query
    def mock_llm_call_router(*args, **kwargs):
        # The first argument to _call_llm_for_var is the llm instance,
        # The second is the prompt string
        # The third is the Pydantic model for structured output

        # args[0] is llm, args[1] is prompt, args[2] is pydantic_model
        pydantic_model_passed = args[2]

        if pydantic_model_passed == LLMTreatmentReferenceLevel:
            return LLMTreatmentReferenceLevel(reference_level="Control", reasoning="Identified from query text.")
        # Add mocks for other LLM calls if interpret_query strictly needs them to proceed
        # For example, for identifying treatment, outcome, covariates, IV, RDD, RCT:
        elif "most likely treatment variable" in args[1]: # Simplified check for treatment prompt
            return MagicMock(variable_name="fertilizer_type")
        elif "most likely outcome variable" in args[1]: # Simplified check for outcome prompt
             return MagicMock(variable_name="crop_yield")
        elif "valid covariates" in args[1]: # Simplified check for covariates prompt
             return MagicMock(covariates=["soil_ph", "rainfall"])
        elif "Instrumental Variables" in args[1]: # Check for IV prompt
            return MagicMock(instrument_variable=None)
        elif "Regression Discontinuity Design" in args[1]: # Check for RDD prompt
            return MagicMock(running_variable=None, cutoff_value=None)
        elif "Randomized Controlled Trial" in args[1]: # Check for RCT prompt
            return MagicMock(is_rct=False, reasoning="No indication of RCT.")
        return MagicMock() # Default mock for other calls

    # Patch _call_llm_for_var which is used internally by interpret_query's helpers
    with patch('causalscientist.auto_causal.components.query_interpreter._call_llm_for_var', side_effect=mock_llm_call_router) as mock_llm_call:
        # Patch get_llm_client to return our mock_llm_instance
        # This ensures that _call_llm_for_var uses the intended LLM mock when called from within interpret_query
        with patch('causalscientist.auto_causal.components.query_interpreter.get_llm_client', return_value=mock_llm_instance) as mock_get_llm:
            
            result = interpret_query(
                query_info=MOCK_QUERY_INFO_REF_LEVEL,
                dataset_analysis=MOCK_DATASET_ANALYSIS_REF_LEVEL,
                dataset_description=MOCK_DATASET_DESCRIPTION_REF_LEVEL
            )

    assert "treatment_reference_level" in result, "treatment_reference_level should be in the result"
    assert result["treatment_reference_level"] == "Control", "Incorrect treatment_reference_level identified"
    
    # Verify that the LLM was called to get the reference level
    # This requires checking the calls made to the mock_llm_call
    found_ref_level_call = False
    for call_args in mock_llm_call.call_args_list:
        # call_args is a tuple; call_args[0] contains positional args, call_args[1] has kwargs
        # The third positional argument to _call_llm_for_var is the pydantic_model
        if len(call_args[0]) >= 3 and call_args[0][2] == LLMTreatmentReferenceLevel:
            found_ref_level_call = True
            # Optionally, check the prompt content here too if needed
            # prompt_content = call_args[0][1]
            # assert "using Control as the baseline" in prompt_content
            break
    assert found_ref_level_call, "LLM call for treatment reference level was not made."

    # Basic checks for other essential variables (assuming they are mocked simply)
    assert result["treatment_variable"] == "fertilizer_type"
    assert result["outcome_variable"] == "crop_yield"
    assert result["is_rct"] is False # Based on mock 