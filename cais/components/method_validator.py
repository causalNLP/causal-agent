"""
Method validator component for causal inference methods.

This module validates the selected causal inference method against
dataset characteristics and available variables.
"""

from typing import Dict, List, Any, Optional


def validate_method(method_info: Dict[str, Any], dataset_analysis: Dict[str, Any], 
                    variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the selected causal method against dataset characteristics.
    
    Args:
        method_info: Information about the selected method from decision_tree
        dataset_analysis: Dataset analysis results from dataset_analyzer
        variables: Identified variables from query_interpreter
        
    Returns:
        Dict with validation results:
            - valid: Boolean indicating if method is valid
            - concerns: List of concerns/issues with the selected method
            - alternative_suggestions: Alternative methods if the selected method is problematic
            - recommended_method: Updated method recommendation if issues are found
    """
    method = method_info.get("selected_method")
    assumptions = method_info.get("method_assumptions", [])
    
    # Get required variables
    treatment = variables.get("treatment_variable")
    outcome = variables.get("outcome_variable")
    covariates = variables.get("covariates", [])
    time_variable = variables.get("time_variable")
    group_variable = variables.get("group_variable")
    instrument_variable = variables.get("instrument_variable")
    running_variable = variables.get("running_variable")
    cutoff_value = variables.get("cutoff_value")
    
    # Initialize validation result
    validation_result = {
        "valid": True,
        "concerns": [],
        "alternative_suggestions": [],
        "recommended_method": method,
    }
    
    # Common validations for all methods
    if treatment is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("Treatment variable is not identified")
    
    if outcome is None:
        validation_result["valid"] = False
        validation_result["concerns"].append("Outcome variable is not identified")
    
    # Method-specific validations
    if method == "propensity_score_matching":
        validate_propensity_score_matching(validation_result, dataset_analysis, variables)
    
    elif method == "regression_adjustment":
        validate_regression_adjustment(validation_result, dataset_analysis, variables)
    
    elif method == "instrumental_variable":
        validate_instrumental_variable(validation_result, dataset_analysis, variables)
    
    elif method == "difference_in_differences":
        validate_difference_in_differences(validation_result, dataset_analysis, variables)
    
    elif method == "regression_discontinuity_design":
        validate_regression_discontinuity(validation_result, dataset_analysis, variables)
    
    elif method == "backdoor_adjustment":
        validate_backdoor_adjustment(validation_result, dataset_analysis, variables)
    
    # If there are serious concerns, recommend alternatives
    if not validation_result["valid"]:
        validation_result["recommended_method"] = recommend_alternative(
            method, validation_result["concerns"], method_info.get("alternatives", [])
        )
    
    # Make sure assumptions are listed in the validation result
    validation_result["assumptions"] = assumptions
    print("--------------------------")
    print("Validation result:", validation_result)
    print("--------------------------")
    return validation_result


def validate_propensity_score_matching(validation_result: Dict[str, Any], 
                                      dataset_analysis: Dict[str, Any],
                                      variables: Dict[str, Any]) -> None:
    """ 
    Validate propensity score matching method requirements.
    
    Args:
        validation_result: Current validation result to update
        dataset_analysis: Dataset analysis results
        variables: Identified variables
    """ 
    treatment = variables.get("treatment_variable")
    covariates = variables.get("covariates", [])
    
    # Check if treatment is binary using column_categories
    is_binary = dataset_analysis.get("column_categories", {}).get(treatment) == "binary"
    
    # Fallback to check if the column has only two unique values (0 and 1)
    if not is_binary:
        column_types = dataset_analysis.get("column_types", {})
        if column_types.get(treatment) == "int64" or column_types.get(treatment) == "int32":
            # Assuming int type with only 0s and 1s is binary
            is_binary = True
    
    if not is_binary:
        validation_result["valid"] = False
        validation_result["concerns"].append(
            "Treatment variable is not binary, which is required for propensity score matching"
        )
    
    # Check if there are sufficient covariates
    if len(covariates) < 2:
        validation_result["concerns"].append(
            "Few covariates identified, which may limit the effectiveness of propensity score matching"
        )
    
    # Check for sufficient overlap
    variable_relationships = dataset_analysis.get("variable_relationships", {})
    treatment_imbalance = variable_relationships.get("treatment_imbalance", 0.5)
    
    if treatment_imbalance < 0.1 or treatment_imbalance > 0.9:
        validation_result["concerns"].append(
            "Treatment groups are highly imbalanced, which may lead to poor matching quality"
        )
        validation_result["alternative_suggestions"].append("regression_adjustment")


def validate_regression_adjustment(validation_result: Dict[str, Any], 
                                 dataset_analysis: Dict[str, Any],
                                 variables: Dict[str, Any]) -> None:
    """
    Validate regression adjustment method requirements.
    
    Args:
        validation_result: Current validation result to update
        dataset_analysis: Dataset analysis results
        variables: Identified variables
    """
    outcome = variables.get("outcome_variable")
    
    # Check outcome type for appropriate regression model
    outcome_data = dataset_analysis.get("variable_types", {}).get(outcome, {})
    outcome_type = outcome_data.get("type")
    
    if outcome_type == "categorical" and outcome_data.get("n_categories", 0) > 2:
        validation_result["concerns"].append(
            "Outcome is categorical with multiple categories, which may require multinomial regression"
        )
    
    # Check for potential nonlinear relationships
    nonlinear_relationships = dataset_analysis.get("nonlinear_relationships", False)
    
    if nonlinear_relationships:
        validation_result["concerns"].append(
            "Potential nonlinear relationships detected, which may require more flexible models"
        )


def validate_instrumental_variable(validation_result: Dict[str, Any], 
                                 dataset_analysis: Dict[str, Any],
                                 variables: Dict[str, Any]) -> None:
    """
    Validate instrumental variable method requirements.
    
    Args:
        validation_result: Current validation result to update
        dataset_analysis: Dataset analysis results
        variables: Identified variables
    """
    instrument_variable = variables.get("instrument_variable")
    treatment = variables.get("treatment_variable")
    
    if instrument_variable is None:
        validation_result["valid"] = False
        validation_result["concerns"].append(
            "No instrumental variable identified, which is required for this method"
        )
        validation_result["alternative_suggestions"].append("propensity_score_matching")
        return
    
    # Check for instrument strength (correlation with treatment)
    variable_relationships = dataset_analysis.get("variable_relationships", {})
    instrument_correlation = next(
        (corr.get("correlation", 0) for corr in variable_relationships.get("correlations", [])
         if corr.get("var1") == instrument_variable and corr.get("var2") == treatment
         or corr.get("var1") == treatment and corr.get("var2") == instrument_variable),
        0
    )
    
    if abs(instrument_correlation) < 0.2:
        validation_result["concerns"].append(
            "Instrument appears weak (low correlation with treatment), which may lead to bias"
        )
        validation_result["alternative_suggestions"].append("propensity_score_matching")


def validate_difference_in_differences(validation_result: Dict[str, Any], 
                                     dataset_analysis: Dict[str, Any],
                                     variables: Dict[str, Any]) -> None:
    """
    Validate difference-in-differences method requirements.
    
    Args:
        validation_result: Current validation result to update
        dataset_analysis: Dataset analysis results
        variables: Identified variables
    """
    time_variable = variables.get("time_variable")
    group_variable = variables.get("group_variable")
    
    if time_variable is None:
        validation_result["valid"] = False
        validation_result["concerns"].append(
            "No time variable identified, which is required for difference-in-differences"
        )
        validation_result["alternative_suggestions"].append("propensity_score_matching")
    
    if group_variable is None:
        validation_result["valid"] = False
        validation_result["concerns"].append(
            "No group variable identified, which is required for difference-in-differences"
        )
        validation_result["alternative_suggestions"].append("propensity_score_matching")
    
    # Check for parallel trends
    temporal_structure = dataset_analysis.get("temporal_structure", {})
    parallel_trends = temporal_structure.get("parallel_trends", False)
    
    if not parallel_trends:
        validation_result["concerns"].append(
            "No evidence of parallel trends, which is a key assumption for difference-in-differences"
        )
        validation_result["alternative_suggestions"].append("synthetic_control")


def validate_regression_discontinuity(validation_result: Dict[str, Any], 
                                    dataset_analysis: Dict[str, Any],
                                    variables: Dict[str, Any]) -> None:
    """
    Validate regression discontinuity method requirements.
    
    Args:
        validation_result: Current validation result to update
        dataset_analysis: Dataset analysis results
        variables: Identified variables
    """ 
    running_variable = variables.get("running_variable")
    cutoff_value = variables.get("cutoff_value")
    
    if running_variable is None:
        validation_result["valid"] = False
        validation_result["concerns"].append(
            "No running variable identified, which is required for regression discontinuity"
        )
        validation_result["alternative_suggestions"].append("propensity_score_matching")
    
    if cutoff_value is None:
        validation_result["valid"] = False
        validation_result["concerns"].append(
            "No cutoff value identified, which is required for regression discontinuity"
        )
        validation_result["alternative_suggestions"].append("propensity_score_matching")
    
    # Check for discontinuity at threshold
    discontinuities = dataset_analysis.get("discontinuities", {})
    has_discontinuity = discontinuities.get("has_discontinuities", False)
    
    if not has_discontinuity:
        validation_result["valid"] = False
        validation_result["concerns"].append(
            "No clear discontinuity detected at the threshold, which is necessary for this method"
        )
        validation_result["alternative_suggestions"].append("regression_adjustment") 

def validate_backdoor_adjustment(validation_result: Dict[str, Any], 
                               dataset_analysis: Dict[str, Any],
                               variables: Dict[str, Any]) -> None:
    """
    Validate backdoor adjustment method requirements.
    
    Args:
        validation_result: Current validation result to update
        dataset_analysis: Dataset analysis results
        variables: Identified variables
    """
    covariates = variables.get("covariates", [])
    
    if len(covariates) == 0:
        validation_result["valid"] = False
        validation_result["concerns"].append(
            "No covariates identified for backdoor adjustment"
        )
        validation_result["alternative_suggestions"].append("regression_adjustment")


def recommend_alternative(method: str, concerns: List[str], alternatives: List[str]) -> str:
    """
    Recommend an alternative method if the current one has issues.
    
    Args:
        method: Current method
        concerns: List of concerns with the current method
        alternatives: List of alternative methods suggested by the decision tree
        
    Returns:
        String with the recommended method
    """
    # If there are alternatives, recommend the first one
    if alternatives:
        return alternatives[0]
    
    # If no alternatives, use regression adjustment as a fallback
    if method != "regression_adjustment":
        return "regression_adjustment"
    
    # If regression adjustment is also problematic, use propensity score matching
    return "propensity_score_matching" 