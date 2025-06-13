"""
Explanation generator component for causal inference methods.

This module generates explanations for causal inference methods, including
what the method does, its assumptions, and how it will be applied to the dataset.
"""

from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseChatModel # For LLM type hint


def generate_explanation(
    method_info: Dict[str, Any],
    validation_result: Dict[str, Any],
    variables: Dict[str, Any],
    results: Dict[str, Any],
    dataset_analysis: Optional[Dict[str, Any]] = None,
    dataset_description: Optional[str] = None,
    llm: Optional[BaseChatModel] = None
) -> Dict[str, str]:
    """
    Generates a comprehensive explanation text for the causal analysis.

    Args:
        method_info: Dictionary containing selected method details.
        validation_result: Dictionary containing method validation results.
        variables: Dictionary containing identified variables.
        results: Dictionary containing numerical results from the method execution.
        dataset_analysis: Optional dictionary with dataset analysis details.
        dataset_description: Optional string describing the dataset.
        llm: Optional language model instance (for potential future use in generation).
        
    Returns:
        Dictionary containing the final explanation text.
    """
    method = method_info.get("method")
    
    # Handle potential None for validation_result
    if validation_result and validation_result.get("valid") is False:
        method = validation_result.get("recommended_method", method)
    
    # Get components
    method_explanation = get_method_explanation(method)
    assumption_explanations = explain_assumptions(method_info.get("assumptions", []))
    application_explanation = explain_application(method, variables.get("treatment_variable"), 
                                                variables.get("outcome_variable"), 
                                                variables.get("covariates", []), variables)
    limitations_explanation = explain_limitations(method, validation_result.get("concerns", []) if validation_result else [])
    interpretation_guide = generate_interpretation_guide(method, variables.get("treatment_variable"), 
                                                       variables.get("outcome_variable"))

    # --- Extract Numerical Results --- 
    effect_estimate = results.get("effect_estimate")
    effect_se = results.get("effect_se")
    ci = results.get("confidence_interval")
    p_value = results.get("p_value") # Assuming method executor returns p_value

    # --- Assemble Final Text --- 
    final_text = f"**Method Used:** {method_info.get('method_name', method)}\n\n"
    final_text += f"**Method Explanation:**\n{method_explanation}\n\n"
    
    # Add Results Section
    final_text += "**Results:**\n"
    if effect_estimate is not None:
        final_text += f"- Estimated Causal Effect: {effect_estimate:.4f}\n"
    if effect_se is not None:
         final_text += f"- Standard Error: {effect_se:.4f}\n"
    if ci and ci[0] is not None and ci[1] is not None:
         final_text += f"- 95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]\n"
    if p_value is not None:
         final_text += f"- P-value: {p_value:.4f}\n"
    final_text += "\n"
    
    final_text += f"**Interpretation Guide:**\n{interpretation_guide}\n\n"
    final_text += f"**Assumptions:**\n"
    for item in assumption_explanations:
        final_text += f"- {item['assumption']}: {item['explanation']}\n"
    final_text += "\n"
    final_text += f"**Limitations:**\n{limitations_explanation}\n\n"

    return {
        "final_explanation_text": final_text
        # Return only the final text, the tool wrapper adds workflow state
    }


def get_method_explanation(method: str) -> str:
    """
    Get explanation for what the method does.
    
    Args:
        method: Causal inference method name
        
    Returns:
        String explaining what the method does
    """
    explanations = {
        "propensity_score_matching": (
            "Propensity Score Matching is a statistical technique that attempts to estimate the effect "
            "of a treatment by accounting for covariates that predict receiving the treatment. "
            "It creates matched sets of treated and untreated subjects who share similar characteristics, "
            "allowing for a more fair comparison between groups."
        ),
        "regression_adjustment": (
            "Regression Adjustment is a method that uses regression models to estimate causal effects "
            "by controlling for covariates. It models the outcome as a function of the treatment and "
            "other potential confounding variables, allowing the isolation of the treatment effect."
        ),
        "instrumental_variable": (
            "The Instrumental Variable method addresses issues of endogeneity or unmeasured confounding "
            "by using an 'instrument' - a variable that affects the treatment but not the outcome directly. "
            "It effectively finds the natural experiment hidden in your data to estimate causal effects."
        ),
        "difference_in_differences": (
            "Difference-in-Differences compares the changes in outcomes over time between a group that "
            "receives a treatment and a group that does not. It controls for time-invariant unobserved "
            "confounders by looking at differences in trends rather than absolute values."
        ),
        "regression_discontinuity": (
            "Regression Discontinuity Design exploits a threshold or cutoff rule that determines treatment "
            "assignment. By comparing observations just above and below this threshold, where treatment "
            "status changes but other characteristics remain similar, it estimates the local causal effect."
        ),
        "backdoor_adjustment": (
            "Backdoor Adjustment controls for confounding variables that create 'backdoor paths' between "
            "treatment and outcome variables in a causal graph. By conditioning on these variables, "
            "it blocks the non-causal associations, allowing for identification of the causal effect."
        ),
    }
    
    return explanations.get(method, 
        f"The {method} method is a causal inference technique used to estimate "
        f"causal effects from observational data.")


def explain_assumptions(assumptions: List[str]) -> List[Dict[str, str]]:
    """
    Explain each assumption of the method.
    
    Args:
        assumptions: List of assumption names
        
    Returns:
        List of dictionaries with assumption name and explanation
    """
    assumption_details = {
        "Treatment is randomly assigned": (
            "This assumes that treatment assignment is not influenced by any factors "
            "related to the outcome, similar to a randomized controlled trial. "
            "In observational data, this assumption rarely holds without conditioning on confounders."
        ),
        "No systematic differences between treatment and control groups": (
            "Treatment and control groups should be balanced on all relevant characteristics "
            "except for the treatment itself. Any systematic differences could bias the estimate."
        ),
        "No unmeasured confounders (conditional ignorability)": (
            "All variables that simultaneously affect the treatment and outcome are measured and "
            "included in the analysis. If important confounders are missing, the estimated causal "
            "effect will be biased."
        ),
        "Sufficient overlap between treatment and control groups": (
            "For each combination of covariate values, there should be both treated and untreated "
            "units. Without overlap, the model must extrapolate, which can lead to biased estimates."
        ),
        "Treatment assignment is not deterministic given covariates": (
            "No combination of covariates should perfectly predict treatment assignment. "
            "If treatment is deterministic for some units, causal comparisons become impossible."
        ),
        "Instrument is correlated with treatment (relevance)": (
            "The instrumental variable must have a clear and preferably strong effect on the "
            "treatment variable. Weak instruments lead to imprecise and potentially biased estimates."
        ),
        "Instrument affects outcome only through treatment (exclusion restriction)": (
            "The instrumental variable must not directly affect the outcome except through its "
            "effect on the treatment. If this assumption fails, the causal estimate will be biased."
        ),
        "Instrument is as good as randomly assigned (exogeneity)": (
            "The instrumental variable must not be correlated with any confounders of the "
            "treatment-outcome relationship. It should be as good as randomly assigned."
        ),
        "Parallel trends between treatment and control groups": (
            "In the absence of treatment, the difference between treatment and control groups "
            "would have remained constant over time. This is the key identifying assumption for "
            "difference-in-differences and cannot be directly tested for the post-treatment period."
        ),
        "No spillover effects between groups": (
            "The treatment of one unit should not affect the outcomes of other units. "
            "If spillovers exist, they can bias the estimated treatment effect."
        ),
        "No anticipation effects before treatment": (
            "Units should not change their behavior in anticipation of future treatment. "
            "If anticipation effects exist, the pre-treatment trends may already reflect treatment effects."
        ),
        "Stable composition of treatment and control groups": (
            "The composition of treatment and control groups should remain stable over time. "
            "If units move between groups based on outcomes, this can bias the estimates."
        ),
        "Units cannot precisely manipulate their position around the cutoff": (
            "In regression discontinuity, units must not be able to precisely control their position "
            "relative to the cutoff. If they can, the randomization-like property of the design fails."
        ),
        "No other variables change discontinuously at the cutoff": (
            "Any discontinuity in outcomes at the cutoff should be attributable only to the change "
            "in treatment status. If other relevant variables also change at the cutoff, the causal "
            "interpretation is compromised."
        ),
        "The relationship between running variable and outcome is continuous at the cutoff": (
            "In the absence of treatment, the relationship between the running variable and the "
            "outcome would be continuous at the cutoff. This allows attributing any observed "
            "discontinuity to the treatment effect."
        ),
        "The model correctly specifies the relationship between variables": (
            "The functional form of the relationship between variables in the model should correctly "
            "capture the true relationship in the data. Misspecification can lead to biased estimates."
        ),
        "No reverse causality": (
            "The treatment must cause the outcome, not the other way around. If the outcome affects "
            "the treatment, the estimated relationship will not have a causal interpretation."
        ),
    }
    
    return [
        {"assumption": assumption, "explanation": assumption_details.get(assumption, 
            "This is a key assumption for the selected causal inference method.")}
        for assumption in assumptions
    ]


def explain_application(method: str, treatment: str, outcome: str, 
                      covariates: List[str], variables: Dict[str, Any]) -> str:
    """
    Explain how the method will be applied to the dataset.
    
    Args:
        method: Causal inference method name
        treatment: Treatment variable name
        outcome: Outcome variable name
        covariates: List of covariate names
        variables: Dictionary of identified variables
        
    Returns:
        String explaining the application
    """
    covariate_str = ", ".join(covariates[:3])
    if len(covariates) > 3:
        covariate_str += f", and {len(covariates) - 3} other variables"
    
    applications = {
        "propensity_score_matching": (
            f"I will estimate the propensity scores (probability of receiving treatment) for each "
            f"observation based on the covariates ({covariate_str}). Then, I'll match treated and "
            f"untreated units with similar propensity scores to create balanced comparison groups. "
            f"Finally, I'll calculate the difference in {outcome} between these matched groups to "
            f"estimate the causal effect of {treatment}."
        ),
        "regression_adjustment": (
            f"I will build a regression model with {outcome} as the dependent variable and "
            f"{treatment} as the independent variable of interest, while controlling for "
            f"potential confounders ({covariate_str}). The coefficient of {treatment} will "
            f"represent the estimated causal effect after adjusting for these covariates."
        ),
        "instrumental_variable": (
            f"I will use {variables.get('instrument_variable')} as an instrumental variable for "
            f"{treatment}. First, I'll estimate how the instrument affects {treatment} (first stage). "
            f"Then, I'll use these predictions to estimate how changes in {treatment} that are induced "
            f"by the instrument affect {outcome} (second stage). This two-stage approach helps "
            f"address potential unmeasured confounding."
        ),
        "difference_in_differences": (
            f"I will compare the change in {outcome} before and after the intervention for the "
            f"group receiving {treatment}, relative to the change in a control group that didn't "
            f"receive the treatment. This approach controls for time-invariant confounders and "
            f"common time trends that affect both groups."
        ),
        "regression_discontinuity": (
            f"I will focus on observations close to the cutoff value "
            f"({variables.get('cutoff_value')}) of the running variable "
            f"({variables.get('running_variable')}), where treatment assignment changes. "
            f"By comparing outcomes just above and below this threshold, I can estimate "
            f"the local causal effect of {treatment} on {outcome}."
        ),
        "backdoor_adjustment": (
            f"I will control for the identified confounding variables ({covariate_str}) to "
            f"block all backdoor paths between {treatment} and {outcome}. This may involve "
            f"stratification, regression adjustment, or inverse probability weighting, depending "
            f"on the data characteristics."
        ),
    }
    
    return applications.get(method, 
        f"I will apply the {method} method to estimate the causal effect of "
        f"{treatment} on {outcome}, controlling for relevant confounding factors "
        f"where appropriate.")


def explain_limitations(method: str, concerns: List[str]) -> str:
    """
    Explain the limitations of the method based on validation concerns.
    
    Args:
        method: Causal inference method name
        concerns: List of concerns from validation
        
    Returns:
        String explaining the limitations
    """
    method_limitations = {
        "propensity_score_matching": (
            "Propensity Score Matching can only account for observed confounders, and its "
            "effectiveness depends on having good overlap between treatment and control groups. "
            "It may also be sensitive to model specification for the propensity score estimation."
        ),
        "regression_adjustment": (
            "Regression Adjustment relies heavily on correct model specification and can only "
            "control for observed confounders. Extrapolation to regions with limited data can lead "
            "to unreliable estimates, and the method may be sensitive to outliers."
        ),
        "instrumental_variable": (
            "Instrumental Variable estimation can be imprecise with weak instruments and is "
            "sensitive to violations of the exclusion restriction. The estimated effect is a local "
            "average treatment effect for 'compliers', which may not generalize to the entire population."
        ),
        "difference_in_differences": (
            "Difference-in-Differences relies on the parallel trends assumption, which cannot be fully "
            "tested for the post-treatment period. It may be sensitive to the choice of comparison group "
            "and can be biased if there are time-varying confounders or anticipation effects."
        ),
        "regression_discontinuity": (
            "Regression Discontinuity provides estimates that are local to the cutoff point and may not "
            "generalize to units far from this threshold. It also requires sufficient data around the "
            "cutoff and is sensitive to the choice of bandwidth and functional form."
        ),
        "backdoor_adjustment": (
            "Backdoor Adjustment requires correctly identifying all confounding variables and their "
            "relationships. It depends on the assumption of no unmeasured confounders and may be "
            "sensitive to model misspecification in complex settings."
        ),
    }
    
    base_limitation = method_limitations.get(method, 
        f"The {method} method has general limitations in terms of its assumptions and applicability.")
    
    # Add specific concerns if any
    if concerns:
        concern_text = " Additionally, specific concerns for this analysis include: " + \
                      "; ".join(concerns) + "."
        return base_limitation + concern_text
    
    return base_limitation


def generate_interpretation_guide(method: str, treatment: str, outcome: str) -> str:
    """
    Generate guide for interpreting the results.
    
    Args:
        method: Causal inference method name
        treatment: Treatment variable name
        outcome: Outcome variable name
        
    Returns:
        String with interpretation guide
    """
    interpretation_guides = {
        "propensity_score_matching": (
            f"The estimated effect represents the Average Treatment Effect (ATE) or the Average "
            f"Treatment Effect on the Treated (ATT), depending on the specific matching approach. "
            f"It can be interpreted as the expected change in {outcome} if a unit were to receive "
            f"{treatment}, compared to not receiving it, for units with similar covariate values."
        ),
        "regression_adjustment": (
            f"The coefficient of {treatment} in the regression model represents the estimated "
            f"average causal effect on {outcome}, holding all included covariates constant. "
            f"For binary treatments, it's the expected difference in outcomes between treated "
            f"and untreated units with the same covariate values."
        ),
        "instrumental_variable": (
            f"The estimated effect represents the Local Average Treatment Effect (LATE) for 'compliers' "
            f"- units whose treatment status is influenced by the instrument. It can be interpreted as "
            f"the average effect of {treatment} on {outcome} for this specific subpopulation."
        ),
        "difference_in_differences": (
            f"The estimated effect represents the average causal impact of {treatment} on {outcome}, "
            f"under the assumption that treatment and control groups would have followed parallel "
            f"trends in the absence of treatment. It accounts for both time-invariant differences "
            f"between groups and common time trends."
        ),
        "regression_discontinuity": (
            f"The estimated effect represents the local causal impact of {treatment} on {outcome} "
            f"at the cutoff point. It can be interpreted as the expected difference in outcomes "
            f"for units just above versus just below the threshold, where treatment status changes."
        ),
        "backdoor_adjustment": (
            f"The estimated effect represents the average causal effect of {treatment} on {outcome} "
            f"after controlling for all identified confounding variables. It can be interpreted as "
            f"the expected difference in outcomes if a unit were to receive versus not receive the "
            f"treatment, holding all confounding factors constant."
        ),
    }
    
    return interpretation_guides.get(method, 
        f"The estimated effect represents the causal impact of {treatment} on {outcome}, "
        f"given the assumptions of the method are met. Careful consideration of these "
        f"assumptions is needed for valid causal interpretation.") 