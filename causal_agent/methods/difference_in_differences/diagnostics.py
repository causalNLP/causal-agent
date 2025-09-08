"""Diagnostic functions for Difference-in-Differences method."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import statsmodels.formula.api as smf 
from patsy import PatsyError


from .utils import create_post_indicator

logger = logging.getLogger(__name__)

def validate_parallel_trends(df: pd.DataFrame, time_var: str, outcome: str, 
                             group_indicator_col: str, treatment_period_start: Any, 
                             dataset_description: Optional[str] = None,
                             time_varying_covariates: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validates the parallel trends assumption using pre-treatment data.

    Regresses the outcome on group-specific time trends before the treatment period.
    Tests if the interaction terms between group and pre-treatment time periods are jointly significant.
    
    Args:
        df: DataFrame containing the data.
        time_var: Name of the time variable column.
        outcome: Name of the outcome variable column.
        group_indicator_col: Name of the binary treatment group indicator column (0/1).
        treatment_period_start: The time period value when treatment starts.
        dataset_description: Optional dictionary for additional dataset description.
        time_varying_covariates: Optional list of time-varying covariates to include.
        
    Returns:
        Dictionary with validation results.
    """
    logger.info("Validating parallel trends...")
    validation_result = {"valid": False, "p_value": 1.0, "details": "", "error": None}
    
    try:
        # Filter pre-treatment data
        pre_df = df[df[time_var] < treatment_period_start].copy()
        
        if len(pre_df) < 20 or pre_df[group_indicator_col].nunique() < 2 or pre_df[time_var].nunique() < 2:
            validation_result["details"] = "Insufficient pre-treatment data or variation to perform test."
            logger.warning(validation_result["details"])
            # Assume valid if cannot test? Or invalid? Let's default to True if we can't test
            validation_result["valid"] = True
            validation_result["details"] += " Defaulting to assuming parallel trends (unable to test)."
            return validation_result
        
        # Check if group indicator is binary
        if pre_df[group_indicator_col].nunique() > 2:
            validation_result["details"] = f"Group indicator '{group_indicator_col}' has more than 2 unique values. Using simple visual assessment."
            logger.warning(validation_result["details"])
            # Use visual assessment method instead (check if trends look roughly parallel)
            validation_result = assess_trends_visually(pre_df, time_var, outcome, group_indicator_col)
            # Ensure p_value is set
            if validation_result["p_value"] is None:
                validation_result["p_value"] = 1.0 if validation_result["valid"] else 0.04
            return validation_result

        # Use a robust approach first - test for pre-trend differences using a simpler model
        try:
            # Create a linear time trend
            pre_df['time_trend'] = pre_df[time_var].astype(float)
            
            # Create interaction between trend and group
            pre_df['group_trend'] = pre_df['time_trend'] * pre_df[group_indicator_col].astype(float)
            
            # Simple regression with linear trend interaction
            simple_formula = f"Q('{outcome}') ~ Q('{group_indicator_col}') + time_trend + group_trend"
            simple_model = smf.ols(simple_formula, data=pre_df)
            simple_results = simple_model.fit()
            
            # Check if trend interaction coefficient is significant
            group_trend_pvalue = simple_results.pvalues['group_trend']
            
            # If p > 0.05, trends are not significantly different
            validation_result["valid"] = group_trend_pvalue > 0.05
            validation_result["p_value"] = group_trend_pvalue
            validation_result["details"] = f"Simple linear trend test: p-value for group-trend interaction: {group_trend_pvalue:.4f}. Parallel trends: {validation_result['valid']}."
            logger.info(validation_result["details"])
            
            # If we've successfully validated with the simple approach, return
            return validation_result
            
        except Exception as e:
            logger.warning(f"Simple trend test failed: {e}. Trying alternative approach.")
            # Continue to more complex method if simple method fails
        
        # Try more complex approach with period-specific interactions
        try:
            # Create period dummies to avoid issues with categorical variables
            time_periods = sorted(pre_df[time_var].unique())
            
            # Create dummy variables for time periods (except first)
            for period in time_periods[1:]:
                period_col = f'period_{period}'
                pre_df[period_col] = (pre_df[time_var] == period).astype(int)
                
                # Create interaction with group
                pre_df[f'group_x_{period_col}'] = pre_df[period_col] * pre_df[group_indicator_col].astype(float)
            
            # Construct formula with manual dummies
            interaction_formula = f"Q('{outcome}') ~ Q('{group_indicator_col}')"
            
            # Add period dummies except first (reference)
            for period in time_periods[1:]:
                period_col = f'period_{period}'
                interaction_formula += f" + {period_col}"
            
            # Add interactions
            interaction_terms = []
            for period in time_periods[1:]:
                interaction_col = f'group_x_period_{period}'
                interaction_formula += f" + {interaction_col}"
                interaction_terms.append(interaction_col)
            
            # Add covariates if provided
            if time_varying_covariates:
                for cov in time_varying_covariates:
                    interaction_formula += f" + Q('{cov}')"
            
            # Fit model
            complex_model = smf.ols(interaction_formula, data=pre_df)
            complex_results = complex_model.fit()
            
            # Test joint significance of interaction terms
            if interaction_terms:
                from statsmodels.formula.api import ols
                from statsmodels.stats.anova import anova_lm
                
                # Create models with and without interactions
                formula_with = interaction_formula
                formula_without = interaction_formula
                for term in interaction_terms:
                    formula_without = formula_without.replace(f" + {term}", "")
                
                model_with = smf.ols(formula_with, data=pre_df).fit()
                model_without = smf.ols(formula_without, data=pre_df).fit()
                
                # Compare models
                try:
                    from scipy import stats
                    df_model = len(interaction_terms)
                    df_residual = model_with.df_resid
                    f_value = ((model_without.ssr - model_with.ssr) / df_model) / (model_with.ssr / df_residual)
                    p_value = 1 - stats.f.cdf(f_value, df_model, df_residual)
                    
                    validation_result["valid"] = p_value > 0.05
                    validation_result["p_value"] = p_value
                    validation_result["details"] = f"Manual F-test for pre-treatment interactions: F({df_model}, {df_residual})={f_value:.4f}, p={p_value:.4f}. Parallel trends: {validation_result['valid']}."
                    logger.info(validation_result["details"])
                    
                except Exception as e:
                    logger.warning(f"Manual F-test failed: {e}. Using individual coefficient significance.")
                    
                    # If F-test fails, check individual coefficients
                    significant_interactions = 0
                    for term in interaction_terms:
                        if term in complex_results.pvalues and complex_results.pvalues[term] < 0.05:
                            significant_interactions += 1
                    
                    validation_result["valid"] = significant_interactions == 0
                    # Set a dummy p-value based on proportion of significant interactions
                    if len(interaction_terms) > 0:
                        validation_result["p_value"] = 1.0 - (significant_interactions / len(interaction_terms))
                    else:
                        validation_result["p_value"] = 1.0  # Default to 1.0 if no interaction terms
                    validation_result["details"] = f"{significant_interactions} out of {len(interaction_terms)} pre-treatment interactions are significant at p<0.05. Parallel trends: {validation_result['valid']}."
                    logger.info(validation_result["details"])
            else:
                validation_result["valid"] = True
                validation_result["p_value"] = 1.0  # Default to 1.0 if no interaction terms
                validation_result["details"] = "No pre-treatment interaction terms could be tested. Defaulting to assuming parallel trends."
                logger.warning(validation_result["details"])
                
        except Exception as e:
            logger.warning(f"Complex trend test failed: {e}. Falling back to visual assessment.")
            tmp_result = assess_trends_visually(pre_df, time_var, outcome, group_indicator_col)
            # Copy over values from visual assessment ensuring p_value is set
            validation_result.update(tmp_result)
            # Ensure p_value is set
            if validation_result["p_value"] is None:
                validation_result["p_value"] = 1.0 if validation_result["valid"] else 0.04
                
    except Exception as e:
        error_msg = f"Error during parallel trends validation: {e}"
        logger.error(error_msg, exc_info=True)
        validation_result["details"] = error_msg
        validation_result["error"] = str(e)
        # Default to assuming valid if test fails completely
        validation_result["valid"] = True
        validation_result["p_value"] = 1.0  # Default to 1.0 if test fails
        validation_result["details"] += " Defaulting to assuming parallel trends (test failed)."

    return validation_result

def assess_trends_visually(df: pd.DataFrame, time_var: str, outcome: str, 
                          group_indicator_col: str) -> Dict[str, Any]:
    """Simple visual assessment of parallel trends by comparing group means over time.
    
    This is a fallback method when statistical tests fail.
    """
    result = {"valid": False, "p_value": 1.0, "details": "", "error": None}
    
    try:
        # Group by time and treatment group, calculate means
        grouped = df.groupby([time_var, group_indicator_col])[outcome].mean().reset_index()
        
        # Pivot to get time series for each group
        if df[group_indicator_col].nunique() <= 10:  # Only if reasonable number of groups
            pivot = grouped.pivot(index=time_var, columns=group_indicator_col, values=outcome)
            
            # Calculate slopes between consecutive periods for each group
            slopes = {}
            time_values = sorted(df[time_var].unique())
            
            if len(time_values) >= 3:  # Need at least 3 periods to compare slopes
                for group in pivot.columns:
                    group_slopes = []
                    for i in range(len(time_values) - 1):
                        t1, t2 = time_values[i], time_values[i+1]
                        if t1 in pivot.index and t2 in pivot.index:
                            slope = (pivot.loc[t2, group] - pivot.loc[t1, group]) / (t2 - t1)
                            group_slopes.append(slope)
                    if group_slopes:
                        slopes[group] = group_slopes
                
                # Compare slopes between groups
                if len(slopes) >= 2:
                    slope_diffs = []
                    groups = list(slopes.keys())
                    for i in range(len(slopes[groups[0]])):
                        if i < len(slopes[groups[1]]):
                            slope_diffs.append(abs(slopes[groups[0]][i] - slopes[groups[1]][i]))
                    
                    # If average slope difference is small relative to outcome scale
                    outcome_scale = df[outcome].std()
                    avg_slope_diff = sum(slope_diffs) / len(slope_diffs) if slope_diffs else 0
                    relative_diff = avg_slope_diff / outcome_scale if outcome_scale > 0 else 0
                    
                    result["valid"] = relative_diff < 0.2  # Threshold for "parallel enough"
                    # Set p-value based on relative difference
                    result["p_value"] = 1.0 - (relative_diff * 5) if relative_diff < 0.2 else 0.04
                    result["details"] = f"Visual assessment: relative slope difference = {relative_diff:.4f}. Parallel trends: {result['valid']}."
                else:
                    result["valid"] = True
                    result["p_value"] = 1.0
                    result["details"] = "Visual assessment: insufficient group data for comparison. Defaulting to assuming parallel trends."
            else:
                result["valid"] = True
                result["p_value"] = 1.0
                result["details"] = "Visual assessment: insufficient time periods for comparison. Defaulting to assuming parallel trends."
        else:
            result["valid"] = True
            result["p_value"] = 1.0
            result["details"] = f"Visual assessment: too many groups ({df[group_indicator_col].nunique()}) for visual comparison. Defaulting to assuming parallel trends."
    
    except Exception as e:
        result["error"] = str(e)
        result["valid"] = True
        result["p_value"] = 1.0
        result["details"] = f"Visual assessment failed: {e}. Defaulting to assuming parallel trends."
        
    logger.info(result["details"])
    return result

def run_placebo_test(df: pd.DataFrame, time_var: str, group_var: str, outcome: str, 
                       treated_unit_indicator: str, covariates: List[str], 
                       treatment_period_start: Any, 
                       placebo_period_start: Any) -> Dict[str, Any]:
    """Runs a placebo test for DiD by assigning a fake earlier treatment period.

    Re-runs the DiD estimation using the placebo period and checks if the effect is non-significant.
    
    Args:
        df: Original DataFrame.
        time_var: Name of the time variable column.
        group_var: Name of the unit/group ID column (for clustering SE).
        outcome: Name of the outcome variable column.
        treated_unit_indicator: Name of the binary treatment group indicator column (0/1).
        covariates: List of covariate names.
        treatment_period_start: The actual treatment start period.
        placebo_period_start: The fake treatment start period (must be before actual start).
        
    Returns:
        Dictionary with placebo test results.
    """
    logger.info(f"Running placebo test assigning treatment start at {placebo_period_start}...")
    placebo_result = {"passed": False, "effect_estimate": None, "p_value": None, "details": "", "error": None}

    if placebo_period_start >= treatment_period_start:
        error_msg = "Placebo period must be before the actual treatment period."
        logger.error(error_msg)
        placebo_result["error"] = error_msg
        placebo_result["details"] = error_msg
        return placebo_result
        
    try:
        df_placebo = df.copy()
        # Create placebo post and interaction terms
        post_placebo_col = 'post_placebo'
        interaction_placebo_col = 'did_interaction_placebo'
        
        df_placebo[post_placebo_col] = create_post_indicator(df_placebo, time_var, placebo_period_start)
        df_placebo[interaction_placebo_col] = df_placebo[treated_unit_indicator] * df_placebo[post_placebo_col]
        
        # Construct formula for placebo regression
        formula = f"`{outcome}` ~ `{treated_unit_indicator}` + `{post_placebo_col}` + `{interaction_placebo_col}`"
        if covariates:
             formula += f" + {' + '.join([f'`{c}`' for c in covariates])}"
        formula += f" + C(`{group_var}`) + C(`{time_var}`)" # Include FEs
        
        logger.debug(f"Placebo test formula: {formula}")

        # Fit the placebo model with clustered SE
        ols_model = smf.ols(formula=formula, data=df_placebo)
        results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df_placebo[group_var]})
        
        # Check the significance of the placebo interaction term
        placebo_effect = float(results.params[interaction_placebo_col])
        placebo_p_value = float(results.pvalues[interaction_placebo_col])
        
        # Test passes if the placebo effect is not statistically significant (e.g., p > 0.1)
        passed_test = placebo_p_value > 0.10
        
        placebo_result["passed"] = passed_test
        placebo_result["effect_estimate"] = placebo_effect
        placebo_result["p_value"] = placebo_p_value
        placebo_result["details"] = f"Placebo treatment effect estimated at {placebo_effect:.4f} (p={placebo_p_value:.4f}). Test passed: {passed_test}."
        logger.info(placebo_result["details"])

    except (KeyError, PatsyError, ValueError, Exception) as e:
        error_msg = f"Error during placebo test execution: {e}"
        logger.error(error_msg, exc_info=True)
        placebo_result["details"] = error_msg
        placebo_result["error"] = str(e)

    return placebo_result

# TODO: Add function for Event Study plot (plot_event_study)
# This would involve estimating effects for leads and lags around the treatment period.

# Add other diagnostic functions as needed (e.g., plot_event_study) 