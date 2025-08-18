"""
Regression Discontinuity Design (RDD) Estimator.

Tries to use DoWhy's RDD implementation first, falling back to a basic
comparison of linear fits around the cutoff if DoWhy fails.
"""

import pandas as pd
import statsmodels.api as sm
from dowhy import CausalModel
from typing import Dict, Any, List, Optional
import logging
from langchain.chat_models.base import BaseChatModel 

from .diagnostics import run_rdd_diagnostics
from .llm_assist import interpret_rdd_results

logger = logging.getLogger(__name__)

# Attempt to import specific functions from the evan-magnusson/rdd package
_rdd_estimator_func_em = None
_rdd_optimal_bw_func_em = None
_rdd_em_import_error_message = ""
try:
    import rdd
    from rdd import rdd

    logger.info("Successfully imported 'rdd' and 'optimal_bandwidth' from evan-magnusson/rdd package.")
except ImportError as e:
    _rdd_em_import_error_message = f"ImportError for evan-magnusson/rdd: {e}. This package is needed for 'effect_estimate_rdd'."
    logger.warning(_rdd_em_import_error_message)
except Exception as e: # Catch other potential errors during import
    _rdd_em_import_error_message = f"An unexpected error occurred during import from evan-magnusson/rdd: {e}"
    logger.warning(_rdd_em_import_error_message)

def estimate_effect_dowhy(df: pd.DataFrame, treatment: str, outcome: str, running_variable: str, cutoff_value: float, covariates: Optional[List[str]], **kwargs) -> Dict[str, Any]:
    """Estimate RDD effect using DoWhy."""
    logger.info("Attempting RDD estimation using DoWhy.")
    if covariates:
        logger.warning("Covariates provided but may not be used by the DoWhy RDD method_name='rdd'. Support varies.")
    # For DoWhy RDD, we don't typically specify common causes in the model
    # constructor in the same way as backdoor. The running variable is handled
    # via method_params. Covariates might be used by specific underlying estimators
    # if supported, but the basic RDD identification doesn't use them directly.
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        # No explicit graph needed for iv.regression_discontinuity method
    )
    
    # Identify the effect (DoWhy internally identifies RDD as IV)
    # Although potentially redundant if method_name implies identification, 
    # the API requires identified_estimand as the first argument.
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # Estimate using RDD method
    # Note: DoWhy's RDD often has limited direct support for covariates.
    # Bandwidth selection is crucial and often done internally or specified.
    bandwidth = kwargs.get('bandwidth') # Get user-specified bandwidth if provided
    if bandwidth is None:
        # Very basic default bandwidth if none provided - consider better methods
        range_rv = df[running_variable].max() - df[running_variable].min()
        bandwidth = 0.1 * range_rv 
        logger.warning(f"No bandwidth specified, using basic default: {bandwidth:.3f}")

    estimate = model.estimate_effect(
        identified_estimand, # ADD identified_estimand argument
        method_name="iv.regression_discontinuity",
        method_params={
            'rd_variable_name': running_variable,
            'rd_threshold_value': cutoff_value,
            'rd_bandwidth': bandwidth,
            # 'covariates': covariates # Support depends on DoWhy version/estimator
        },
        test_significance=True # Ask DoWhy to calculate p-values if possible
    )
    
    # Extract results - DoWhy's RDD estimate structure might vary
    effect = estimate.value
    # DoWhy's RDD significance testing might be limited/indirect
    # Try to get p-value if estimate object supports it, else None
    p_value = getattr(estimate, 'test_significance_pvalue', None)
    if isinstance(p_value, (list, tuple)):
        p_value = p_value[0] # Handle cases where it might be wrapped
        
    # Confidence intervals might not be directly available from this method easily
    conf_int = getattr(estimate, 'confidence_interval', None)
    std_err = getattr(estimate, 'standard_error', None)

    return {
        'effect_estimate': effect,
        'p_value': p_value,
        'confidence_interval': conf_int,
        'standard_error': std_err,
        'method_details': f"DoWhy RDD (Bandwidth: {bandwidth:.3f})",
    }

def estimate_effect_fallback(df: pd.DataFrame, treatment: str, outcome: str, running_variable: str, cutoff_value: float, covariates: Optional[List[str]], **kwargs) -> Dict[str, Any]:
    """Estimate RDD effect using simple linear regression comparison fallback."""
    logger.warning("DoWhy RDD failed or not used. Falling back to simple linear regression comparison.")
    if covariates:
        logger.warning("Covariates provided but are ignored in the fallback RDD linear regression estimation.")

    bandwidth = kwargs.get('bandwidth')
    if bandwidth is None:
        range_rv = df[running_variable].max() - df[running_variable].min()
        bandwidth = 0.1 * range_rv
        logger.warning(f"No bandwidth specified for fallback, using basic default: {bandwidth:.3f}")

    # Filter data within bandwidth
    df_bw = df[(df[running_variable] >= cutoff_value - bandwidth) & (df[running_variable] <= cutoff_value + bandwidth)].copy()
    if df_bw.empty:
        raise ValueError("No data within the specified bandwidth.")
        
    df_bw['above_cutoff'] = (df_bw[running_variable] >= cutoff_value).astype(int)

    # Define predictors for the regression
    # Interaction term allows different slopes above and below the cutoff
    df_bw['running_centered'] = df_bw[running_variable] - cutoff_value
    df_bw['running_x_above'] = df_bw['running_centered'] * df_bw['above_cutoff']
    predictors = ['above_cutoff', 'running_centered', 'running_x_above']

    # Covariates are NOT included in this basic RDD model
    # if covariates:
        # predictors.extend(covariates) # REMOVED as per user request

    required_cols = [outcome] + predictors
    missing_cols = [col for col in required_cols if col not in df_bw.columns]
    if missing_cols:
         raise ValueError(f"Fallback RDD missing columns: {missing_cols}")
         
    df_analysis = df_bw[required_cols].dropna()
    if df_analysis.empty:
        raise ValueError("No data remaining after dropping NaNs for fallback RDD.")

    X = df_analysis[predictors]
    X = sm.add_constant(X)
    y = df_analysis[outcome]
    
    formula = f"{outcome} ~ {' + '.join(predictors)} + const"
    logger.info(f"Running fallback RDD regression: {formula}")
    
    model = sm.OLS(y, X)
    # Use robust standard errors
    results = model.fit(cov_type='HC1')
    
    # The coefficient for 'above_cutoff' represents the jump at the cutoff
    effect = results.params['above_cutoff']
    p_value = results.pvalues['above_cutoff']
    conf_int = results.conf_int().loc['above_cutoff'].tolist()
    std_err = results.bse['above_cutoff']
    
    return {
        'effect_estimate': effect,
        'p_value': p_value,
        'confidence_interval': conf_int,
        'standard_error': std_err,
        'method_details': f"Fallback Linear Interaction (Bandwidth: {bandwidth:.3f})",
        'formula': formula,
        'model_summary': results.summary()
    }

def effect_estimate_rdd(
    df: pd.DataFrame,
    outcome: str,
    running_variable: str,
    cutoff_value: float,
    treatment: Optional[str] = None, # Kept for API consistency, but unused by evan-magnusson/rdd
    covariates: Optional[List[str]] = None,
    bandwidth: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Estimates RDD effect using the 'evan-magnusson/rdd' package.
    Uses IK optimal bandwidth selection from the same package by default.
    """
    logger.info(f"Attempting RDD estimation using 'evan-magnusson/rdd' for outcome '{outcome}' and running variable '{running_variable}'.")

    

    if treatment:
        logger.info(f"Treatment variable '{treatment}' provided but is not explicitly used by the evan-magnusson/rdd estimation function.")
    if covariates:
        logger.warning("Covariates provided but are ignored by this 'evan-magnusson/rdd' implementation.")

    # --- Bandwidth Selection ---
    final_bandwidth = None
    bandwidth_selection_method = "unknown"

    if bandwidth is not None and bandwidth > 0:
        logger.info(f"Using user-specified bandwidth: {bandwidth:.4f}")
        final_bandwidth = bandwidth
        bandwidth_selection_method = "user-specified"
    else:
        if bandwidth is not None and bandwidth <= 0:
            logger.warning(f"User-specified bandwidth {bandwidth} is not positive. Attempting IK optimal bandwidth selection.")
        try:
            logger.info(f"Attempting IK optimal bandwidth selection using _rdd_optimal_bw_func_em for {outcome} ~ {running_variable} cut at {cutoff_value}.")
            optimal_bw_val = rdd.optimal_bandwidth(df[outcome], df[running_variable], cut=cutoff_value)
            if optimal_bw_val is not None and optimal_bw_val > 0:
                final_bandwidth = optimal_bw_val
                bandwidth_selection_method = "ik_optimal (evan-magnusson/rdd)"
                logger.info(f"IK optimal bandwidth from evan-magnusson/rdd: {final_bandwidth:.4f}")
            else:
                logger.warning(f"IK optimal bandwidth from evan-magnusson/rdd was None or non-positive: {optimal_bw_val}. Falling back to default.")
        except Exception as e:
            logger.warning(f"IK optimal bandwidth selection from evan-magnusson/rdd failed: {e}. Falling back to default.")

        if final_bandwidth is None: # Fallback if user did not specify and IK failed/invalid
            logger.info("Falling back to default bandwidth (10% of running variable range).")
            rv_min = df[running_variable].min()
            rv_max = df[running_variable].max()
            rv_range = rv_max - rv_min
            if rv_range > 0:
                final_bandwidth = 0.1 * rv_range
                bandwidth_selection_method = "default_10_percent_range"
                logger.info(f"Using default 10% range bandwidth: {final_bandwidth:.4f}")
            else:
                err_msg = "Running variable range is not positive. Cannot determine a default bandwidth for evan-magnusson/rdd."
                logger.error(err_msg)
                raise ValueError(err_msg)

    if final_bandwidth is None or final_bandwidth <= 0:
        raise ValueError(f"Could not determine a valid positive bandwidth for evan-magnusson/rdd. Last method: {bandwidth_selection_method}")

    # --- RDD Estimation ---
    try:
        logger.info(f"Running RDD estimation with evan-magnusson/rdd: y='{outcome}', x='{running_variable}', cut={cutoff_value}, bw={final_bandwidth:.4f}")
        # The evan-magnusson/rdd package's rdd function typically handles dataframes directly
        # Ensure correct xname for truncated_data
        data_rdd = rdd.truncated_data(df, running_variable,final_bandwidth, cut=cutoff_value)
        model = rdd.rdd(
            data_rdd, 
            xname=running_variable,  # Correct: Name of the running variable column
            yname=outcome,           # Correct: Name of the outcome variable column
            cut=cutoff_value
        )
        
        # Extract results - this package creates a treatment dummy 'TREATED'
        # The 'model' object has a 'results' attribute which is a statsmodels result instance
        sm_results = model.fit()
        print(sm_results.summary())
        
        # Extract results - using 'TREATED' based on the provided summary output
        effect = sm_results.params.get('TREATED')
        std_err = sm_results.bse.get('TREATED')
        p_value = sm_results.pvalues.get('TREATED')
        
        conf_int_series = sm_results.conf_int()
        conf_int = conf_int_series.loc['TREATED'].tolist() if 'TREATED' in conf_int_series.index else [None, None]

        n_obs = model.nobs # or model.n_ if nobs is not available (check package details)
        
        # The formula is implicit in the local linear regression performed by the package
        # Update to reflect 'TREATED' as the dummy variable name if consistently used by the package
        formula_desc = f"Local linear RDD: {outcome} ~ TREATED + {running_variable}_centered + TREATED*{running_variable}_centered (implicit, from evan-magnusson/rdd)"

        return {
            'effect_estimate': effect,
            'standard_error': std_err,
            'p_value': p_value,
            'confidence_interval': conf_int,
            'method_details': f"RDD (evan-magnusson/rdd package, Bandwidth: {final_bandwidth:.4f})",
            'bandwidth_used': final_bandwidth,
            'bandwidth_selection_method': bandwidth_selection_method,
            'n_obs_in_bandwidth': int(n_obs) if n_obs is not None else None,
            'formula': formula_desc,
            'model_summary': sm_results.summary().as_text() if sm_results else "Summary not available."
        }

    except Exception as e:
        logger.error(f"RDD estimation using 'evan-magnusson/rdd' failed: {e}", exc_info=True)
        # Consider re-raising or returning a more structured error
        raise e # Or return a dict like in the import failure case

def estimate_effect(
    df: pd.DataFrame,
    treatment: str, 
    outcome: str,
    running_variable: str, 
    cutoff_value: float, 
    covariates: Optional[List[str]] = None,
    bandwidth: Optional[float] = None, # Optional bandwidth param
    query: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
    **kwargs # Capture other args like rd_estimator from DoWhy if needed
) -> Dict[str, Any]:
    """
    Estimates the causal effect using Regression Discontinuity Design.

    Tries DoWhy implementation first if use_dowhy=True, otherwise uses fallback.

    Args:
        df: Input DataFrame.
        treatment: Name of the treatment variable (often implicitly defined by cutoff).
                   DoWhy might still need it, fallback doesn't use it directly.
        outcome: Name of the outcome variable.
        running_variable: Name of the variable determining treatment assignment.
        cutoff: The threshold value for the running variable.
        covariates: Optional list of covariate names (support varies).
        bandwidth: Optional bandwidth around the cutoff. If None, a default is used.
        use_dowhy: Whether to attempt using the DoWhy library first.
        query: Optional user query for context.
        llm: Optional Language Model instance.
        **kwargs: Additional keyword arguments for underlying methods.

    Returns:
        Dictionary containing estimation results.
    """
    required_args = { 
        "running_variable": running_variable,
        "cutoff_value": cutoff_value
    }
    if any(val is None for val in required_args.values()):
        raise ValueError(f"Missing required RDD arguments: running_variable and cutoff must be provided.")

    results = {}
    rdd_em_estimation_error = None # Error from effect_estimate_rdd (evan-magnusson)
    fallback_estimation_error = None # Error from estimate_effect_fallback
    
    # --- Try effect_estimate_rdd (evan-magnusson/rdd) First --- 
    try:
        logger.info("Attempting RDD estimation using 'effect_estimate_rdd' (evan-magnusson/rdd package).")
        # Note: treatment is passed but might be unused, covariates are also passed but typically ignored by this specific rdd package
        results = effect_estimate_rdd(
            df, 
            outcome, 
            running_variable, 
            cutoff_value, 
            treatment=treatment, # For API consistency, though evan-magnusson/rdd doesn't use it explicitly
            covariates=covariates, 
            bandwidth=bandwidth, 
            **kwargs
        )
        results['method_used'] = 'evan-magnusson/rdd' # Ensure method_used is set
        logger.info("Successfully estimated effect using 'effect_estimate_rdd'.")
    except ImportError as ie: # Specifically catch import errors for the rdd package
        logger.warning(f"'effect_estimate_rdd' could not run due to ImportError (likely evan-magnusson/rdd package not available/functional): {ie}")
        rdd_em_estimation_error = ie
    except Exception as e:
        logger.warning(f"'effect_estimate_rdd' failed during execution: {e}")
        rdd_em_estimation_error = e
            
    # --- Fallback to estimate_effect_fallback if effect_estimate_rdd failed ---
    if not results: # If effect_estimate_rdd wasn't used or failed
        logger.info("'effect_estimate_rdd' did not produce results. Attempting fallback using 'estimate_effect_fallback'.")
        try:
            fallback_results = estimate_effect_fallback(df, treatment, outcome, running_variable, cutoff_value, covariates, bandwidth=bandwidth, **kwargs)
            results.update(fallback_results)
            results['method_used'] = 'Fallback RDD (Linear Interaction with Robust Errors)'
            fallback_estimation_error = None # Clear fallback error if it succeeded
            logger.info("Successfully estimated effect using 'estimate_effect_fallback'.")
        except Exception as e:
            logger.error(f"Fallback RDD estimation ('estimate_effect_fallback') also failed: {e}")
            fallback_estimation_error = e 

    # Determine final error status
    final_estimation_error = None
    if not results: # If still no results, determine which error to report
        if fallback_estimation_error: # Fallback was attempted and failed
            final_estimation_error = fallback_estimation_error
            logger.error(f"All RDD estimation attempts failed. Last error (from fallback): {final_estimation_error}")
        elif rdd_em_estimation_error: # effect_estimate_rdd was attempted and failed, fallback was not (or also failed but error not captured)
            final_estimation_error = rdd_em_estimation_error
            logger.error(f"All RDD estimation attempts failed. Last error (from effect_estimate_rdd): {final_estimation_error}")
        else:
             logger.error("All RDD estimation attempts failed for an unknown reason.")
        
        if final_estimation_error:
            raise ValueError(f"RDD estimation failed. Last error: {final_estimation_error}")
        else:
            raise ValueError("RDD estimation failed using all available methods for an unknown reason.")
        
    # --- Diagnostics --- 
    try:
        diag_results = run_rdd_diagnostics(df, outcome, running_variable, cutoff_value, covariates, bandwidth)
        results['diagnostics'] = diag_results
    except Exception as diag_e:
        logger.error(f"RDD Diagnostics failed: {diag_e}")
        results['diagnostics'] = {"status": "Failed", "error": str(diag_e)}
        
    # --- Interpretation --- 
    try:
        interpretation = interpret_rdd_results(results, results.get('diagnostics'), llm=llm)
        results['interpretation'] = interpretation
    except Exception as interp_e:
        logger.error(f"RDD Interpretation failed: {interp_e}")
        results['interpretation'] = "Interpretation failed."
        
    # Add info about primary attempt if fallback was used
    if rdd_em_estimation_error and results.get('method_used', '').startswith('Fallback'):
         results['primary_rdd_em_error_info'] = str(rdd_em_estimation_error)

    return results
