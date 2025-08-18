import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from dowhy import CausalModel
from typing import Dict, Any, List, Union, Optional
import logging
from langchain.chat_models.base import BaseChatModel

from .diagnostics import run_iv_diagnostics
from .llm_assist import identify_instrument_variable, validate_instrument_assumptions_qualitative, interpret_iv_results

logger = logging.getLogger(__name__)

def build_iv_graph_gml(treatment: str, outcome: str, instruments: List[str], covariates: List[str]) -> str:
    """
    Constructs a GML string representing the causal graph for IV.

    Assumptions:
    - Instruments cause Treatment
    - Covariates cause Treatment and Outcome
    - Treatment causes Outcome
    - Instruments do NOT directly cause Outcome (Exclusion)
    - Instruments are NOT caused by Covariates (can be relaxed if needed)
    - Unobserved Confounder (U) affects Treatment and Outcome

    Args:
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        instruments: List of instrument variable names.
        covariates: List of covariate names.

    Returns:
        A GML graph string.
    """
    nodes = []
    edges = []

    # Define nodes - ensure no duplicates if a variable is both instrument and covariate (SHOULD NOT HAPPEN)
    # Use a set to ensure unique variable names
    all_vars_set = set([treatment, outcome] + instruments + covariates + ['U'])
    all_vars = list(all_vars_set)
    
    for var in all_vars:
        nodes.append(f'node [ id "{var}" label "{var}" ]')

    # Define edges
    # Instruments -> Treatment
    for inst in instruments:
        edges.append(f'edge [ source "{inst}" target "{treatment}" ]')

    # Covariates -> Treatment
    for cov in covariates:
        # Ensure we don't add self-loops or duplicate edges if cov == treatment (shouldn't happen)
        if cov != treatment: 
            edges.append(f'edge [ source "{cov}" target "{treatment}" ]')

    # Covariates -> Outcome
    for cov in covariates:
         if cov != outcome:
            edges.append(f'edge [ source "{cov}" target "{outcome}" ]')

    # Treatment -> Outcome
    edges.append(f'edge [ source "{treatment}" target "{outcome}" ]')

    # Unobserved Confounder -> Treatment and Outcome
    edges.append(f'edge [ source "U" target "{treatment}" ]')
    edges.append(f'edge [ source "U" target "{outcome}" ]')

    # Core IV Assumption: Instruments are NOT caused by U (implicitly handled by not adding edge)
    # Core IV Assumption: Instruments do NOT directly cause Outcome (handled by not adding edge)

    # Format nodes and edges with indentation before inserting into f-string
    formatted_nodes = '\n  '.join(nodes)
    formatted_edges = '\n  '.join(edges)

    gml_string = f"""
graph [
  directed 1
  {formatted_nodes}
  {formatted_edges}
]
"""
    # Convert print to logger
    logger.debug("\n--- Generated GML Graph ---")
    logger.debug(gml_string)
    logger.debug("-------------------------\n")
    return gml_string

def format_iv_results(estimate: Optional[float], raw_results: Dict, diagnostics: Dict, treatment: str, outcome: str, instrument: List[str], method_used: str, llm: Optional[BaseChatModel] = None) -> Dict[str, Any]:
    """
    Formats the results from IV estimation into a standardized dictionary.

    Args:
        estimate: The point estimate of the causal effect.
        raw_results: Dictionary containing raw outputs from DoWhy/statsmodels.
        diagnostics: Dictionary containing diagnostic results.
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        instrument: List of instrument variable names.
        method_used: 'dowhy' or 'statsmodels'.
        llm: Optional LLM instance for interpretation.

    Returns:
        Standardized results dictionary.
    """
    formatted = {
        "effect_estimate": estimate,
        "treatment_variable": treatment,
        "outcome_variable": outcome,
        "instrument_variables": instrument,
        "method_used": method_used,
        "diagnostics": diagnostics,
        "raw_results": {k: str(v) for k, v in raw_results.items() if "object" not in k}, # Avoid serializing large objects
        "confidence_interval": None,
        "standard_error": None,
        "p_value": None,
        "interpretation": "Placeholder"
    }

    # Extract details from statsmodels results if available
    sm_results = raw_results.get('statsmodels_results_object')
    if method_used == 'statsmodels' and sm_results:
        try:
            # Use .bse for standard error in statsmodels results
            formatted["standard_error"] = float(sm_results.bse[treatment])
            formatted["p_value"] = float(sm_results.pvalues[treatment])
            conf_int = sm_results.conf_int().loc[treatment].tolist()
            formatted["confidence_interval"] = [float(ci) for ci in conf_int]
        except AttributeError as e:
            logger.warning(f"Could not extract all details from statsmodels results object (likely missing attribute): {e}")
        except Exception as e:
            logger.warning(f"Error extracting details from statsmodels results: {e}")

    # Extract details from DoWhy results if available
    # Note: DoWhy's CausalEstimate object structure needs inspection
    dw_results = raw_results.get('dowhy_results_object')
    if method_used == 'dowhy' and dw_results:
         try:
             # Attempt common attributes, may need adjustment based on DoWhy version/output
             if hasattr(dw_results, 'stderr'):
                 formatted["standard_error"] = float(dw_results.stderr)
             if hasattr(dw_results, 'p_value'):
                  formatted["p_value"] = float(dw_results.p_value)
             if hasattr(dw_results, 'conf_intervals'):
                 # Assuming it's stored similarly to statsmodels, might need adjustment
                 ci = dw_results.conf_intervals().loc[treatment].tolist() # Fictional attribute/method - check DoWhy docs!
                 formatted["confidence_interval"] = [float(c) for c in ci]
             elif hasattr(dw_results, 'get_confidence_intervals'):
                  ci = dw_results.get_confidence_intervals() # Check DoWhy docs for format
                  # Check format of ci before converting
                  if isinstance(ci, (list, tuple)) and len(ci) == 2:
                      formatted["confidence_interval"] = [float(c) for c in ci] # Adapt parsing
                  else:
                      logger.warning(f"Could not parse confidence intervals from DoWhy object: {ci}")

         except Exception as e:
             logger.warning(f"Could not extract all details from DoWhy results: {e}. Structure might be different.", exc_info=True)
             # Avoid printing dir in production code, use logger.debug if needed for dev
             # logger.debug(f"DoWhy result object dir(): {dir(dw_results)}")

    # Generate LLM interpretation - pass llm object
    if estimate is not None:
        formatted["interpretation"] = interpret_iv_results(formatted, diagnostics, llm=llm)
    else:
        formatted["interpretation"] = "Estimation failed, cannot interpret results."


    return formatted

def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str],
    query: Optional[str] = None, 
    dataset_description: Optional[str] = None, 
    llm: Optional[BaseChatModel] = None, 
    **kwargs
) -> Dict[str, Any]:
    
    instrument = kwargs.get('instrument_variable')
    if not instrument:
        return {"error": "Instrument variable ('instrument_variable') not found in kwargs.", "method_used": "none", "diagnostics": {}}
        
    instrument_list = [instrument] if isinstance(instrument, str) else instrument
    valid_instruments = [inst for inst in instrument_list if isinstance(inst, str)]
    clean_covariates = [cov for cov in covariates if cov not in valid_instruments]
    
    logger.info(f"\n--- Starting Instrumental Variable Estimation ---")
    logger.info(f"Treatment: {treatment}, Outcome: {outcome}, Instrument(s): {valid_instruments}, Original Covariates: {covariates}, Cleaned Covariates: {clean_covariates}")
    results = {}
    method_used = "none"
    sm_results_obj = None
    dw_results_obj = None
    identified_estimand = None # Initialize
    model = None             # Initialize
    refutation_results = {}  # Initialize

    # --- Input Validation --- 
    required_cols = [treatment, outcome] + valid_instruments + clean_covariates
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {"error": f"Missing required columns in DataFrame: {missing_cols}", "method_used": method_used, "diagnostics": {}}
    if not valid_instruments:
        return {"error": "Instrument variable(s) must be provided and valid.", "method_used": method_used, "diagnostics": {}}

    # --- LLM Pre-Checks --- 
    if query and llm:
        qualitative_check = validate_instrument_assumptions_qualitative(treatment, outcome, valid_instruments, clean_covariates, query, llm=llm)
        results['llm_assumption_check'] = qualitative_check
        logger.info(f"LLM Qualitative Assumption Check: {qualitative_check}")
        
    # --- Build Graph and Instantiate CausalModel (Do this before estimation attempts) ---
    # This allows using identify_effect and refute_estimate even if DoWhy estimation fails
    try:
        graph = build_iv_graph_gml(treatment, outcome, valid_instruments, clean_covariates)
        if not graph:
            raise ValueError("Failed to build GML graph for DoWhy.")
        
        model = CausalModel(data=df, treatment=treatment, outcome=outcome, graph=graph)
        
        # Identify Effect (essential for refutation later)
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        logger.debug("\nDoWhy Identified Estimand:")
        logger.debug(identified_estimand)
        if not identified_estimand:
             raise ValueError("DoWhy could not identify a valid estimand.")
             
    except Exception as model_init_e:
        logger.error(f"Failed to initialize CausalModel or identify effect: {model_init_e}", exc_info=True)
        # Cannot proceed without model/estimand for DoWhy or refutation
        results['error'] = f"Failed to initialize CausalModel: {model_init_e}"
        # Attempt statsmodels anyway? Or return error? Let's try statsmodels.
        pass # Allow falling through to statsmodels if desired

    # --- Primary Path: DoWhy Estimation --- 
    if model and identified_estimand and not kwargs.get('force_statsmodels', False):
        logger.info("\nAttempting estimation with DoWhy...")
        try:
            dw_results_obj = model.estimate_effect(
                identified_estimand,
                method_name="iv.instrumental_variable", 
                method_params={'iv_instrument_name': valid_instruments} 
            )
            logger.debug("\nDoWhy Estimation Result:")
            logger.debug(dw_results_obj)
            results['dowhy_estimate'] = dw_results_obj.value
            results['dowhy_results_object'] = dw_results_obj
            method_used = 'dowhy'
            logger.info("DoWhy estimation successful.")
        except Exception as e:
            logger.error(f"DoWhy IV estimation failed: {e}", exc_info=True)
            results['dowhy_error'] = str(e)
            if not kwargs.get('allow_fallback', True):
                 logger.warning("Fallback to statsmodels disabled. Estimation failed.")
                 method_used = "dowhy_failed"
                 # Still run diagnostics and format output
            else:
                logger.info("Proceeding to statsmodels fallback.")
    elif not model or not identified_estimand:
         logger.warning("Skipping DoWhy estimation due to CausalModel initialization/identification failure.")
         # Ensure we proceed to statsmodels if fallback is allowed
         if not kwargs.get('allow_fallback', True):
             logger.error("Cannot estimate effect: CausalModel failed and fallback disabled.")
             method_used = "dowhy_failed"
         else:
              logger.info("Proceeding to statsmodels fallback.")

    # --- Fallback Path: statsmodels IV2SLS --- 
    if method_used not in ['dowhy', 'dowhy_failed']:
        logger.info("\nAttempting estimation with statsmodels IV2SLS...")
        try:
            df_copy = df.copy().dropna(subset=required_cols) 
            if df_copy.empty:
                 raise ValueError("DataFrame becomes empty after dropping NAs in required columns.")
            df_copy['intercept'] = 1
            exog_regressors = ['intercept'] + clean_covariates
            endog_var = treatment 
            all_instruments_for_sm = list(dict.fromkeys(exog_regressors + valid_instruments))
            endog_data = df_copy[outcome]
            exog_data_sm_cols = list(dict.fromkeys(exog_regressors + [endog_var]))
            exog_data_sm = df_copy[exog_data_sm_cols]
            instrument_data_sm = df_copy[all_instruments_for_sm]
            num_endog = 1 
            num_external_iv = len(valid_instruments) 
            if num_endog > num_external_iv:
                 raise ValueError(f"Model underidentified: More endogenous regressors ({num_endog}) than unique external instruments ({num_external_iv}).")
            iv_model = IV2SLS(endog=endog_data, exog=exog_data_sm, instrument=instrument_data_sm)
            sm_results_obj = iv_model.fit()
            logger.info("\nStatsmodels Estimation Summary:")
            logger.info(f"  Estimate for {treatment}: {sm_results_obj.params[treatment]}")
            logger.info(f"  Std Error: {sm_results_obj.bse[treatment]}")
            logger.info(f"  P-value: {sm_results_obj.pvalues[treatment]}")
            results['statsmodels_estimate'] = sm_results_obj.params[treatment]
            results['statsmodels_results_object'] = sm_results_obj
            method_used = 'statsmodels'
            logger.info("Statsmodels estimation successful.")
        except Exception as sm_e:
            logger.error(f"Statsmodels IV estimation also failed: {sm_e}", exc_info=True)
            results['statsmodels_error'] = str(sm_e)
            method_used = 'statsmodels_failed' if method_used == "none" else "dowhy_failed_sm_failed"

    # --- Diagnostics --- 
    logger.info("\nRunning diagnostics...")
    diagnostics = run_iv_diagnostics(df, treatment, outcome, valid_instruments, clean_covariates, sm_results_obj, dw_results_obj)
    results['diagnostics'] = diagnostics

    # --- Refutation Step --- 
    final_estimate_value = results.get('dowhy_estimate') if method_used == 'dowhy' else results.get('statsmodels_estimate')
    
    # Only run permute refuter if estimate is valid AND came from DoWhy
    if method_used == 'dowhy' and dw_results_obj and final_estimate_value is not None: 
        logger.info("\nRunning refutation test (Placebo Treatment - Permute - requires DoWhy estimate object)...")
        try:
            # Pass the actual DoWhy estimate object
            refuter_result = model.refute_estimate(
                identified_estimand, 
                dw_results_obj, # Pass the original DoWhy result object
                method_name="placebo_treatment_refuter", 
                placebo_type="permute" # Necessary for IV according to docs/examples
            )
            logger.info("Refutation test completed.")
            logger.debug(f"Refuter Result:\n{refuter_result}")
            # Store relevant info from refuter_result (check its structure)
            refutation_results = {
                "refuter": "placebo_treatment_refuter",
                "new_effect": getattr(refuter_result, 'new_effect', 'N/A'),
                "p_value": getattr(refuter_result, 'refutation_result', {}).get('p_value', 'N/A') if hasattr(refuter_result, 'refutation_result') else 'N/A',
                # Passed if p-value > 0.05 (or not statistically significant)
                "passed": getattr(refuter_result, 'refutation_result', {}).get('is_statistically_significant', None) == False if hasattr(refuter_result, 'refutation_result') else None 
            }
        except Exception as refute_e:
            logger.error(f"Refutation test failed: {refute_e}", exc_info=True)
            refutation_results = {"error": f"Refutation failed: {refute_e}"}
            
    elif final_estimate_value is not None and method_used == 'statsmodels':
        logger.warning("Skipping placebo permutation refuter: Estimate was generated by statsmodels, not DoWhy's IV estimator.")
        refutation_results = {"status": "skipped_wrong_estimator_for_permute"}
        
    elif final_estimate_value is None:
        logger.warning("Skipping refutation test because estimation failed.")
        refutation_results = {"status": "skipped_due_to_failed_estimation"}
        
    else: # Model or estimand failed earlier, or unknown method_used
        logger.warning(f"Skipping refutation test due to earlier failure (method_used: {method_used}).")
        refutation_results = {"status": "skipped_due_to_model_failure_or_unknown"}
        
    results['refutation_results'] = refutation_results # Add to main results

    # --- Formatting Results --- 
    if final_estimate_value is None and method_used not in ['dowhy', 'statsmodels']:
        logger.error("ERROR: Both estimation methods failed.")
        # Ensure error key exists if not set earlier
        if 'error' not in results:
            results['error'] = "Both DoWhy and statsmodels IV estimation failed."

    logger.info("\n--- Formatting Final Results ---")
    formatted_results = format_iv_results(
        final_estimate_value, # Pass the numeric value
        results, # Pass the dict containing estimate objects and refutation results
        diagnostics,
        treatment,
        outcome,
        valid_instruments,
        method_used,
        llm=llm
    )

    logger.info("--- Instrumental Variable Estimation Complete ---\n")
    return formatted_results 