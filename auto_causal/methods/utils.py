"""
Utility functions for causal inference methods.

This module provides common utility functions used across
different causal inference methods.
"""

from typing import Dict, List, Set, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_binary_treatment(treatment_series: pd.Series) -> bool:
    """
    Check if treatment variable is binary.
    
    Args:
        treatment_series: Series containing treatment variable
        
    Returns:
        Boolean indicating if treatment is binary
    """
    unique_values = set(treatment_series.unique())
    # Remove NaN values if present
    unique_values = {x for x in unique_values if pd.notna(x)}
    
    # Check if there are exactly 2 unique values
    if len(unique_values) != 2:
        return False
    
    # Check if values are 0/1 or similar binary encoding
    sorted_vals = sorted(unique_values)
    
    # Check common binary encodings: 0/1, False/True, etc.
    binary_pairs = [
        (0, 1),
        (False, True),
        ("0", "1"),
        ("no", "yes"),
        ("false", "true")
    ]
    
    # Convert to strings for comparison if needed
    if not all(isinstance(v, (int, float, bool)) for v in sorted_vals):
        # Convert to lowercase strings for comparison
        str_vals = [str(v).lower() for v in sorted_vals]
        for pair in binary_pairs:
            str_pair = [str(v).lower() for v in pair]
            if str_vals == str_pair:
                return True
        return False
    
    # For numeric values, check if they're 0/1 or can be easily mapped to 0/1
    if sorted_vals == [0, 1]:
        return True
    
    # Check if there are only two values that could be easily mapped
    return len(unique_values) == 2


def calculate_standardized_differences(df: pd.DataFrame, treatment: str, covariates: List[str]) -> Dict[str, float]:
    """
    Calculate standardized differences between treated and control groups.
    
    Args:
        df: DataFrame containing the data
        treatment: Name of treatment variable
        covariates: List of covariate variable names
        
    Returns:
        Dictionary with standardized differences for each covariate
    """
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]
    
    std_diffs = {}
    
    for cov in covariates:
        # Skip if covariate has missing values
        if df[cov].isna().any():
            std_diffs[cov] = np.nan
            continue
            
        t_mean = treated[cov].mean()
        c_mean = control[cov].mean()
        
        t_var = treated[cov].var()
        c_var = control[cov].var()
        
        # Pooled standard deviation
        pooled_std = np.sqrt((t_var + c_var) / 2)
        
        # Avoid division by zero
        if pooled_std == 0:
            std_diffs[cov] = 0
        else:
            std_diffs[cov] = (t_mean - c_mean) / pooled_std
    
    return std_diffs


def check_overlap(df: pd.DataFrame, treatment: str, propensity_scores: np.ndarray, 
                 threshold: float = 0.5) -> Dict[str, Any]:
    """
    Check overlap in propensity scores between treated and control groups.
    
    Args:
        df: DataFrame containing the data
        treatment: Name of treatment variable
        propensity_scores: Array of propensity scores
        threshold: Threshold for sufficient overlap (proportion of range)
        
    Returns:
        Dictionary with overlap statistics
    """
    df_copy = df.copy()
    df_copy['propensity_score'] = propensity_scores
    
    treated = df_copy[df_copy[treatment] == 1]['propensity_score']
    control = df_copy[df_copy[treatment] == 0]['propensity_score']
    
    min_treated = treated.min()
    max_treated = treated.max()
    min_control = control.min()
    max_control = control.max()
    
    overall_min = min(min_treated, min_control)
    overall_max = max(max_treated, max_control)
    
    # Range of overlap
    overlap_min = max(min_treated, min_control)
    overlap_max = min(max_treated, max_control)
    
    # Check if there is any overlap
    if overlap_max < overlap_min:
        overlap_proportion = 0
        sufficient_overlap = False
    else:
        # Calculate proportion of overall range that has overlap
        overall_range = overall_max - overall_min
        if overall_range == 0:
            # All values are the same
            overlap_proportion = 1.0
            sufficient_overlap = True
        else:
            overlap_proportion = (overlap_max - overlap_min) / overall_range
            sufficient_overlap = overlap_proportion >= threshold
    
    return {
        "treated_range": (float(min_treated), float(max_treated)),
        "control_range": (float(min_control), float(max_control)),
        "overlap_range": (float(overlap_min), float(overlap_max)),
        "overlap_proportion": float(overlap_proportion),
        "sufficient_overlap": sufficient_overlap
    }


def plot_propensity_overlap(df: pd.DataFrame, treatment: str, propensity_scores: np.ndarray,
                           save_path: Optional[str] = None) -> None:
    """
    Plot overlap in propensity scores.
    
    Args:
        df: DataFrame containing the data
        treatment: Name of treatment variable
        propensity_scores: Array of propensity scores
        save_path: Optional path to save the plot
    """
    df_copy = df.copy()
    df_copy['propensity_score'] = propensity_scores
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    sns.histplot(df_copy.loc[df_copy[treatment] == 1, 'propensity_score'], 
                bins=20, alpha=0.5, label='Treated', color='blue', kde=True)
    sns.histplot(df_copy.loc[df_copy[treatment] == 0, 'propensity_score'], 
                bins=20, alpha=0.5, label='Control', color='red', kde=True)
    
    plt.title('Propensity Score Distributions')
    plt.xlabel('Propensity Score')
    plt.ylabel('Count')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_covariate_balance(standardized_diffs: Dict[str, float], threshold: float = 0.1,
                          save_path: Optional[str] = None) -> None:
    """
    Plot standardized differences for covariates before and after matching.
    
    Args:
        standardized_diffs: Dictionary with standardized differences
        threshold: Threshold for acceptable balance
        save_path: Optional path to save the plot
    """
    # Convert to DataFrame for plotting
    df = pd.DataFrame({
        'Covariate': list(standardized_diffs.keys()),
        'Standardized Difference': list(standardized_diffs.values())
    })
    
    # Sort by absolute standardized difference
    df['Absolute Difference'] = np.abs(df['Standardized Difference'])
    df = df.sort_values('Absolute Difference', ascending=False)
    
    plt.figure(figsize=(12, len(standardized_diffs) * 0.4 + 2))
    
    # Plot horizontal bars
    ax = sns.barplot(x='Standardized Difference', y='Covariate', data=df,
                    palette=['red' if abs(x) > threshold else 'green' for x in df['Standardized Difference']])
    
    # Add vertical lines for thresholds
    plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=-threshold, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
    
    plt.title('Covariate Balance: Standardized Differences')
    plt.xlabel('Standardized Difference')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def check_temporal_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check if dataset has temporal structure.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with temporal structure information
    """
    # Check for date/time columns
    date_cols = []
    
    for col in df.columns:
        # Check if column has date in name
        if any(date_term in col.lower() for date_term in ['date', 'time', 'year', 'month', 'day', 'period']):
            date_cols.append(col)
        
        # Check if column can be converted to datetime
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='raise')
                date_cols.append(col)
            except:
                pass
    
    # Check for panel structure - look for ID columns
    id_cols = []
    
    for col in df.columns:
        # Check if column has ID in name
        if any(id_term in col.lower() for id_term in ['id', 'identifier', 'key', 'code']):
            unique_count = df[col].nunique()
            # If column has multiple values but fewer than 10% of rows, likely an ID
            if 1 < unique_count < len(df) * 0.1:
                id_cols.append(col)
    
    # Check if there are multiple observations per unit
    is_panel = False
    panel_units = None
    
    if id_cols and date_cols:
        # For each ID column, check if there are multiple time periods
        for id_col in id_cols:
            obs_per_id = df.groupby(id_col).size()
            if (obs_per_id > 1).any():
                is_panel = True
                panel_units = id_col
                break
    
    return {
        "has_temporal_structure": len(date_cols) > 0,
        "temporal_columns": date_cols,
        "potential_id_columns": id_cols,
        "is_panel_data": is_panel,
        "panel_units": panel_units
    }


def check_for_discontinuities(df: pd.DataFrame, outcome: str, 
                             threshold_zscore: float = 3.0) -> Dict[str, Any]:
    """
    Check for potential discontinuities in continuous variables.
    
    Args:
        df: DataFrame to check
        outcome: Name of outcome variable
        threshold_zscore: Z-score threshold for detecting discontinuities
        
    Returns:
        Dictionary with discontinuity information
    """
    potential_running_vars = []
    
    # Check only numeric columns that aren't the outcome
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != outcome]
    
    for col in numeric_cols:
        # Skip if too many unique values (unlikely to be a running variable)
        if df[col].nunique() > 100:
            continue
            
        # Sort values and calculate differences
        sorted_vals = np.sort(df[col].unique())
        if len(sorted_vals) <= 1:
            continue
            
        diffs = np.diff(sorted_vals)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        # Skip if all differences are the same
        if std_diff == 0:
            continue
            
        # Calculate z-scores of differences
        zscores = (diffs - mean_diff) / std_diff
        
        # Check if any z-score exceeds threshold
        if np.any(np.abs(zscores) > threshold_zscore):
            # Potential discontinuity found
            max_idx = np.argmax(np.abs(zscores))
            threshold = (sorted_vals[max_idx] + sorted_vals[max_idx + 1]) / 2
            
            # Check if outcome means differ across threshold
            below_mean = df[df[col] < threshold][outcome].mean()
            above_mean = df[df[col] >= threshold][outcome].mean()
            
            # Only include if outcome means differ substantially
            if abs(above_mean - below_mean) > 0.1 * df[outcome].std():
                potential_running_vars.append({
                    "variable": col,
                    "threshold": float(threshold),
                    "z_score": float(zscores[max_idx]),
                    "outcome_diff": float(above_mean - below_mean)
                })
    
    return {
        "has_discontinuities": len(potential_running_vars) > 0,
        "potential_running_variables": potential_running_vars
    }


def find_potential_instruments(df: pd.DataFrame, treatment: str, outcome: str,
                              correlation_threshold: float = 0.3) -> Dict[str, Any]:
    """
    Find potential instrumental variables.
    
    Args:
        df: DataFrame to check
        treatment: Name of treatment variable
        outcome: Name of outcome variable
        correlation_threshold: Threshold for correlation with treatment
        
    Returns:
        Dictionary with potential instruments information
    """
    # Get numeric columns that aren't treatment or outcome
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    potential_ivs = [col for col in numeric_cols if col != treatment and col != outcome]
    
    iv_results = []
    
    for col in potential_ivs:
        # Skip if column has too many missing values
        if df[col].isna().mean() > 0.1:
            continue
        
        # Check correlation with treatment
        corr_treatment = df[[col, treatment]].corr().iloc[0, 1]
        
        # Check correlation with outcome
        corr_outcome = df[[col, outcome]].corr().iloc[0, 1]
        
        # Potential IV should be correlated with treatment but not directly with outcome
        if abs(corr_treatment) > correlation_threshold and abs(corr_outcome) < correlation_threshold/2:
            iv_results.append({
                "variable": col,
                "correlation_with_treatment": float(corr_treatment),
                "correlation_with_outcome": float(corr_outcome),
                "strength": "Strong" if abs(corr_treatment) > 0.5 else "Moderate"
            })
    
    return {
        "has_potential_instruments": len(iv_results) > 0,
        "potential_instruments": iv_results
    }


def test_parallel_trends(df: pd.DataFrame, treatment: str, outcome: str, 
                        time_var: str, unit_var: str) -> Dict[str, Any]:
    """
    Test for parallel trends assumption in difference-in-differences.
    
    Args:
        df: DataFrame to check
        treatment: Name of treatment variable
        outcome: Name of outcome variable
        time_var: Name of time variable
        unit_var: Name of unit variable
        
    Returns:
        Dictionary with parallel trends test results
    """
    # Ensure time_var is properly formatted
    df = df.copy()
    
    if df[time_var].dtype != 'int64':
        # Try to convert to datetime and then to period
        try:
            df[time_var] = pd.to_datetime(df[time_var])
            # Get unique periods and map to integers
            periods = df[time_var].dt.to_period('M').unique()
            period_dict = {p: i for i, p in enumerate(sorted(periods))}
            df['time_period'] = df[time_var].dt.to_period('M').map(period_dict)
            time_var = 'time_period'
        except:
            # If conversion fails, try to map unique values to integers
            unique_times = df[time_var].unique()
            time_dict = {t: i for i, t in enumerate(sorted(unique_times))}
            df['time_period'] = df[time_var].map(time_dict)
            time_var = 'time_period'
    
    # Identify treatment and control groups
    # Treatment indicator should be 0 or 1 for each unit (not time-varying)
    unit_treatment = df.groupby(unit_var)[treatment].max()
    treatment_units = unit_treatment[unit_treatment == 1].index
    control_units = unit_treatment[unit_treatment == 0].index
    
    # Find time of treatment implementation
    if len(treatment_units) > 0:
        treatment_time = df[df[unit_var].isin(treatment_units) & (df[treatment] == 1)][time_var].min()
    else:
        # No treated units found
        return {
            "parallel_trends": False,
            "reason": "No treated units found",
            "pre_trend_correlation": None,
            "pre_trend_p_value": None
        }
    
    # Select pre-treatment periods
    pre_treatment = df[df[time_var] < treatment_time]
    
    # Calculate average outcome by time and group
    treated_means = pre_treatment[pre_treatment[unit_var].isin(treatment_units)].groupby(time_var)[outcome].mean()
    control_means = pre_treatment[pre_treatment[unit_var].isin(control_units)].groupby(time_var)[outcome].mean()
    
    # Need enough pre-treatment periods to test
    if len(treated_means) < 3:
        return {
            "parallel_trends": None,
            "reason": "Insufficient pre-treatment periods",
            "pre_trend_correlation": None,
            "pre_trend_p_value": None
        }
    
    # Align indices and calculate trends
    common_periods = sorted(set(treated_means.index).intersection(set(control_means.index)))
    
    if len(common_periods) < 3:
        return {
            "parallel_trends": None,
            "reason": "Insufficient common pre-treatment periods",
            "pre_trend_correlation": None,
            "pre_trend_p_value": None
        }
    
    treated_trends = np.diff(treated_means[common_periods])
    control_trends = np.diff(control_means[common_periods])
    
    # Calculate correlation between trends
    correlation, p_value = stats.pearsonr(treated_trends, control_trends)
    
    # Test if trends are parallel (high correlation, not significantly different)
    parallel_trends = correlation > 0.7 and p_value < 0.05
    
    return {
        "parallel_trends": parallel_trends,
        "reason": "Trends are parallel" if parallel_trends else "Trends are not parallel",
        "pre_trend_correlation": float(correlation),
        "pre_trend_p_value": float(p_value)
    }


def preprocess_data(df: pd.DataFrame, treatment_var: str, outcome_var: str, 
                    covariates: List[str], verbose: bool = True) -> pd.DataFrame:
    """
    Preprocess the dataset to handle missing values and encode categorical variables.
    
    Args:
        df (pd.DataFrame): The dataset
        treatment_var (str): The treatment variable name
        outcome_var (str): The outcome variable name
        covariates (list): List of covariate variable names
        verbose (bool): Whether to print verbose output
        
    Returns:
        Tuple[pd.DataFrame, str, str, List[str], Dict[str, Any]]: 
            Preprocessed dataset, updated treatment var name, 
            updated outcome var name, updated covariates list, 
            and column mappings.
    """
    df_processed = df.copy()
    column_mappings: Dict[str, Any] = {}

    # Store original dtypes for mapping
    original_dtypes = {col: str(df_processed[col].dtype) for col in df_processed.columns}

    # Report missing values
    all_vars = [treatment_var, outcome_var] + covariates
    missing_data = df_processed[all_vars].isnull().sum()
    total_missing = missing_data.sum()
    
    if total_missing > 0:
        if verbose:
            logger.info(f"Dataset contains {total_missing} missing values:")
        for col in missing_data[missing_data > 0].index:
            percent = (missing_data[col] / len(df_processed)) * 100
            if verbose:
                logger.info(f"  - {col}: {missing_data[col]} missing values ({percent:.2f}%)")
    else:
        if verbose:
            logger.info("No missing values found in relevant columns.")
        # return df_processed # No preprocessing needed if no missing values

    # Handle missing values in treatment variable
    if df_processed[treatment_var].isnull().sum() > 0:
        if verbose:
            logger.info(f"Filling missing values in treatment variable '{treatment_var}' with mode")
        # For treatment, use mode (most common value)
        mode_val = df_processed[treatment_var].mode()[0] if not df_processed[treatment_var].mode().empty else 0
        df_processed[treatment_var] = df_processed[treatment_var].fillna(mode_val)
    
    # Handle missing values in outcome variable
    if df_processed[outcome_var].isnull().sum() > 0:
        if verbose:
            logger.info(f"Filling missing values in outcome variable '{outcome_var}' with mean")
        # For outcome, use mean
        mean_val = df_processed[outcome_var].mean()
        df_processed[outcome_var] = df_processed[outcome_var].fillna(mean_val)
    
    # Handle missing values in covariates
    for col in covariates:
        if df_processed[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                # For numeric covariates, use mean
                if verbose:
                    logger.info(f"Filling missing values in numeric covariate '{col}' with mean")
                mean_val = df_processed[col].mean()
                df_processed[col] = df_processed[col].fillna(mean_val)
            elif pd.api.types.is_categorical_dtype(df_processed[col]) or df_processed[col].dtype == 'object':
                # For categorical covariates, use mode
                mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else "Missing"
                if verbose:
                    logger.info(f"Filling missing values in categorical covariate '{col}' with mode ('{mode_val}')")
                df_processed[col] = df_processed[col].fillna(mode_val)
            else:
                # For other types, create a "Missing" category
                if verbose:
                    logger.info(f"Filling missing values in covariate '{col}' of type {df_processed[col].dtype} with 'Missing' category")
                # Ensure the column is of object type before filling with string
                if df_processed[col].dtype != 'object':
                    try:
                        df_processed[col] = df_processed[col].astype(object)
                    except Exception as e:
                        logger.warning(f"Could not convert column {col} to object type to fill NAs: {e}. Skipping fill.")
                        continue
                df_processed[col] = df_processed[col].fillna("Missing")
    
    # --- Categorical Encoding ---
    updated_treatment_var = treatment_var
    updated_outcome_var = outcome_var
    
    # Helper function for label encoding binary categoricals
    def label_encode_binary(series: pd.Series, var_name: str) -> Tuple[pd.Series, Dict[int, Any]]:
        uniques = series.dropna().unique()
        mapping = {}
        if len(uniques) == 2:
            # Try to map to 0 and 1 consistently, e.g., sort and assign
            # Or if boolean, map True to 1, False to 0
            if series.dtype == 'bool':
                mapping = {0: False, 1: True}
                return series.astype(int), mapping
            
            # For non-boolean, sort to ensure consistent mapping
            # However, direct replacement is safer to control which becomes 0 and 1
            # For simplicity here, we'll make a simple map.
            # A more robust approach might involve explicit mapping rules or user input.
            sorted_uniques = sorted(uniques, key=lambda x: str(x)) # sort to make it deterministic
            map_dict = {sorted_uniques[0]: 0, sorted_uniques[1]: 1}
            mapping = {v: k for k, v in map_dict.items()} # Inverse map for column_mappings
            if verbose:
                logger.info(f"Label encoding binary variable '{var_name}': {map_dict}")
            return series.map(map_dict), mapping
        elif len(uniques) == 1: # Single unique value, treat as constant (encode as 0)
             if verbose:
                logger.info(f"Binary variable '{var_name}' has only one unique value '{uniques[0]}'. Encoding as 0.")
             map_dict = {uniques[0]:0}
             mapping = {0: uniques[0]}
             return series.map(map_dict), mapping
        return series, mapping # No change if not binary

    # Encode Treatment Variable
    if df_processed[treatment_var].dtype == 'object' or df_processed[treatment_var].dtype == 'category' or df_processed[treatment_var].dtype == 'bool':
        original_series = df_processed[treatment_var].copy()
        df_processed[treatment_var], value_map = label_encode_binary(df_processed[treatment_var], treatment_var)
        if value_map: # If encoding happened
            column_mappings[treatment_var] = {
                'original_dtype': original_dtypes[treatment_var],
                'transformed_as': 'label_encoded_binary',
                'new_column_name': treatment_var, # Name doesn't change
                'value_map': value_map 
            }
            if verbose:
                 logger.info(f"Encoded treatment variable '{treatment_var}' to numeric.")

    # Encode Outcome Variable
    if df_processed[outcome_var].dtype == 'object' or df_processed[outcome_var].dtype == 'category' or df_processed[outcome_var].dtype == 'bool':
        original_series = df_processed[outcome_var].copy()
        df_processed[outcome_var], value_map = label_encode_binary(df_processed[outcome_var], outcome_var)
        if value_map: # If encoding happened
            column_mappings[outcome_var] = {
                'original_dtype': original_dtypes[outcome_var],
                'transformed_as': 'label_encoded_binary',
                'new_column_name': outcome_var, # Name doesn't change
                'value_map': value_map
            }
            if verbose:
                 logger.info(f"Encoded outcome variable '{outcome_var}' to numeric.")

    # Encode Covariates (One-Hot Encoding for non-numeric)
    updated_covariates = []
    categorical_covariates_to_encode = []
    for cov in covariates:
        if cov not in df_processed.columns: # If a covariate was dropped or is an instrument etc.
            if verbose:
                logger.warning(f"Covariate '{cov}' not found in DataFrame columns after initial processing. Skipping encoding for it.")
            continue

        if df_processed[cov].dtype == 'object' or df_processed[cov].dtype == 'category' or pd.api.types.is_bool_dtype(df_processed[cov]):
            # Check if it's binary - if so, can also label encode
            # However, for consistency with get_dummies and to handle multi-category,
            # we'll let get_dummies handle it, or apply label encoding for binary covariates too.
            # For simplicity, let's stick to one-hot for all categorical covariates.
            if len(df_processed[cov].dropna().unique()) > 1 : # Only encode if more than 1 unique value
                 categorical_covariates_to_encode.append(cov)
            else: # If only one unique value or all NaNs (already handled), it's constant-like
                 if verbose:
                    logger.info(f"Categorical covariate '{cov}' has <= 1 unique value after NA handling. Treating as constant-like, not one-hot encoding.")
                 updated_covariates.append(cov) # Keep as is, will likely be numeric 0 or some constant
        else: # Already numeric
            updated_covariates.append(cov)

    if categorical_covariates_to_encode:
        if verbose:
            logger.info(f"One-hot encoding categorical covariates: {categorical_covariates_to_encode} using pd.get_dummies (drop_first=True)")
        
        # Store original columns before get_dummies to identify new ones
        original_df_columns = set(df_processed.columns)
        
        df_processed = pd.get_dummies(df_processed, columns=categorical_covariates_to_encode, 
                                      prefix_sep='_', drop_first=True, dummy_na=False) # dummy_na=False since we handled NAs
        
        # Identify new columns created by get_dummies
        new_dummy_columns = list(set(df_processed.columns) - original_df_columns)
        updated_covariates.extend(new_dummy_columns)
        
        for original_cov_name in categorical_covariates_to_encode:
            # Find which dummy columns correspond to this original covariate
            related_dummies = [col for col in new_dummy_columns if col.startswith(original_cov_name + '_')]
            column_mappings[original_cov_name] = {
                'original_dtype': original_dtypes[original_cov_name],
                'transformed_as': 'one_hot_encoded',
                'encoded_columns': related_dummies,
                # 'dropped_category': can be inferred if needed, but not explicitly stored for simplicity here
            }
            if verbose:
                logger.info(f"  Original covariate '{original_cov_name}' resulted in dummy variables: {related_dummies}")
    
    if verbose:
        logger.info("Preprocessing complete.")
        if column_mappings:
            logger.info(f"Column mappings generated: {column_mappings}")
        else:
            logger.info("No column encodings were applied.")

    return df_processed, updated_treatment_var, updated_outcome_var, list(dict.fromkeys(updated_covariates)), column_mappings


def check_collinearity(df: pd.DataFrame, covariates: List[str]) -> Optional[List[str]]:
    # Implementation of check_collinearity function
    # This function should return a list of collinear variables or None
    pass 