"""
Basic descriptive statistics for Difference in Means.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def run_dim_diagnostics(df: pd.DataFrame, treatment: str, outcome: str) -> Dict[str, Any]:
    """
    Calculates basic descriptive statistics for treatment and control groups.
    
    Args:
        df: Input DataFrame (should already be filtered for NaNs in treatment/outcome).
        treatment: Name of the binary treatment variable column.
        outcome: Name of the outcome variable column.
        
    Returns:
        Dictionary containing group means, standard deviations, and counts.
    """
    details = {}
    try:
        grouped = df.groupby(treatment)[outcome]
        stats = grouped.agg(['mean', 'std', 'count'])
        
        # Ensure both groups (0 and 1) are present if possible
        control_stats = stats.loc[0].to_dict() if 0 in stats.index else {'mean': np.nan, 'std': np.nan, 'count': 0}
        treated_stats = stats.loc[1].to_dict() if 1 in stats.index else {'mean': np.nan, 'std': np.nan, 'count': 0}
        
        details['control_group_stats'] = control_stats
        details['treated_group_stats'] = treated_stats
        
        if control_stats['count'] == 0 or treated_stats['count'] == 0:
             logger.warning("One or both treatment groups have zero observations.")
             return {"status": "Warning - Empty Group(s)", "details": details}
        
        # Simple check for variance difference (Levene's test could be added)
        control_std = control_stats.get('std', 0)
        treated_std = treated_stats.get('std', 0)
        if control_std > 0 and treated_std > 0:
            ratio = (control_std**2) / (treated_std**2)
            details['variance_ratio_control_div_treated'] = ratio
            if ratio > 4 or ratio < 0.25: # Rule of thumb
                details['variance_homogeneity_status'] = "Potentially Unequal (ratio > 4 or < 0.25)"
            else:
                 details['variance_homogeneity_status'] = "Likely Similar"
        else:
            details['variance_homogeneity_status'] = "Could not calculate (zero variance in a group)"
            
        return {"status": "Success", "details": details}
        
    except KeyError as ke:
         logger.error(f"KeyError during diagnostics: {ke}. Treatment levels might not be 0/1.")
         return {"status": "Failed", "error": f"Treatment levels might not be 0/1: {ke}", "details": details}
    except Exception as e:
        logger.error(f"Error running Difference in Means diagnostics: {e}")
        return {"status": "Failed", "error": str(e), "details": details}
