# Utility functions for Difference-in-Differences
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_post_indicator(df: pd.DataFrame, time_var: str, treatment_period_start: any) -> pd.Series:
    """Creates the post-treatment indicator variable.
    Checks if time_var is already a 0/1 indicator; otherwise, compares to treatment_period_start.
    """
    try:
        time_var_series = df[time_var]
        # Ensure numeric for checks and direct comparison
        if pd.api.types.is_bool_dtype(time_var_series):
            time_var_series = time_var_series.astype(int)
        
        # Check if it's already a binary 0/1 indicator
        if pd.api.types.is_numeric_dtype(time_var_series):
            unique_vals = set(time_var_series.dropna().unique())
            if unique_vals == {0, 1}:
                logger.info(f"Time variable '{time_var}' is already a binary 0/1 indicator. Using it directly as post indicator.")
                return time_var_series.astype(int)
            else:
                # Numeric, but not 0/1, so compare with treatment_period_start
                logger.info(f"Time variable '{time_var}' is numeric. Comparing with treatment_period_start: {treatment_period_start}")
                return (time_var_series >= treatment_period_start).astype(int)
        else:
             pass # Let it fall through to TypeError if not numeric here

        return (df[time_var] >= treatment_period_start).astype(int)

    except TypeError:
        # If direct comparison fails (e.g., comparing datetime with int/str, or non-numeric string with number),
        # attempt to convert both to datetime objects for comparison.
        logger.info(f"Direct comparison/numeric check failed for time_var '{time_var}'. Attempting datetime conversion.")
        try:
            time_series_dt = pd.to_datetime(df[time_var], errors='coerce')
            # Try to convert treatment_period_start to datetime if it's not already
            # This handles cases where treatment_period_start might be a date string
            try:
                treatment_start_dt = pd.to_datetime(treatment_period_start)
            except Exception as e_conv:
                logger.error(f"Could not convert treatment_period_start '{treatment_period_start}' to datetime: {e_conv}")
                raise TypeError(f"treatment_period_start '{treatment_period_start}' could not be converted to a comparable datetime format.")

            if time_series_dt.isna().all(): # if all values are NaT after conversion
                raise ValueError(f"Time variable '{time_var}' could not be converted to datetime (all values NaT).")
            if pd.isna(treatment_start_dt):
                raise ValueError(f"Treatment start period '{treatment_period_start}' converted to NaT.")
            
            logger.info(f"Comparing time_var '{time_var}' (as datetime) with treatment_start_dt '{treatment_start_dt}' (as datetime).")
            return (time_series_dt >= treatment_start_dt).astype(int)
        except Exception as e:
            logger.error(f"Failed to compare time variable '{time_var}' with treatment start '{treatment_period_start}' using datetime logic: {e}", exc_info=True)
            raise TypeError(f"Could not compare time variable '{time_var}' with treatment start '{treatment_period_start}'. Ensure they are comparable or convertible to datetime. Error: {e}")
    except Exception as ex:
        # Catch any other unexpected errors during the initial numeric processing
        logger.error(f"Unexpected error processing time_var '{time_var}' for post indicator: {ex}", exc_info=True)
        raise TypeError(f"Unexpected error processing time_var '{time_var}': {ex}") 