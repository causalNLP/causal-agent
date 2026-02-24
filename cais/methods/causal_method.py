"""
Abstract base class for all causal inference methods.

This module defines the interface that all causal inference methods must implement, ensuring consistent behavior across different methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from models import Variables
import pandas as pd


class CausalMethod(ABC):
    """Base class for all causal inference methods.
    
    This abstract class defines the required methods that all causal
    inference implementations must provide. It ensures a consistent
    interface across different methods like propensity score matching,
    instrumental variables, etc.
    
    Each implementation should handle the specifics of the causal
    inference method while conforming to this interface.
    """

    @abstractmethod
    def describe(self) -> str:
        """Explain this causal method, its assumptions, and the required variables.
        
        Returns:
            String with detailed explanation of the method
        """
        pass 
    
    @abstractmethod
    def validate_assumptions(self, df: pd.DataFrame, variables: Variables) -> Dict[str, Any]:
        """Validate method assumptions against the dataset. Checks whether any key variables for the method are None/missing
        
        Args:
            df: DataFrame containing the dataset
            variables: Variables pydantic model containing the extract variables
            
        Returns:
            Dict containing validation results with keys:
                - assumptions_valid (bool): Whether all assumptions are met
                - failed_assumptions (List[str]): List of failed assumptions
                - warnings (List[str]): List of warnings
                - suggestions (List[str]): Suggestions for addressing issues
        """
        pass
    
    @staticmethod
    @abstractmethod
    def estimate_effect(df: pd.DataFrame, variables: Variables) -> Dict[str, Any]:
        """Estimate causal effect using this method.
        
        Args:
            df: DataFrame containing the dataset
            variables: Pydantic model with Variables and their types
            
        Returns:
            Dict containing estimation results with keys:
                - effect_estimate (float): Estimated causal effect
                - confidence_interval (tuple): Confidence interval (lower, upper)
                - p_value (float): P-value of the estimate
                - additional_metrics (Dict): Any method-specific metrics
        """
        pass

    def __call__(self, df, variables):
        return CausalMethod.estimate_effect(df, variables)
    

'''
Can be made into tools for future LangChain workflows by

class OLS(CausalMethod):
    pass

class input_schema(BaseModel):
    variables: Field(variables, etc)

@tool(args_schema=input_schema)
def linear_regression(init_args, callable_args):
    return OLS(callable_args)
'''