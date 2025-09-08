"""
Abstract base class for all causal inference methods.

This module defines the interface that all causal inference methods
must implement, ensuring consistent behavior across different methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
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
    def validate_assumptions(self, df: pd.DataFrame, treatment: str, 
                           outcome: str, covariates: List[str]) -> Dict[str, Any]:
        """Validate method assumptions against the dataset.
        
        Args:
            df: DataFrame containing the dataset
            treatment: Name of the treatment variable column
            outcome: Name of the outcome variable column
            covariates: List of covariate column names
            
        Returns:
            Dict containing validation results with keys:
                - assumptions_valid (bool): Whether all assumptions are met
                - failed_assumptions (List[str]): List of failed assumptions
                - warnings (List[str]): List of warnings
                - suggestions (List[str]): Suggestions for addressing issues
        """
        pass
    
    @abstractmethod
    def estimate_effect(self, df: pd.DataFrame, treatment: str,
                      outcome: str, covariates: List[str]) -> Dict[str, Any]:
        """Estimate causal effect using this method.
        
        Args:
            df: DataFrame containing the dataset
            treatment: Name of the treatment variable column
            outcome: Name of the outcome variable column
            covariates: List of covariate column names
            
        Returns:
            Dict containing estimation results with keys:
                - effect_estimate (float): Estimated causal effect
                - confidence_interval (tuple): Confidence interval (lower, upper)
                - p_value (float): P-value of the estimate
                - additional_metrics (Dict): Any method-specific metrics
        """
        pass
    
    @abstractmethod
    def generate_code(self, dataset_path: str, treatment: str,
                    outcome: str, covariates: List[str]) -> str:
        """Generate executable code for this causal method.
        
        Args:
            dataset_path: Path to the dataset file
            treatment: Name of the treatment variable column
            outcome: Name of the outcome variable column
            covariates: List of covariate column names
            
        Returns:
            String containing executable Python code implementing this method
        """
        pass
    
    @abstractmethod
    def explain(self) -> str:
        """Explain this causal method, its assumptions, and when to use it.
        
        Returns:
            String with detailed explanation of the method
        """
        pass 