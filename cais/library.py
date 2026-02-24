'''
Estimator library for easy agent access and user modification for custom estimators. Essentially a glorified dictionary with checks for new estimators. 

TODO: Possibly convert each CausalMethod to their own Pydantic models with accessible assumption attributes instead. Could be easier to incorporate into future LangChain/graph workflows.
'''

import logging
from typing import List, Set
from cais.methods.causal_method import CausalMethod
from cais.methods.linear_regression.estimator import LinearRegression
from cais.methods.regression_discontinuity.estimator import RDDRegression
from cais.methods.difference_in_differences.estimator import DiDRegression
logger = logging.getLogger(__name__)

class Estimators:
    def __init__(self):

        self.estimators = None
        self.refresh_()

    def __getitem__(self, key):
        return self.estimators[key]

    def __iter__(self):
        return iter(self.estimators)

    def __len__(self):
        return len(self.estimators)

    def refresh_(self):
        '''
        Initialize or refresh the library of default estimators.
        '''
        logger.info("Refreshed estimator library.")
        default_estimators = {
            # Add variables, i.e.
            LinearRegression.name : LinearRegression(),
            RDDRegression.name : RDDRegression(),
            DiDRegression.name : DiDRegression()
        }

        self.estimators = default_estimators

    def describe(self):
        '''
        Returns:
            Dict[str, str] of the available causal estimators and their description. Description should include any assumptions, see cais/methods/causal_method.py
        '''
        return {k: v.describe() for k,v in self.estimators.items()}
    
    def add_estimator(self, name: str, estimator: CausalMethod) -> None:
        '''
        Helper function to add custom causal estimators. Estimators must inhereit from CausalMethod ABC.

        Args:
            name: string for the name of the causal estimator
            estimator: object of CausalMethod subclass with methods for estimating causal effects

        Returns:
            None
        '''
        assert isinstance(estimator, CausalMethod)

        if name in self.estimators:
            logger.warning(f"Cannot add {name} estimator as it is already found in the estimator library. Current keys: {list(self.estimators.keys())}")
            return

        self.estimators[name] = estimator

    def remove_estimator(self, name: str) -> None:
        '''
        Helper function to remove causal estimators from the default list. Useful to narrow down candidate estimators.

        Args:
            name: string for the name of the causal estimator
        
        Returns:
            None
        '''

        if name not in self.estimators:
            logger.warning(f"Cannot remove {name} estimator as it is not found in the estimator library.")
            return

        del self.estimators[name]

        if len(self.estimators) == 0:
            logger.warning("No available estimators in the library.")

    def names(self) -> Set[str]:
        return set(self.estimators.keys())