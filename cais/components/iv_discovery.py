"""
IV Discovery component using the IV-LLM pipeline.

This module provides IV discovery functionality using the Hypothesizer,
ConfounderMiner, ExclusionCritic, and IndependenceCritic agents.
"""

import logging
from typing import Dict, List, Any, Optional

from langchain_core.language_models import BaseChatModel

from cais.iv_llm.src.agents.hypothesizer import Hypothesizer
from cais.iv_llm.src.agents.confounder_miner import ConfounderMiner
from cais.iv_llm.src.critics.exclusion_critic import ExclusionCritic
from cais.iv_llm.src.critics.independence_critic import IndependenceCritic


logger = logging.getLogger(__name__)


class LangChainLLMAdapter:
    """
    Adapter that wraps a LangChain BaseChatModel to provide the .generate(prompt)
    interface expected by iv_llm components.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM given a prompt string."""
        response = self.llm.invoke(prompt)
        # Handle different response types
        if hasattr(response, 'content'):
            return response.content
        return str(response)


class IVDiscovery:
    def __init__(self, llm: Optional[BaseChatModel] = None, k_ivs: int = 5, j_confounders: int = 5):
        """
        Initialize IV Discovery with the workspace's standard LLM client.

        Args:
            llm: Optional LangChain BaseChatModel. If not provided, uses get_llm_client().
            k_ivs: Number of instrumental variables to propose.
            j_confounders: Number of confounders to identify.
        """
        # Use workspace's standard LLM client if not provided
        if llm is None:
            from cais.config import get_llm_client
            llm = get_llm_client()

        # Wrap the LangChain LLM with adapter for iv_llm compatibility
        self.llm_client = LangChainLLMAdapter(llm)

        self.hypothesizer = Hypothesizer(self.llm_client, k=k_ivs)
        self.confounder_miner = ConfounderMiner(self.llm_client, j=j_confounders)
        self.exclusion_critic = ExclusionCritic(self.llm_client)
        self.independence_critic = IndependenceCritic(self.llm_client)

    def discover_instruments(
        self,
        treatment: str,
        outcome: str,
        context: str = "",
        confounders: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Discover valid instrumental variables for the given treatment and outcome.

        Args:
            treatment: Name of the treatment variable
            outcome: Name of the outcome variable
            context: Additional context about the dataset/query
            confounders: List of known confounders (optional)

        Returns:
            Dict containing proposed IVs, valid IVs, and validation results
        """
        logger.info(
            "Discovering instruments for treatment: %s, outcome: %s",
            treatment,
            outcome,
        )

        # Hypothesize IVs
        proposed_ivs = self.hypothesizer.propose_ivs(treatment, outcome, context=context)
        logger.info("Proposed IVs: %s", proposed_ivs)

        if not proposed_ivs:
            return {
                "proposed_ivs": [],
                "valid_ivs": [],
                "validation_results": [],
                "confounders": confounders or [],
            }

        # Identify confounders if not provided
        if confounders is None:
            confounders = self.confounder_miner.identify_confounders(
                treatment, outcome, context=context
            )
        logger.info("Identified confounders: %s", confounders)

        # Validate IVs with critics
        validation_results = []
        valid_ivs = []

        # Run exclusion and independence critics
        exclusion_results = {}
        independence_results = {}

        # First pass: exclusion critic for all IVs
        for iv in proposed_ivs:
            exclusion_results[iv] = self.exclusion_critic.validate_exclusion(
                iv, treatment, outcome, confounders
            )

        # Second pass: independence critic for all IVs
        for iv in proposed_ivs:
            independence_results[iv] = self.independence_critic.validate_independence(
                iv, treatment, outcome, confounders
            )

        # Combine results
        for iv in proposed_ivs:
            exclusion_valid = exclusion_results[iv]
            independence_valid = independence_results[iv]

            validation_results.append(
                {
                    "iv": iv,
                    "exclusion_valid": exclusion_valid,
                    "independence_valid": independence_valid,
                    "overall_valid": exclusion_valid and independence_valid,
                }
            )

            if exclusion_valid and independence_valid:
                valid_ivs.append(iv)

        logger.info("Valid IVs found: %s", valid_ivs)

        return {
            "proposed_ivs": proposed_ivs,
            "valid_ivs": valid_ivs,
            "validation_results": validation_results,
            "confounders": confounders,
        }


def discover_instruments(
    treatment: str,
    outcome: str,
    context: str = "",
    confounders: Optional[List[str]] = None,
    llm: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """
    Convenience function to discover instruments using IVDiscovery.

    Args:
        treatment: Name of the treatment variable
        outcome: Name of the outcome variable
        context: Additional context about the dataset/query
        confounders: List of known confounders (optional)
        llm: Optional LangChain BaseChatModel. If not provided, uses get_llm_client().

    Returns:
        Dict containing proposed IVs, valid IVs, and validation results
    """
    discovery = IVDiscovery(llm=llm)
    return discovery.discover_instruments(treatment, outcome, context, confounders)
