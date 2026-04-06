import logging
from typing import Dict, List, Any, Optional

from cais.config import get_llm_client
from cais.iv_llm.src.agents.hypothesizer import Hypothesizer
from cais.iv_llm.src.agents.confounder_miner import ConfounderMiner
from cais.iv_llm.src.critics.exclusion_critic import ExclusionCritic
from cais.iv_llm.src.critics.independence_critic import IndependenceCritic
from cais.iv_llm.src.llm.client import LLMClient


logger = logging.getLogger(__name__)

class IVDiscovery:
    def __init__(self):
        llm = LLMClient()
        self.hypothesizer = Hypothesizer(llm, k=5)
        self.confounder_miner = ConfounderMiner(llm, j=5)
        self.exclusion_critic = ExclusionCritic(llm)
        self.independence_critic = IndependenceCritic(llm)
    
    def discover_instruments(self, treatment: str, outcome: str, context: str = "", confounders: Optional[List[str]] = None) -> Dict[str, Any]:
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
        logger.info(f"Discovering instruments for treatment: {treatment}, outcome: {outcome}")
        
        # Step 1: Hypothesize IVs
        proposed_ivs = self.hypothesizer.propose_ivs(treatment, outcome, context=context)
        logger.info(f"Proposed IVs: {proposed_ivs}")
        
        if not proposed_ivs:
            return {
                'proposed_ivs': [],
                'valid_ivs': [],
                'validation_results': [],
                'confounders': confounders or []
            }
        
        # Step 2: Identify confounders if not provided
        if confounders is None:
            confounders = self.confounder_miner.identify_confounders(treatment, outcome, context=context)
        logger.info(f"Identified confounders: {confounders}")
        
        # Step 3: Validate IVs with critics
        validation_results = []
        valid_ivs = []
        
        # Run exclusion and independence critics
        exclusion_results = {}
        independence_results = {}
        
        # First pass: exclusion critic for all IVs
        for iv in proposed_ivs:
            exclusion_results[iv] = self.exclusion_critic.validate_exclusion(iv, treatment, outcome, confounders)
        
        # Second pass: independence critic for all IVs
        for iv in proposed_ivs:
            independence_results[iv] = self.independence_critic.validate_independence(iv, treatment, outcome, confounders)
        
        # Combine results
        for iv in proposed_ivs:
            exclusion_valid = exclusion_results[iv]
            independence_valid = independence_results[iv]
            
            validation_results.append({
                'iv': iv,
                'exclusion_valid': exclusion_valid,
                'independence_valid': independence_valid,
                'overall_valid': exclusion_valid and independence_valid
            })
            
            if exclusion_valid and independence_valid:
                valid_ivs.append(iv)
        
        logger.info(f"Valid IVs found: {valid_ivs}")
        
        return {
            'proposed_ivs': proposed_ivs,
            'valid_ivs': valid_ivs,
            'validation_results': validation_results,
            'confounders': confounders
        }

def discover_instruments(treatment: str, outcome: str, context: str = "", confounders: Optional[List[str]] = None, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to discover instruments using default IVDiscovery instance.
    """
    # `config_path` is currently unused; keep it for backwards compatibility.
    discovery = IVDiscovery()
    return discovery.discover_instruments(treatment, outcome, context, confounders)