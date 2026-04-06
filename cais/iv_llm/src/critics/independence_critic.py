import json
import logging
from typing import List

from langchain_core.language_models import BaseChatModel

from ..prompts.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

class IndependenceCritic:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self.prompt_loader = PromptLoader()
    
    def validate_independence(self, iv: str, treatment: str, outcome: str, confounders: List[str]) -> bool:
        # Independence: check IV against each confounder separately
        responses = {}
        
        for confounder in confounders:
            prompt = self.prompt_loader.format_independence_prompt(iv, treatment, outcome, confounder)
            response = self.llm.invoke(prompt)
            responses[confounder] = response
            
            if not self._parse_validity(response):
                # Log detailed output
                logger.info(json.dumps({
                    'name': f'independence_critic_{iv}',
                    'inputs': {'iv': iv, 'treatment': treatment, 'outcome': outcome, 'confounders': confounders},
                    'outputs': {'valid': False, 'failed_on': confounder},
                    'raw_response': responses,
                }, default=str))
                return False
        
        # Log successful validation
        logger.info(json.dumps({
            'name': f'independence_critic_{iv}',
            'inputs': {'iv': iv, 'treatment': treatment, 'outcome': outcome, 'confounders': confounders},
            'outputs': {'valid': True},
            'raw_response': responses,
        }, default=str))
        
        return True
    
    def _parse_validity(self, response: str) -> bool:
        import re
        match = re.search(r'<Answer>(Valid|Invalid)</Answer>', response)
        return match.group(1) == 'Valid' if match else False