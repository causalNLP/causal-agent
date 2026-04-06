import json
import logging
from typing import List

from langchain_core.language_models import BaseChatModel

from ..prompts.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

class ExclusionCritic:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self.prompt_loader = PromptLoader()
    
    def validate_exclusion(self, iv: str, treatment: str, outcome: str, confounders: List[str]) -> bool:
        # Exclusion restriction: does IV affect outcome only through treatment?
        # Confounders not directly relevant here - just check direct pathways
        prompt = self.prompt_loader.format_exclusion_prompt(iv, treatment, outcome, confounders)
        response = self.llm.invoke(prompt)
        result = self._parse_validity(response)
        
        # Log detailed output
        logger.info(json.dumps({
            'name': f'exclusion_critic_{iv}',
            'inputs': {'iv': iv, 'treatment': treatment, 'outcome': outcome, 'confounders': confounders},
            'outputs': {'valid': result},
            'raw_response': response,
        }, default=str))
        
        return result
    
    def _parse_validity(self, response: str) -> bool:
        import re
        match = re.search(r'<Answer>(Valid|Invalid)</Answer>', response)
        return match.group(1) == 'Valid' if match else False