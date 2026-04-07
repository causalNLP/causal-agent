import json
import logging
from typing import List

from langchain_core.language_models import BaseChatModel

from ..prompts.prompt_loader import PromptLoader
from ..variable_utils import extract_available_columns, filter_to_available

logger = logging.getLogger(__name__)

class ConfounderMiner:
    def __init__(self, llm: BaseChatModel, j: int = 5) -> None:
        self.llm = llm
        self.j = j
        self.prompt_loader = PromptLoader()
    
    def identify_confounders(self, treatment: str, outcome: str, context: str = "") -> List[str]:
        prompt = self.prompt_loader.format_confounder_prompt(treatment, outcome, self.j, context=context)
        response = self.llm.invoke(prompt)
        confounders_raw = self._parse_confounders(response)

        available_cols = extract_available_columns(context)
        confounders = (
            filter_to_available(confounders_raw, available_cols)
            if available_cols
            else confounders_raw
        )

        confounders = confounders[: self.j]
        
        logger.info(json.dumps({
            'name': 'confounder_miner',
            'inputs': {'treatment': treatment, 'outcome': outcome, 'j': self.j},
            'outputs': {'confounders': confounders},
            'raw_response': response,
        }, default=str))
        
        return confounders
    
    def _parse_confounders(self, response: str) -> List[str]:
        import re

        def _clean(name: str) -> str:
            return name.strip().strip('"\'').strip('`').strip('*').strip()
        
        # Try XML format first
        match = re.search(r'<Answer>\[(.*?)\]</Answer>', response)
        if match:
            confounders_str = match.group(1)
            confounders = [_clean(c) for c in confounders_str.split(',')]
            return confounders[:self.j]
        
        # Fallback: look for bracket format without XML
        bracket_match = re.search(r'\[([^\]]+)\]', response)
        if bracket_match:
            confounders_str = bracket_match.group(1)
            confounders = [_clean(c) for c in confounders_str.split(',')]
            return confounders[:self.j]
        
        print(f"WARNING: Could not parse confounders from response: {response[:200]}...")
        return []