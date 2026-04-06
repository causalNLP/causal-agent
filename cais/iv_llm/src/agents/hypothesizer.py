import json
import logging
from typing import List

from langchain_core.language_models import BaseChatModel

from ..prompts.prompt_loader import PromptLoader
from ..variable_utils import extract_available_columns, filter_to_available, fallback_candidates

logger = logging.getLogger(__name__)

class Hypothesizer:
    def __init__(self, llm: BaseChatModel, k: int = 5) -> None:
        self.llm = llm
        self.k = k
        self.prompt_loader = PromptLoader()
    
    def propose_ivs(self, treatment: str, outcome: str, context: str = "") -> List[str]:
        prompt = self.prompt_loader.format_hypothesizer_prompt(treatment, outcome, self.k, context=context)
        response = self.llm.invoke(prompt)
        ivs_raw = self._parse_ivs(response)

        available_cols = extract_available_columns(context)
        ivs = filter_to_available(ivs_raw, available_cols) if available_cols else ivs_raw

        if available_cols and not ivs:
            ivs = fallback_candidates(available_cols, exclude=[treatment, outcome])[: self.k]

        ivs = ivs[: self.k]
        
        logger.info(json.dumps({
            'name': 'hypothesizer',
            'inputs': {'treatment': treatment, 'outcome': outcome, 'k': self.k},
            'outputs': {'proposed_ivs': ivs},
            'raw_response': response,
        }, default=str))
        
        return ivs
    
    def _parse_ivs(self, response: str) -> List[str]:
        import re

        def _clean(name: str) -> str:
            return name.strip().strip('"\'').strip('`').strip('*').strip()
        
        # Try XML format first
        match = re.search(r'<Answer>\[(.*?)\]</Answer>', response)
        if match:
            ivs_str = match.group(1)
            ivs = [_clean(iv) for iv in ivs_str.split(',')]
            return ivs[:self.k]
        
        # Fallback: look for bracket format
        bracket_match = re.search(r'\[([^\]]+)\]', response)
        if bracket_match:
            ivs_str = bracket_match.group(1)
            ivs = [_clean(iv) for iv in ivs_str.split(',')]
            return ivs[:self.k]
        
        # Fallback: look for numbered list format
        lines = response.split('\n')
        ivs = []
        for line in lines:
            line = line.strip()
            # Match patterns like "1. Something:" or "- Something:"
            if re.match(r'^\d+\.\s+(.+?):', line):
                iv = _clean(re.match(r'^\d+\.\s+(.+?):', line).group(1))
                ivs.append(iv)
            elif re.match(r'^-\s+(.+?):', line):
                iv = _clean(re.match(r'^-\s+(.+?):', line).group(1))
                ivs.append(iv)
        
        if ivs:
            return ivs[:self.k]
        
        print(f"WARNING: Could not parse IVs from response: {response[:200]}...")
        return []