from ..prompts.prompt_loader import PromptLoader
from ..variable_utils import extract_available_columns, filter_to_available


class ConfounderMiner:
    def __init__(self, llm_client, j=5):
        self.llm_client = llm_client
        self.j = j
        self.prompt_loader = PromptLoader()
    
    def identify_confounders(self, treatment, outcome, context=""):
        prompt = self.prompt_loader.format_confounder_prompt(treatment, outcome, self.j, context=context)
        response = self.llm_client.generate(prompt)
        confounders_raw = self._parse_confounders(response)

        available_cols = extract_available_columns(context)
        confounders = (
            filter_to_available(confounders_raw, available_cols)
            if available_cols
            else confounders_raw
        )

        confounders = confounders[: self.j]
        
        # Log detailed output
        from ..llm.output_tracker import tracker
        tracker.log_agent_output(
            'confounder_miner',
            {'treatment': treatment, 'outcome': outcome, 'j': self.j},
            {'confounders': confounders},
            response
        )
        
        return confounders
    
    def _parse_confounders(self, response):
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