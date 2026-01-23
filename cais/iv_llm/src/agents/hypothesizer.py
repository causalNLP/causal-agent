from ..prompts.prompt_loader import PromptLoader
from ..variable_utils import extract_available_columns, filter_to_available, fallback_candidates


class Hypothesizer:
    def __init__(self, llm_client, k=5):
        self.llm_client = llm_client
        self.k = k
        self.prompt_loader = PromptLoader()
    
    def propose_ivs(self, treatment, outcome, context=""):
        prompt = self.prompt_loader.format_hypothesizer_prompt(treatment, outcome, self.k, context=context)
        response = self.llm_client.generate(prompt)
        ivs_raw = self._parse_ivs(response)

        available_cols = extract_available_columns(context)
        ivs = filter_to_available(ivs_raw, available_cols) if available_cols else ivs_raw

        # Ensure we return only dataset column names (and ideally not treatment/outcome).
        if available_cols and not ivs:
            ivs = fallback_candidates(available_cols, exclude=[treatment, outcome])[: self.k]

        ivs = ivs[: self.k]
        
        # Log detailed output
        from ..llm.output_tracker import tracker
        tracker.log_agent_output(
            'hypothesizer',
            {'treatment': treatment, 'outcome': outcome, 'k': self.k},
            {'proposed_ivs': ivs},
            response
        )
        
        return ivs
    
    def _parse_ivs(self, response):
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