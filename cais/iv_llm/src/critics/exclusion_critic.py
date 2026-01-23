from ..prompts.prompt_loader import PromptLoader


class ExclusionCritic:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_loader = PromptLoader()
    
    def validate_exclusion(self, iv, treatment, outcome, confounders):
        # Exclusion restriction: does IV affect outcome only through treatment?
        # Confounders not directly relevant here - just check direct pathways
        prompt = self.prompt_loader.format_exclusion_prompt(iv, treatment, outcome, confounders)
        response = self.llm_client.generate(prompt)
        result = self._parse_validity(response)
        
        # Log detailed output
        from ..llm.output_tracker import tracker
        tracker.log_agent_output(
            f'exclusion_critic_{iv}',
            {'iv': iv, 'treatment': treatment, 'outcome': outcome, 'confounders': confounders},
            {'valid': result},
            response
        )
        
        return result
    
    def _parse_validity(self, response):
        import re
        match = re.search(r'<Answer>(Valid|Invalid)</Answer>', response)
        return match.group(1) == 'Valid' if match else False