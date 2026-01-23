from ..prompts.prompt_loader import PromptLoader


class IndependenceCritic:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_loader = PromptLoader()
    
    def validate_independence(self, iv, treatment, outcome, confounders):
        # Independence: check IV against each confounder separately
        responses = {}
        
        for confounder in confounders:
            prompt = self.prompt_loader.format_independence_prompt(iv, treatment, outcome, confounder)
            response = self.llm_client.generate(prompt)
            responses[confounder] = response
            
            if not self._parse_validity(response):
                # Log detailed output
                from ..llm.output_tracker import tracker
                tracker.log_agent_output(
                    f'independence_critic_{iv}',
                    {'iv': iv, 'treatment': treatment, 'outcome': outcome, 'confounders': confounders},
                    {'valid': False, 'failed_on': confounder},
                    responses
                )
                return False
        
        # Log successful validation
        from ..llm.output_tracker import tracker
        tracker.log_agent_output(
            f'independence_critic_{iv}',
            {'iv': iv, 'treatment': treatment, 'outcome': outcome, 'confounders': confounders},
            {'valid': True},
            responses
        )
        
        return True
    
    def _parse_validity(self, response):
        import re
        match = re.search(r'<Answer>(Valid|Invalid)</Answer>', response)
        return match.group(1) == 'Valid' if match else False