from __future__ import annotations

from pathlib import Path

class PromptLoader:
    def __init__(self, prompts_dir: str | Path | None = None):
        # Default to the prompts folder shipped with this package.
        self.prompts_dir = Path(prompts_dir) if prompts_dir is not None else Path(__file__).resolve().parent
    
    def load_prompt(self, prompt_name):
        prompt_path = Path(self.prompts_dir) / f"{prompt_name}.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def format_hypothesizer_prompt(self, treatment, outcome, k=5, context=""):
        template = self.load_prompt("hypothesizer")
        context_text = f"\n\nAdditional context: {context}" if context else ""
        return template.format(treatment=treatment, outcome=outcome, k=k) + context_text
    
    def format_confounder_prompt(self, treatment, outcome, j=5, context=""):
        template = self.load_prompt("confounder_miner")
        context_text = f"\n\nAdditional context: {context}" if context else ""
        return template.format(treatment=treatment, outcome=outcome, j=j) + context_text
    
    def format_exclusion_prompt(self, iv, treatment, outcome, confounders=None, context=""):
        template = self.load_prompt("exclusion_critic")
        context_text = f"\n\nAdditional context: {context}" if context else ""
        return template.format(iv=iv, treatment=treatment, outcome=outcome) + context_text
    
    def format_independence_prompt(self, iv, treatment, outcome, confounder, context=""):
        template = self.load_prompt("independence_critic")
        context_text = f"\n\nAdditional context: {context}" if context else ""
        return template.format(iv=iv, treatment=treatment, outcome=outcome,
                             confounder=confounder) + context_text
    
    def format_conceptual_equivalence_prompt(self, proposed_iv, gold_ivs):
        template = self.load_prompt("conceptual_equivalence")
        return template.format(proposed_iv=proposed_iv, gold_ivs=gold_ivs)
    
    def format_human_proxy_prompt(self, variable1, variable2):
        template = self.load_prompt("human_proxy")
        return template.format(variable1=variable1, variable2=variable2)