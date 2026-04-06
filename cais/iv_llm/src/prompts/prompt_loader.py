from __future__ import annotations

from pathlib import Path

class PromptLoader:
    def __init__(self, prompts_dir: str | Path | None = None) -> None:
        # Default to the prompts folder shipped with this package.
        self.prompts_dir = Path(prompts_dir) if prompts_dir is not None else Path(__file__).resolve().parent
    
    def load_prompt(self, prompt_name: str) -> str:
        prompt_path = Path(self.prompts_dir) / f"{prompt_name}.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def format_hypothesizer_prompt(self, treatment: str, outcome: str, k: int = 5, context: str = "") -> str:
        template = self.load_prompt("hypothesizer")
        context_text = f"\n\nAdditional context: {context}" if context else ""
        return template.format(treatment=treatment, outcome=outcome, k=k) + context_text
    
    def format_confounder_prompt(self, treatment: str, outcome: str, j: int = 5, context: str = "") -> str:
        template = self.load_prompt("confounder_miner")
        context_text = f"\n\nAdditional context: {context}" if context else ""
        return template.format(treatment=treatment, outcome=outcome, j=j) + context_text
    
    def format_exclusion_prompt(
        self,
        iv: str,
        treatment: str,
        outcome: str,
        confounders: list[str] | None = None,
        context: str = "",
    ) -> str:
        template = self.load_prompt("exclusion_critic")
        context_text = f"\n\nAdditional context: {context}" if context else ""
        return template.format(iv=iv, treatment=treatment, outcome=outcome) + context_text
    
    def format_independence_prompt(
        self,
        iv: str,
        treatment: str,
        outcome: str,
        confounder: str,
        context: str = "",
    ) -> str:
        template = self.load_prompt("independence_critic")
        context_text = f"\n\nAdditional context: {context}" if context else ""
        return template.format(iv=iv, treatment=treatment, outcome=outcome,
                             confounder=confounder) + context_text
    
    def format_conceptual_equivalence_prompt(self, proposed_iv: str, gold_ivs: str | list[str]) -> str:
        template = self.load_prompt("conceptual_equivalence")
        return template.format(proposed_iv=proposed_iv, gold_ivs=gold_ivs)
    
    def format_human_proxy_prompt(self, variable1: str, variable2: str) -> str:
        template = self.load_prompt("human_proxy")
        return template.format(variable1=variable1, variable2=variable2)