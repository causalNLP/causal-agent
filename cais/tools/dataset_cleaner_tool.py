from typing import Dict, Any, Optional
from langchain.tools import tool
import logging

from cais.components.dataset_cleaner import run_cleaning_stage
from cais.components.state_manager import create_workflow_state_update
from cais.models import DatasetCleanerInput, Variables

logger = logging.getLogger(__name__)

@tool
def dataset_cleaner_tool(dataset_path: str,
                         variables: Dict[str, Any],
                         dataset_description: Optional[str] = None,
                         original_query: Optional[str] = None, causal_method: Optional[str] = None,
                         llm: Optional[Any] = None) -> Dict[str, Any]:
    """
    Build and validate a typed preprocessing contract, execute the approved
    deterministic transformations on a temporary copy, and return the cleaned
    path, contract, and audit report.
    """
    logger.info("Running dataset_cleaner_tool...")
    out = run_cleaning_stage(
        dataset_path=dataset_path,
        variables=variables,
        dataset_description=dataset_description,
        original_query=original_query,
        causal_method=causal_method,
        llm=llm,
    )

    # Wire into workflow state
    wf = create_workflow_state_update(
        current_step="dataset_cleaning",
        step_completed_flag=True,
        next_tool="method_validator_tool",
        next_step_reason="Dataset cleaned; proceed to validate method assumptions."
    )

    result = {
        "cleaned_dataset_path": out["cleaned_dataset_path"],
        "cleaning_report_md": out["cleaning_report_md"],
        "generated_code": out.get("generated_code", ""),
        "stdout": out.get("stdout", ""),
        "stderr": out.get("stderr", ""),
        "typed_data_contract": out.get("typed_data_contract", {}),
        "typed_data_contract_path": out.get("typed_data_contract_path"),
        "survival_roles": out.get("survival_roles", {}),
        **wf
    }
    logger.info(f"Cleaning complete. Saved: {result['cleaned_dataset_path']}")
    return result
