# IV-LLM package

import logging
import os
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _get_log_path() -> Path:
    env_dir = os.getenv("IV_LLM_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir) / "iv_llm.jsonl"
    return _find_project_root() / "logs" / "iv_llm.jsonl"


# Configure a file handler on the "cais.iv_llm" logger so that every child
# logger (agents, critics, etc.) automatically writes to the IV-LLM log file.
_logger = logging.getLogger(__name__)  # "cais.iv_llm"
if not _logger.handlers:
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = True  # still propagate to root for console output
    try:
        _log_path = _get_log_path()
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _handler = logging.FileHandler(str(_log_path), encoding="utf-8")
        _handler.setLevel(logging.INFO)
        _handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(_handler)
    except Exception:
        pass