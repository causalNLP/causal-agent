from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _json_default(value: Any) -> str:
    # Best-effort serialization fallback.
    try:
        return str(value)
    except Exception:
        return repr(value)


@dataclass
class OutputTracker:
    """Best-effort logger for IV-LLM agent/critic outputs.

    This is intentionally lightweight: failures to write logs should never
    break the core pipeline.
    """

    log_path: Path

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_agent_output(
        self,
        name: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        raw_response: Any,
        *,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "inputs": inputs,
            "outputs": outputs,
            "raw_response": raw_response,
        }
        if meta:
            record["meta"] = meta

        try:
            line = json.dumps(record, ensure_ascii=False, default=_json_default)
        except Exception:
            # As a last resort, stringify the whole record.
            line = json.dumps({"ts": record.get("ts"), "name": name, "record": _json_default(record)})

        try:
            with self._lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            # Never let logging failures crash the pipeline.
            return


def _default_log_path() -> Path:
    # Allow callers/tests to override.
    env_dir = os.getenv("IV_LLM_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir) / "iv_llm_outputs.jsonl"

    # Repo-level default (relative to CWD), matching the existing cache_dir style.
    return Path(".cache") / "iv_llm" / "iv_llm_outputs.jsonl"


tracker = OutputTracker(_default_log_path())
