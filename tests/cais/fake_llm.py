"""A scripted stand-in for the chat model, so tests never call a real LLM API.

``ScriptedChatModel`` answers from a fixed script:

* ``with_structured_output(Schema)`` returns the object registered for
  ``Schema.__name__`` in ``structured``.
* a plain ``invoke`` returns the reply of the first ``(substring, reply)`` rule
  in ``text_rules`` whose substring occurs in the prompt.

A call with no matching entry raises ``UnscriptedCall`` (instead of guessing),
so a new LLM call in the pipeline is not silently answered. Much of the
pipeline catches exceptions around LLM calls, so each unscripted call is also
recorded in ``unscripted`` for the test to assert on. Every call is recorded in
``calls``.
"""

from typing import Any, Dict, List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field


class UnscriptedCall(AssertionError):
    """The code under test made an LLM call that the script does not cover."""


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, BaseMessage):
        return str(prompt.content)
    if isinstance(prompt, (list, tuple)):
        return "\n".join(_prompt_text(p) for p in prompt)
    if hasattr(prompt, "to_messages"):
        return _prompt_text(prompt.to_messages())
    return str(prompt)


class ScriptedChatModel(BaseChatModel):
    structured: Dict[str, Any] = Field(default_factory=dict)
    text_rules: List[Tuple[str, str]] = Field(default_factory=list)
    calls: List[Tuple[str, str]] = Field(default_factory=list)
    unscripted: List[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _reply(self, prompt: str) -> str:
        self.calls.append(("text", prompt))
        for needle, reply in self.text_rules:
            if needle in prompt:
                return reply
        self.unscripted.append(f"text: {prompt[:300]!r}")
        raise UnscriptedCall(f"No scripted text reply for prompt starting: {prompt[:300]!r}")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = self._reply(_prompt_text(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def with_structured_output(self, schema, **kwargs):
        name = getattr(schema, "__name__", str(schema))

        def run(prompt):
            self.calls.append((f"structured:{name}", _prompt_text(prompt)))
            if name not in self.structured:
                self.unscripted.append(f"{name}: {_prompt_text(prompt)[:300]!r}")
                raise UnscriptedCall(
                    f"No scripted structured reply for {name}; prompt starts: {_prompt_text(prompt)[:300]!r}"
                )
            reply = self.structured[name]
            if callable(reply):
                reply = reply(_prompt_text(prompt))
            if isinstance(reply, dict):
                reply = schema(**reply)
            return reply

        return RunnableLambda(run)


def install_llm(monkeypatch, llm: BaseChatModel) -> BaseChatModel:
    """Make every ``get_llm_client`` in the ``cais`` package return ``llm``.

    The pipeline modules do ``from cais.config import get_llm_client``, so the
    name is bound once per module; patch each binding, not only ``cais.config``.
    """
    import sys

    for name, module in list(sys.modules.items()):
        if (name == "cais" or name.startswith("cais.")) and hasattr(module, "get_llm_client"):
            monkeypatch.setattr(module, "get_llm_client", lambda *args, **kwargs: llm)
    return llm
