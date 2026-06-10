"""Real Patcher model client — talks to the vLLM OpenAI-compatible endpoint
stood up by ``serve/serve.sh``.

This file is the inference-side mirror of ``train/finetune.py``: both import
``build_messages`` from ``config`` so the format the model saw during SFT is
byte-identical to what it sees at inference. Do NOT redefine the template
here — if it ever needs to change, change it in ``config.py``.

Interface contract
------------------
``VLLMModelClient.generate_patch`` matches ``agents.stubs.StubModelClient.generate_patch``
exactly, so ``build_graph(model_client=...)`` is the only seam that flips
between the CPU test path and the real GPU path. The agents themselves do
not change.

Optional context: the training template is just ``[system, user(CWE, vulnerable)]``.
``triage_report`` and ``repair_feedback`` are appended to the user turn as
extra context sections; the base template is left intact.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from config import build_messages


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "patchproof-merged"
DEFAULT_API_KEY = "EMPTY"   # vLLM accepts any non-empty string when auth is off
DEFAULT_TIMEOUT = 120.0
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024


_FENCED_JAVA = re.compile(r"```java\s*\n(.*?)```", re.DOTALL)
_FENCED_ANY = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.DOTALL)


class ModelClientError(RuntimeError):
    """Raised when the vLLM endpoint is unreachable, mis-configured, or
    returns an empty / malformed response. The agent loop catches this and
    routes through the self-heal path the same way it handles a bad patch."""


@dataclass
class VLLMModelClient:
    """Patcher client over the vLLM OpenAI-compatible endpoint.

    All connection knobs are env-overridable so the same code runs on the
    MI300X (default port 8000, localhost) and against a port-forwarded
    endpoint on a dev box without code changes.

    Env vars:
        PATCHPROOF_VLLM_BASE_URL   default http://127.0.0.1:8000/v1
        PATCHPROOF_VLLM_MODEL      default patchproof-merged
        PATCHPROOF_VLLM_API_KEY    default "EMPTY"
    """

    base_url: str = field(default_factory=lambda: os.environ.get(
        "PATCHPROOF_VLLM_BASE_URL", DEFAULT_BASE_URL))
    model: str = field(default_factory=lambda: os.environ.get(
        "PATCHPROOF_VLLM_MODEL", DEFAULT_MODEL))
    api_key: str = field(default_factory=lambda: os.environ.get(
        "PATCHPROOF_VLLM_API_KEY", DEFAULT_API_KEY))
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT
    extract_fenced: bool = True

    _client: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover - import-time failure
            raise ModelClientError(
                "openai client not installed. `pip install openai`."
            ) from e
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    # ------------------------------------------------------------------ API

    def generate_patch(
        self,
        *,
        code: str,
        cwe: str | None,
        triage_report: str | None,
        repair_feedback: str | None,
    ) -> str:
        msgs = build_messages(cwe or "CWE-UNKNOWN", code)

        # Append optional context to the user turn only; system + user-prefix
        # remain byte-identical to training so the SFT prior still applies.
        extras: list[str] = []
        if triage_report:
            extras.append(f"Triage report:\n{triage_report}")
        if repair_feedback:
            extras.append(
                "Previous attempt failed verification. Repair feedback:\n"
                f"{repair_feedback}"
            )
        if extras:
            msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + "\n\n".join(extras)

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            raise ModelClientError(
                f"vLLM endpoint at {self.base_url} unreachable or rejected the "
                f"request (model={self.model!r}): {e}"
            ) from e

        if not getattr(resp, "choices", None):
            raise ModelClientError(f"vLLM returned no choices: {resp!r}")

        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise ModelClientError("vLLM returned an empty completion")

        return _extract_java(text) if self.extract_fenced else text


def _extract_java(text: str) -> str:
    """Pull the ```java fenced block — the format the trainer's assistant
    turn used. Falls back to any fenced block, then to raw text, so a model
    that drifts on formatting still gets a chance through the verifier."""
    m = _FENCED_JAVA.search(text)
    if m:
        return m.group(1).rstrip()
    m = _FENCED_ANY.search(text)
    if m:
        return m.group(1).rstrip()
    return text.strip()


__all__ = ("VLLMModelClient", "ModelClientError")
