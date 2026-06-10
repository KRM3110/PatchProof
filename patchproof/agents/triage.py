"""Triage agent — classify the CWE and locate the vulnerable code.

The real implementation will use an LLM (and optionally Semgrep). The agent
itself is a thin ``state -> partial state`` function; the LLM is injected so
unit tests can pass a stub.

The ``llm_client`` must expose:

    triage(code: str) -> dict with keys {"cwe": str, "report": str}
"""

from __future__ import annotations

from .state import State


def triage(state: State, *, llm_client) -> dict:
    code = state["code"]
    out = llm_client.triage(code)
    return {
        "cwe": out["cwe"],
        "triage_report": out["report"],
    }
