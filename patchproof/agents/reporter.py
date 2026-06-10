"""Reporter agent — assemble the final result (explanation + evidence).

Runs as the last node of the graph regardless of outcome. If the verdict is
not ``verified`` (budget exhausted), the result is flagged as escalated so the
demo/orchestrator caller can surface it differently.

The ``llm_client`` must expose:

    explain(state: dict) -> str
"""

from __future__ import annotations

from .state import State


def report(state: State, *, llm_client) -> dict:
    explanation = llm_client.explain(dict(state))
    final = {
        "verdict": state.get("verdict"),
        "explanation": explanation,
        "evidence": {
            "vuln_id": state.get("vuln_id"),
            "cwe": state.get("cwe"),
            "candidate_patch": state.get("candidate_patch"),
            "logs": state.get("logs"),
            "attempts": state.get("attempt"),
            "max_attempts": state.get("max_attempts"),
        },
        "escalated": state.get("verdict") != "verified",
    }
    return {"final_result": final}
