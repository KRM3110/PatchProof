"""Patcher agent — produce a candidate fix.

The real Patcher calls a fine-tuned model via vLLM. Here it's dependency-
injected so tests run against a scriptable stub. The patcher MUST pass
``state['repair_feedback']`` to the model on every call after the first, so
the Self-Heal feedback actually conditions the next attempt.

The ``model_client`` must expose:

    generate_patch(
        *, code: str, cwe: str | None, triage_report: str | None,
        repair_feedback: str | None,
    ) -> str
"""

from __future__ import annotations

from .state import State


def patch(state: State, *, model_client) -> dict:
    attempt = state.get("attempt", 0) + 1
    candidate = model_client.generate_patch(
        code=state["code"],
        cwe=state.get("cwe"),
        triage_report=state.get("triage_report"),
        repair_feedback=state.get("repair_feedback"),
    )
    return {
        "candidate_patch": candidate,
        "attempt": attempt,
    }
