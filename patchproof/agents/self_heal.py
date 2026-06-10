"""Self-Heal agent — read verifier logs, diagnose, write repair feedback.

Runs only when the verifier returned a non-``verified`` verdict AND we still
have attempts left. Its sole job is to produce ``repair_feedback`` that the
next Patcher call will consume. The orchestrator wires Self-Heal -> Patcher,
so the feedback is guaranteed to land in the prompt for the next attempt.

The ``llm_client`` must expose:

    diagnose(verdict: str, logs: dict) -> str
"""

from __future__ import annotations

from .state import State


def self_heal(state: State, *, llm_client) -> dict:
    feedback = llm_client.diagnose(
        verdict=state["verdict"],
        logs=state.get("logs", {}),
    )
    return {"repair_feedback": feedback}
