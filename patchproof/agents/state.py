"""Shared State for the PatchProof multi-agent loop.

Every agent is a pure-ish ``state -> partial state`` function. The LangGraph
orchestrator merges each returned dict into the running State; ``total=False``
lets us treat the State as progressively-filled rather than required-from-day-1.

Fields are exactly the ones listed in CLAUDE.md, plus ``vuln_id`` (the verifier
needs an id to know which Vul4J project to slot the patch into).

Lifecycle of the fields, in the order they get written:

    vuln_id, code, max_attempts        -- supplied by the caller
    cwe, triage_report                 -- written by triage()
    candidate_patch, attempt           -- written by patch()
    verdict, logs                      -- written by verify_node()
    repair_feedback                    -- written by self_heal() on retry
    final_result                       -- written by report() at the end
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Verdict = Literal[
    "compile_error",
    "still_vulnerable",
    "regression_broken",
    "verified",
]


class State(TypedDict, total=False):
    # Inputs
    vuln_id: str            # which Vul4J project the verifier slots the patch into
    code: str               # the vulnerable method body the loop must fix
    max_attempts: int       # retry budget for the patch->verify loop

    # Triage outputs
    cwe: str
    triage_report: str

    # Patcher outputs
    candidate_patch: str
    attempt: int            # incremented each time patch() runs (starts at 0)

    # Verifier outputs
    verdict: Verdict
    logs: dict[str, Any]

    # Self-heal output (only present on retries)
    repair_feedback: str

    # Reporter output
    final_result: dict[str, Any]


def initial_state(vuln_id: str, code: str, max_attempts: int = 3) -> State:
    """Construct the entry-state for a new run.

    ``attempt`` starts at 0 and is incremented inside ``patch()``, so after the
    Nth patch attempt the counter reads N. The router checks
    ``attempt >= max_attempts`` to stop after the budget runs out.
    """
    return {
        "vuln_id": vuln_id,
        "code": code,
        "max_attempts": max_attempts,
        "attempt": 0,
    }
