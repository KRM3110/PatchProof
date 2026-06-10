"""Verifier node — thin wrapper around the deterministic verifier.

No logic of its own. Pulls (vuln_id, candidate_patch) from State and hands them
to the injected verifier_fn. In production this is wired to
``verifier.verify.verify``; in tests it's a stub that returns a scripted
verdict sequence.

The ``verifier_fn`` must be callable as:

    verifier_fn(vuln_id: str, patched_code: str) -> dict with keys
        {"verdict": Verdict, "logs": dict}
"""

from __future__ import annotations

from .state import State


def verify_node(state: State, *, verifier_fn) -> dict:
    result = verifier_fn(state["vuln_id"], state["candidate_patch"])
    return {
        "verdict": result["verdict"],
        "logs": result.get("logs", {}),
    }
