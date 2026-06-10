"""LangGraph orchestrator for PatchProof.

The router + the LangGraph wiring **are** the orchestrator. There is no LLM
sitting on top making decisions; the retry budget and the verdict-driven
branching are deterministic.

Topology
--------

    START -> triage -> patch -> verify -> (router)
                          ^                  |
                          |                  +-- verified           -> report -> END
                          |                  +-- attempt>=budget    -> report -> END
                          +-- self_heal <----+-- otherwise

``attempt`` is incremented inside ``patch()``. The router runs after ``verify``
and reads ``attempt`` + ``max_attempts`` + ``verdict``.

Dependency injection
--------------------
``build_graph(llm_client=..., model_client=..., verifier_fn=...)`` returns a
compiled LangGraph. Tests pass the stubs from ``agents.stubs``; production
will pass the real vLLM client and ``verifier.verify.verify``.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from . import patcher, reporter, self_heal, triage, verifier_agent
from .state import State

RouterDecision = Literal["report", "self_heal"]


def route_after_verify(state: State) -> RouterDecision:
    """Decide the next node after the verifier writes its verdict.

    Order matters: a ``verified`` verdict beats budget exhaustion (a fix that
    lands on the final allowed attempt should still be reported as a success,
    not an escalation).
    """
    if state.get("verdict") == "verified":
        return "report"
    if state.get("attempt", 0) >= state.get("max_attempts", 0):
        return "report"
    return "self_heal"


def build_graph(*, llm_client, model_client, verifier_fn):
    """Compile the LangGraph with the given injected dependencies."""

    def _triage(s: State) -> dict:
        return triage.triage(s, llm_client=llm_client)

    def _patch(s: State) -> dict:
        return patcher.patch(s, model_client=model_client)

    def _verify(s: State) -> dict:
        return verifier_agent.verify_node(s, verifier_fn=verifier_fn)

    def _self_heal(s: State) -> dict:
        return self_heal.self_heal(s, llm_client=llm_client)

    def _report(s: State) -> dict:
        return reporter.report(s, llm_client=llm_client)

    g: StateGraph = StateGraph(State)
    g.add_node("triage", _triage)
    g.add_node("patch", _patch)
    g.add_node("verify", _verify)
    g.add_node("self_heal", _self_heal)
    g.add_node("report", _report)

    g.set_entry_point("triage")
    g.add_edge("triage", "patch")
    g.add_edge("patch", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"report": "report", "self_heal": "self_heal"},
    )
    g.add_edge("self_heal", "patch")
    g.add_edge("report", END)

    return g.compile()
