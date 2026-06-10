"""End-to-end orchestrator tests using stub deps.

These exercise the compiled LangGraph the same way production will, but with
no LLM, no GPU, and no Docker. The stub clients in ``agents.stubs`` record
their calls, so we can assert on call counts and on what the Patcher actually
received on the retry.
"""

from __future__ import annotations

from agents.graph import build_graph
from agents.state import initial_state
from agents.stubs import StubLLMClient, StubModelClient, StubVerifierFn


def _run(verdicts: list[str], patches: list[str] | None = None, max_attempts: int = 3):
    """Build the graph with stubs scripted to ``verdicts`` and run it."""
    llm = StubLLMClient()
    model = StubModelClient(patches=list(patches or []))
    verifier = StubVerifierFn(verdicts=list(verdicts))
    graph = build_graph(llm_client=llm, model_client=model, verifier_fn=verifier)
    final = graph.invoke(initial_state(
        vuln_id="VUL4J-10",
        code="// vulnerable method body",
        max_attempts=max_attempts,
    ))
    return final, llm, model, verifier


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_verifies_on_first_try():
    final, llm, model, verifier = _run(verdicts=["verified"])

    assert len(model.calls) == 1, "patcher should be called exactly once"
    assert len(verifier.calls) == 1
    assert llm.diagnose_calls == [], "self-heal must NOT run when first attempt verifies"

    # The first patcher call has no repair feedback yet.
    assert model.calls[0]["repair_feedback"] is None

    assert final["verdict"] == "verified"
    assert final["attempt"] == 1
    result = final["final_result"]
    assert result["verdict"] == "verified"
    assert result["escalated"] is False
    assert result["evidence"]["attempts"] == 1


# ---------------------------------------------------------------------------
# Retry path: fail once, then verify. Confirm repair feedback threaded through.
# ---------------------------------------------------------------------------

def test_retry_threads_repair_feedback_into_second_patch():
    final, llm, model, verifier = _run(
        verdicts=["still_vulnerable", "verified"],
        patches=["// attempt 1", "// attempt 2"],
    )

    assert len(model.calls) == 2, "patcher should be called twice"
    assert len(verifier.calls) == 2
    assert len(llm.diagnose_calls) == 1, "self-heal should run exactly once on retry"

    # First call: no repair feedback. Second call: repair feedback present and
    # equal to whatever self_heal produced (the stub returns a fixed string).
    assert model.calls[0]["repair_feedback"] is None
    assert model.calls[1]["repair_feedback"] == llm.diagnose_text, (
        "patcher must consume the feedback that self_heal wrote on the retry"
    )

    # Self-heal received the failing verdict and the logs from the first run.
    diagnose_args = llm.diagnose_calls[0]
    assert diagnose_args["verdict"] == "still_vulnerable"
    assert diagnose_args["logs"]["attempt_index"] == 1

    assert final["verdict"] == "verified"
    assert final["attempt"] == 2
    assert final["final_result"]["escalated"] is False


# ---------------------------------------------------------------------------
# Budget exhaustion: always fail, stops after max_attempts, escalates.
# ---------------------------------------------------------------------------

def test_budget_exhaustion_escalates_after_max_attempts():
    # Always fail. Verifier returns its default ("still_vulnerable") forever.
    final, llm, model, verifier = _run(verdicts=[], max_attempts=3)

    assert len(model.calls) == 3, "patcher must be called exactly max_attempts times"
    assert len(verifier.calls) == 3
    assert len(llm.diagnose_calls) == 2, (
        "self-heal runs after every failure except the last "
        "(the last failure routes straight to report)"
    )

    # Repair feedback should be absent on the first call and present on the
    # later two calls.
    assert model.calls[0]["repair_feedback"] is None
    assert model.calls[1]["repair_feedback"] == llm.diagnose_text
    assert model.calls[2]["repair_feedback"] == llm.diagnose_text

    assert final["verdict"] != "verified"
    assert final["attempt"] == 3
    result = final["final_result"]
    assert result["escalated"] is True
    assert result["evidence"]["attempts"] == 3
    assert result["evidence"]["max_attempts"] == 3


# ---------------------------------------------------------------------------
# Triage runs exactly once even across retries.
# ---------------------------------------------------------------------------

def test_triage_runs_once_per_run():
    _, llm, _, _ = _run(verdicts=["still_vulnerable", "still_vulnerable", "verified"])
    assert len(llm.triage_calls) == 1
