"""Stubs for the injected dependencies — used by unit tests and any CPU-only
end-to-end exercise of the graph. No real LLM, no GPU, no Docker.

Each stub records its calls so tests can assert on what was passed in. The
``Stub*Client`` classes mirror the surface the real LLM/model client will
expose; the real wire-up lands in Step 4/5 and the agents themselves don't
change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Triage / Self-Heal / Reporter LLM client
# ---------------------------------------------------------------------------

@dataclass
class StubLLMClient:
    """Scriptable stand-in for the LLM used by triage, self-heal, and reporter.

    Pass canned responses via the constructor. Every call is recorded on the
    matching ``*_calls`` list so tests can inspect call count + arguments.
    """

    triage_result: dict = field(default_factory=lambda: {
        "cwe": "CWE-89",
        "report": "SQL injection — user input concatenated into a query.",
    })
    diagnose_text: str = "Use parameterized queries and never concatenate user input."
    explain_text: str = "Replaced string concatenation with a PreparedStatement."

    triage_calls: list[dict] = field(default_factory=list)
    diagnose_calls: list[dict] = field(default_factory=list)
    explain_calls: list[dict] = field(default_factory=list)

    def triage(self, code: str) -> dict:
        self.triage_calls.append({"code": code})
        return dict(self.triage_result)

    def diagnose(self, *, verdict: str, logs: dict) -> str:
        self.diagnose_calls.append({"verdict": verdict, "logs": logs})
        return self.diagnose_text

    def explain(self, state: dict) -> str:
        self.explain_calls.append({"state": state})
        return self.explain_text


# ---------------------------------------------------------------------------
# Patcher model client
# ---------------------------------------------------------------------------

@dataclass
class StubModelClient:
    """Scriptable stand-in for the Patcher's fine-tuned-model client.

    ``patches`` is consumed in order; once exhausted, ``default_patch`` is
    returned. Every call is recorded so tests can confirm that repair feedback
    was actually threaded through on retries.
    """

    patches: list[str] = field(default_factory=list)
    default_patch: str = "// stub patch"
    calls: list[dict] = field(default_factory=list)

    def generate_patch(
        self,
        *,
        code: str,
        cwe: str | None,
        triage_report: str | None,
        repair_feedback: str | None,
    ) -> str:
        self.calls.append({
            "code": code,
            "cwe": cwe,
            "triage_report": triage_report,
            "repair_feedback": repair_feedback,
        })
        if self.patches:
            return self.patches.pop(0)
        return self.default_patch


# ---------------------------------------------------------------------------
# Verifier function
# ---------------------------------------------------------------------------

@dataclass
class StubVerifierFn:
    """Scriptable verifier_fn.

    ``verdicts`` is consumed in order; the loop test cases (happy path, retry,
    budget exhaustion) just script the sequence they need. Once the list is
    exhausted, ``default_verdict`` is returned so a test misconfiguration
    surfaces as a real assertion failure (e.g. "still_vulnerable forever")
    rather than a crash.
    """

    verdicts: list[str] = field(default_factory=list)
    default_verdict: str = "still_vulnerable"
    logs_template: dict = field(default_factory=lambda: {"steps": [], "note": "stub"})
    calls: list[dict] = field(default_factory=list)

    def __call__(self, vuln_id: str, candidate_patch: str) -> dict:
        self.calls.append({"vuln_id": vuln_id, "candidate_patch": candidate_patch})
        verdict = self.verdicts.pop(0) if self.verdicts else self.default_verdict
        logs = dict(self.logs_template)
        logs["attempt_index"] = len(self.calls)
        logs["verdict"] = verdict
        return {"verdict": verdict, "logs": logs}


VerifierFn = Callable[[str, str], dict[str, Any]]
