"""Deterministic verifier — the single source of truth for "is this patch good?".

verify(vuln_id, patched_code) -> (verdict, logs)

Verdict is one of:
    compile_error      — `vul4j compile` failed
    still_vulnerable   — compiled, but a PoV/exploit test still fails
    regression_broken  — PoV passes, but a regression test fails
    verified           — PoV passes AND regression suite passes

`logs` is a single dict that the Self-Heal agent and Reporter agent both read.
Keep it serializable.

Never put an LLM in this file. The whole project depends on this being
deterministic.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from . import vul4j_runner as vr

Verdict = Literal["compile_error", "still_vulnerable", "regression_broken", "verified"]


@dataclass
class VerifyLogs:
    vuln_id: str
    steps: list[dict] = field(default_factory=list)   # ordered command-by-command record
    testing_results: dict | None = None               # raw VUL4J/testing_results.json
    vulnerability_info: dict | None = None
    error: str | None = None                          # set when verify aborts before a verdict

    def add(self, label: str, res: vr.CmdResult) -> None:
        self.steps.append({
            "step": label,
            "cmd": res.cmd,
            "returncode": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        })

    def to_dict(self) -> dict:
        return {
            "vuln_id": self.vuln_id,
            "steps": self.steps,
            "testing_results": self.testing_results,
            "vulnerability_info": self.vulnerability_info,
            "error": self.error,
        }


def _failing_tests(testing_results: dict | None) -> list[dict]:
    """Pull the failing-test list out of vul4j's testing_results.json shape.

    Vul4J writes ``{"tests": {"overall": {...}, "tests": {"failing_tests": [...], "error_tests": [...]}}}``.
    Treat error_tests as failures too — both block a "verified" verdict.
    """
    if not testing_results:
        return []
    tests = (testing_results.get("tests") or {}).get("tests") or {}
    return list(tests.get("failing_tests") or []) + list(tests.get("error_tests") or [])


def verify(vuln_id: str, patched_code: str) -> tuple[Verdict, dict]:
    """Run the full compile + PoV + regression pipeline on ``patched_code``.

    Sequence:
      1. checkout {vuln_id} into a tempdir
      2. write patched_code over the project's modified target file
      3. vul4j compile           -> compile_error on failure
      4. vul4j test -b povs      -> still_vulnerable if any PoV fails
      5. vul4j test              -> regression_broken if any non-PoV fails
      6. verified
    """
    if not vr.docker_available():
        raise RuntimeError("docker not found on PATH — verifier needs Docker to run Vul4J")

    logs = VerifyLogs(vuln_id=vuln_id)

    with vr.ephemeral_workdir() as workdir:
        co = vr.checkout(workdir, vuln_id)
        logs.add("checkout", co)
        if not co.ok:
            logs.error = "checkout failed"
            return "compile_error", logs.to_dict()

        info = vr.read_vulnerability_info(workdir)
        logs.vulnerability_info = info
        if not info:
            logs.error = "vulnerability_info.json missing after checkout"
            return "compile_error", logs.to_dict()

        try:
            vr.write_patched_file(workdir, info, patched_code)
        except Exception as e:
            logs.error = f"could not write patched file: {e}"
            return "compile_error", logs.to_dict()

        comp = vr.compile_(workdir)
        logs.add("compile", comp)
        if not comp.ok:
            return "compile_error", logs.to_dict()

        pov = vr.test(workdir, batch_type="povs")
        logs.add("test_povs", pov)
        pov_results = vr.read_testing_results(workdir)
        if _failing_tests(pov_results):
            logs.testing_results = pov_results
            return "still_vulnerable", logs.to_dict()

        full = vr.test(workdir)  # PoV + regression
        logs.add("test_all", full)
        full_results = vr.read_testing_results(workdir)
        logs.testing_results = full_results

        failing = _failing_tests(full_results)
        # A PoV failure here would mean the PoV-only step disagreed with the
        # full run — surface that as still_vulnerable to be safe.
        pov_test_ids = set((info.get("tests") or {}).get("pov_tests") or [])

        def is_pov(t: dict) -> bool:
            if not pov_test_ids:
                return False
            cls = t.get("test_class") or ""
            method = t.get("test_method") or ""
            return cls in pov_test_ids or f"{cls}#{method}" in pov_test_ids

        if any(is_pov(t) for t in failing):
            return "still_vulnerable", logs.to_dict()
        if failing:
            return "regression_broken", logs.to_dict()
        return "verified", logs.to_dict()


# ---------------------------------------------------------------------------
# CLI: `python -m verifier.verify --smoke`
# ---------------------------------------------------------------------------

def _main() -> int:
    parser = argparse.ArgumentParser(description="PatchProof verifier")
    parser.add_argument("--smoke", action="store_true", help="run the smoke test")
    parser.add_argument("--id", help="verify a single vuln_id, reading patched code from --file")
    parser.add_argument("--file", help="path to the candidate patched file")
    args = parser.parse_args()

    if args.smoke:
        from .smoke_test import run_smoke
        return run_smoke()

    if args.id and args.file:
        code = Path(args.file).read_text()
        verdict, logs = verify(args.id, code)
        print(f"verdict: {verdict}")
        if logs.get("error"):
            print(f"error:   {logs['error']}", file=sys.stderr)
        return 0 if verdict == "verified" else 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(_main())
