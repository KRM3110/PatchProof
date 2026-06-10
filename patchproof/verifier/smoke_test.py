"""Smoke test for the verifier — Step 1 DoD.

For each vuln_id in SMOKE_IDS:

  * Build the *vulnerable* version of the project (vul4j checkout +
    `vul4j apply -v vulnerable`), read the target file's content, hand it
    to verify(). Must come back as ``still_vulnerable``.

  * Build the *human-patched* version of the project (vul4j checkout +
    `vul4j apply -v human_patch`), read the target file's content, hand it
    to verify(). Must come back as ``verified``.

Each direction uses its own ephemeral workdir so the two runs cannot interfere.

Run from the repo root:

    python -m verifier.verify --smoke

Notes
-----
* Requires Docker (the verifier shells out to ``tuhhsse/vul4j:alldeps``).
* Override the image with ``VUL4J_IMAGE`` if you've built your own.
* SMOKE_IDS currently has VUL4J-10 plus four placeholders — fill those in
  before the first run, picking them from the llm-vul known-good list.
"""

from __future__ import annotations

import sys
import traceback

from . import vul4j_runner as vr
from .verify import verify


# VUL4J-10 is the canonical sanity-check id; the four "VUL4J-XX" entries are
# placeholders. Replace them with ids you've confirmed are known-good in
# llm-vul before running the smoke test for real.
SMOKE_IDS: list[str] = [
    "VUL4J-10",  # commons-fileupload, CVE-2013-2186, CWE-20
    "VUL4J-6",   # commons-compress, CVE-2018-1324, CWE-835
    "VUL4J-8",   # commons-compress, CVE-2019-12402, CWE-835
    "VUL4J-12",  # commons-imaging, CVE-2018-17202, CWE-835
    "VUL4J-13",  # commons-imaging, CVE-2018-17201, CWE-835
]


def _materialize_version(vuln_id: str, version: str) -> str:
    """Return the *full file contents* of the patch-target for ``version``.

    We rely on Vul4J itself to produce the reference file — that way the smoke
    test stays honest even if llm-vul reorganizes their layout. ``version`` is
    passed through to ``vul4j apply -v`` so values like ``vulnerable`` and
    ``human_patch`` work directly.
    """
    with vr.ephemeral_workdir(prefix=f"vul4j-{version}-") as workdir:
        co = vr.checkout(workdir, vuln_id)
        if not co.ok:
            raise RuntimeError(f"checkout failed for {vuln_id}:\n{co.render()}")
        ap = vr.apply(workdir, version)
        if not ap.ok:
            raise RuntimeError(f"apply {version!r} failed for {vuln_id}:\n{ap.render()}")
        info = vr.read_vulnerability_info(workdir)
        if not info:
            raise RuntimeError(f"vulnerability_info.json missing for {vuln_id}")
        target = vr.target_file_path(workdir, info)
        if not target.exists():
            raise RuntimeError(f"target file {target} does not exist after apply")
        return target.read_text()


def _check(vuln_id: str, version: str, expected: str) -> tuple[bool, str]:
    try:
        code = _materialize_version(vuln_id, version)
        verdict, _logs = verify(vuln_id, code)
    except Exception as e:
        return False, f"raised: {e}\n{traceback.format_exc()}"
    ok = verdict == expected
    return ok, f"verdict={verdict} expected={expected}"


def run_smoke() -> int:
    if not vr.docker_available():
        print("FAIL: docker not on PATH — install Docker or run from a host that has it",
              file=sys.stderr)
        return 2

    unresolved = [i for i in SMOKE_IDS if i == "VUL4J-XX"]
    if unresolved:
        print(
            f"WARN: {len(unresolved)} placeholder id(s) in SMOKE_IDS — "
            "fill them in before relying on this smoke test.",
            file=sys.stderr,
        )

    passes = 0
    fails: list[str] = []
    total = 0
    for vuln_id in SMOKE_IDS:
        if vuln_id == "VUL4J-XX":
            continue
        for version, expected in (("human_patch", "verified"), ("vulnerable", "still_vulnerable")):
            total += 1
            label = f"{vuln_id} [{version}]"
            print(f"... {label}", flush=True)
            ok, detail = _check(vuln_id, version, expected)
            if ok:
                passes += 1
                print(f"  PASS  {detail}")
            else:
                fails.append(f"{label}: {detail}")
                print(f"  FAIL  {detail}")

    print()
    print(f"smoke: {passes}/{total} checks passed")
    for f in fails:
        print(f"  - {f}")
    return 0 if not fails and total > 0 else 1
