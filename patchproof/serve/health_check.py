"""One-shot health check for the vLLM Patcher endpoint.

Sends ONE formatted request through ``agents.model_client.VLLMModelClient`` —
the same client the agent loop uses and the same chat template used at
training — then prints the returned fix. Non-zero exit on failure so it can
gate the runbook.

Usage (from repo root):

    python -m serve.health_check
    python -m serve.health_check --base-url http://127.0.0.1:8000/v1
    python -m serve.health_check --model patchproof-merged
"""

from __future__ import annotations

import argparse
import sys

from agents.model_client import ModelClientError, VLLMModelClient


VULN_SNIPPET = """\
public User findUser(String name) {
    String q = "SELECT * FROM users WHERE name = '" + name + "'";
    return jdbc.queryForObject(q, userRowMapper);
}
"""


def main() -> int:
    p = argparse.ArgumentParser(description="vLLM Patcher endpoint health check.")
    p.add_argument("--base-url", default=None,
                   help="override PATCHPROOF_VLLM_BASE_URL")
    p.add_argument("--model", default=None,
                   help="override PATCHPROOF_VLLM_MODEL (the --served-model-name)")
    p.add_argument("--api-key", default=None,
                   help="override PATCHPROOF_VLLM_API_KEY")
    args = p.parse_args()

    kwargs = {}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if args.model:
        kwargs["model"] = args.model
    if args.api_key:
        kwargs["api_key"] = args.api_key

    try:
        client = VLLMModelClient(**kwargs)
        out = client.generate_patch(
            code=VULN_SNIPPET,
            cwe="CWE-89",
            triage_report="Probable SQL injection via string concatenation.",
            repair_feedback=None,
        )
    except ModelClientError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    if not out.strip():
        print("FAIL: endpoint returned empty patch", file=sys.stderr)
        return 1

    print(f"OK: {client.base_url} model={client.model!r} returned a fix:")
    print("-" * 60)
    print(out)
    print("-" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
