"""Step 6 — base vs fine-tuned vs +agent-loop verified-fix-rate on held-out Vul4J.

Honest comparison rule
----------------------
``base`` is the SAME ``Qwen2.5-Coder-7B-Instruct`` we fine-tuned
(``config.MODEL_ID``), prompt-only via vLLM — NOT a frontier API. The whole
point of Slide 4 is "what did the fine-tune buy us?" — that question only has
meaning against the same base model.

Modes (one process per mode so two models needn't be served at once):

    --mode base         single-shot against the base-model endpoint
    --mode tuned        single-shot against the fine-tuned endpoint
    --mode tuned_loop   full agent loop (triage -> patch -> verify -> self-heal)
                        against the fine-tuned endpoint, retry budget honored
    --mode plot         read outputs/benchmark_results.json, emit chart + table
    --mode throughput   tokens/sec on a batch (FP16 vs FP8 optimization slide)

Eval set = the held-out VUL4J ids loaded from ``config.load_eval_vul4j_ids``.
Those are also held out by the dataset builder (data/held_out_cves.txt) so
nothing in this set was seen at training time.

Each per-id record is appended to ``outputs/benchmark_results.json``, so
running all three modes builds the full Slide-4 dataset incrementally.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from config import MODEL_ID, build_messages, load_eval_vul4j_ids


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RESULTS = "outputs/benchmark_results.json"
DEFAULT_CHART = "outputs/benchmark_chart.png"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TIMEOUT = 180.0
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024

VALID_MODES = ("base", "tuned", "tuned_loop", "plot", "throughput")

_FENCED_JAVA = re.compile(r"```java\s*\n(.*?)```", re.DOTALL)
_FENCED_ANY = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.DOTALL)
_CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "benchmark.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger("patchproof.bench")
    logger.info("logging to %s", log_path)
    return logger


# ---------------------------------------------------------------------------
# Per-id result + persistence
# ---------------------------------------------------------------------------

@dataclass
class IdResult:
    mode: str
    vuln_id: str
    verdict: str | None
    tokens_in: int
    tokens_out: int
    latency_seconds: float
    attempts: int
    endpoint: str
    model: str
    error: str | None = None


def _append_results(path: Path, records: list[dict]) -> None:
    """Merge into outputs/benchmark_results.json. Running modes one at a time
    must not clobber prior runs — they're additive evidence for the slide."""
    existing: list[dict] = []
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            existing = list(payload.get("results", []))
        except json.JSONDecodeError:
            # Don't silently overwrite a corrupt-looking file; back it up first.
            backup = path.with_suffix(path.suffix + ".bak")
            path.rename(backup)
            logging.getLogger("patchproof.bench").warning(
                "results file was unparseable; moved to %s", backup,
            )
    existing.extend(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": existing}, indent=2))


# ---------------------------------------------------------------------------
# Vul4J fixture: materialize the vulnerable version of an eval id
# ---------------------------------------------------------------------------

def _materialize_vulnerable(vuln_id: str) -> tuple[str, str]:
    """Return ``(vulnerable_source_text, cwe)`` for a Vul4J id.

    Mirrors verifier.smoke_test._materialize_version: ``vul4j checkout``,
    ``vul4j apply -v vulnerable``, then read the target file's contents.
    The CWE comes from the project's vulnerability_info.json so the prompt's
    label matches what the verifier actually treats as the vulnerability.
    """
    from verifier import vul4j_runner as vr  # local import keeps plot mode dep-free

    if not vr.docker_available():
        raise RuntimeError("docker not on PATH — benchmark needs the real verifier")

    with vr.ephemeral_workdir(prefix=f"bench-{vuln_id}-") as workdir:
        co = vr.checkout(workdir, vuln_id)
        if not co.ok:
            raise RuntimeError(f"checkout failed for {vuln_id}:\n{co.render()}")
        ap = vr.apply(workdir, "vulnerable")
        if not ap.ok:
            raise RuntimeError(f"apply 'vulnerable' failed for {vuln_id}:\n{ap.render()}")
        info = vr.read_vulnerability_info(workdir)
        if not info:
            raise RuntimeError(f"vulnerability_info.json missing for {vuln_id}")
        target = vr.target_file_path(workdir, info)
        if not target.exists():
            raise RuntimeError(f"target file {target} does not exist after apply")
        code = target.read_text()
        cwe = str(info.get("cwe_id") or info.get("cwe") or "CWE-UNKNOWN")
    return code, cwe


# ---------------------------------------------------------------------------
# OpenAI client + helpers
# ---------------------------------------------------------------------------

def _make_openai(base_url: str, api_key: str, timeout: float):
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "openai client not installed. `pip install openai`."
        ) from e
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _extract_java(text: str) -> str:
    m = _FENCED_JAVA.search(text)
    if m:
        return m.group(1).rstrip()
    m = _FENCED_ANY.search(text)
    if m:
        return m.group(1).rstrip()
    return text.strip()


# ---------------------------------------------------------------------------
# Single-shot path (base + tuned)
# ---------------------------------------------------------------------------

def _single_shot(client, model: str, cwe: str, code: str,
                 temperature: float, max_tokens: int) -> tuple[str, int, int]:
    """One chat completion using the shared chat template. Returns
    ``(patch_text, prompt_tokens, completion_tokens)``."""
    msgs = build_messages(cwe, code)
    resp = client.chat.completions.create(
        model=model, messages=msgs,
        temperature=temperature, max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    pin = int(getattr(usage, "prompt_tokens", 0) or 0)
    pout = int(getattr(usage, "completion_tokens", 0) or 0)
    return _extract_java(text), pin, pout


def _run_single_shot_mode(
    *,
    mode: str, ids: list[str], endpoint: str, model: str, api_key: str,
    temperature: float, max_tokens: int, timeout: float,
    logger: logging.Logger,
) -> list[IdResult]:
    from verifier.verify import verify

    client = _make_openai(endpoint, api_key, timeout)
    out: list[IdResult] = []
    for vuln_id in ids:
        t0 = time.perf_counter()
        verdict: str | None = None
        pin = pout = 0
        err: str | None = None
        try:
            code, cwe = _materialize_vulnerable(vuln_id)
            patch_text, pin, pout = _single_shot(
                client, model, cwe, code, temperature, max_tokens,
            )
            verdict, _logs = verify(vuln_id, patch_text)
        except Exception as e:
            logger.exception("id %s failed: %s", vuln_id, e)
            err = str(e)
        elapsed = time.perf_counter() - t0
        rec = IdResult(
            mode=mode, vuln_id=vuln_id, verdict=verdict,
            tokens_in=pin, tokens_out=pout, latency_seconds=elapsed,
            attempts=1, endpoint=endpoint, model=model, error=err,
        )
        out.append(rec)
        logger.info("[%s] %s -> %s (%.1fs, in=%d out=%d)",
                    mode, vuln_id, verdict, elapsed, pin, pout)
    return out


# ---------------------------------------------------------------------------
# Tuned-loop path: counting shim + real model/LLM clients
# ---------------------------------------------------------------------------

class _CountingClientShim:
    """Wraps an OpenAI client and accumulates ``prompt_tokens`` and
    ``completion_tokens`` across every chat call. All three agent roles
    (Patcher, Triage/Self-Heal/Reporter LLM) funnel through one shim, so the
    per-id totals reflect the WHOLE loop, not just the model_client step."""

    def __init__(self, client, model: str):
        self._client = client
        self._model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, messages: list[dict], *, temperature: float, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        usage = getattr(resp, "usage", None)
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        return resp.choices[0].message.content or ""

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0


class _LoopModelClient:
    """Patcher model client matching ``StubModelClient.generate_patch``,
    routing through the shared counting shim. Mirrors the prompt shape used
    by ``agents.model_client.VLLMModelClient`` (system + user(CWE, vulnerable)
    + optional triage / repair extras appended to the user turn) so the
    loop's behavior here matches the loop's behavior at serve time."""

    def __init__(self, shim: _CountingClientShim, temperature: float, max_tokens: int):
        self._shim = shim
        self._t = temperature
        self._n = max_tokens

    def generate_patch(self, *, code: str, cwe: str | None,
                       triage_report: str | None, repair_feedback: str | None) -> str:
        msgs = build_messages(cwe or "CWE-UNKNOWN", code)
        extras: list[str] = []
        if triage_report:
            extras.append(f"Triage report:\n{triage_report}")
        if repair_feedback:
            extras.append(
                "Previous attempt failed verification. Repair feedback:\n"
                f"{repair_feedback}"
            )
        if extras:
            msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + "\n\n".join(extras)
        return _extract_java(self._shim.chat(msgs, temperature=self._t, max_tokens=self._n))


class _LoopLLMClient:
    """Minimal real LLM client implementing ``triage`` / ``diagnose`` /
    ``explain`` over the same vLLM endpoint. The tuned model wears all three
    hats for the benchmark — separating them out into a hosted frontier
    model would be a different (and weaker) honest-comparison story."""

    _TRIAGE_SYS = (
        "You are a Java security triage assistant. Given a vulnerable Java method, "
        "identify the most likely CWE (one of the OWASP top common ones — answer in "
        "the form 'CWE-89') and write a 1-2 sentence report of the vulnerability."
    )
    _DIAGNOSE_SYS = (
        "You are debugging a Java security patch. Given the verifier verdict and "
        "logs from a failed patch attempt, write 2-3 sentences of repair feedback "
        "that the next patcher attempt can use to actually fix the issue."
    )
    _EXPLAIN_SYS = (
        "You are documenting a security patch outcome. Given the final state of a "
        "patch attempt, write a 2-3 sentence human-readable explanation of what was "
        "fixed (or why no fix landed)."
    )

    def __init__(self, shim: _CountingClientShim, temperature: float, max_tokens: int):
        self._shim = shim
        self._t = temperature
        self._n = max_tokens

    def _call(self, system: str, user: str) -> str:
        return self._shim.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=self._t, max_tokens=self._n,
        )

    def triage(self, code: str) -> dict:
        text = self._call(self._TRIAGE_SYS, f"Vulnerable method:\n```java\n{code}\n```")
        m = _CWE_RE.search(text)
        cwe = m.group(0).upper() if m else "CWE-UNKNOWN"
        return {"cwe": cwe, "report": text.strip()}

    def diagnose(self, *, verdict: str, logs: dict) -> str:
        # Logs can be large (full stdout/stderr trees) — cap so triage doesn't
        # blow past the context budget on the regression-suite log dump.
        logs_excerpt = json.dumps(logs, indent=2)[:4000]
        prompt = f"Verdict: {verdict}\n\nLogs (truncated to 4 KB):\n{logs_excerpt}"
        return self._call(self._DIAGNOSE_SYS, prompt).strip()

    def explain(self, state: dict) -> str:
        prompt = (
            f"Verdict: {state.get('verdict')}\n"
            f"CWE: {state.get('cwe')}\n"
            f"Attempts: {state.get('attempt')}\n\n"
            f"Final patch:\n```java\n{state.get('candidate_patch', '')}\n```"
        )
        return self._call(self._EXPLAIN_SYS, prompt).strip()


def _run_tuned_loop_mode(
    *,
    ids: list[str], endpoint: str, model: str, api_key: str,
    temperature: float, max_tokens: int, max_attempts: int, timeout: float,
    logger: logging.Logger,
) -> list[IdResult]:
    from agents.graph import build_graph
    from agents.state import initial_state
    from verifier.verify import verify

    client = _make_openai(endpoint, api_key, timeout)
    shim = _CountingClientShim(client, model)
    model_client = _LoopModelClient(shim, temperature, max_tokens)
    llm_client = _LoopLLMClient(shim, temperature, max_tokens)
    graph = build_graph(
        llm_client=llm_client, model_client=model_client, verifier_fn=verify,
    )

    out: list[IdResult] = []
    for vuln_id in ids:
        shim.reset()
        t0 = time.perf_counter()
        verdict: str | None = None
        attempts = 0
        err: str | None = None
        try:
            code, _cwe = _materialize_vulnerable(vuln_id)
            final = graph.invoke(
                initial_state(vuln_id=vuln_id, code=code, max_attempts=max_attempts),
            )
            verdict = final.get("verdict")
            attempts = int(final.get("attempt") or 0)
        except Exception as e:
            logger.exception("id %s failed: %s", vuln_id, e)
            err = str(e)
        elapsed = time.perf_counter() - t0
        rec = IdResult(
            mode="tuned_loop", vuln_id=vuln_id, verdict=verdict,
            tokens_in=shim.prompt_tokens, tokens_out=shim.completion_tokens,
            latency_seconds=elapsed, attempts=attempts,
            endpoint=endpoint, model=model, error=err,
        )
        out.append(rec)
        logger.info(
            "[tuned_loop] %s -> %s in %d attempt(s) (%.1fs, in=%d out=%d)",
            vuln_id, verdict, attempts, elapsed,
            shim.prompt_tokens, shim.completion_tokens,
        )
    return out


# ---------------------------------------------------------------------------
# Throughput (optional, for the FP16 vs FP8 optimization slide)
# ---------------------------------------------------------------------------

_THROUGHPUT_SNIPPET = (
    "public User findUser(String name) {\n"
    "    String q = \"SELECT * FROM users WHERE name = '\" + name + \"'\";\n"
    "    return jdbc.queryForObject(q, userRowMapper);\n"
    "}\n"
)


def _run_throughput(
    *,
    endpoint: str, model: str, api_key: str, label: str,
    batch_size: int, max_tokens: int, timeout: float,
    logger: logging.Logger,
) -> dict:
    client = _make_openai(endpoint, api_key, timeout)
    msgs = build_messages("CWE-89", _THROUGHPUT_SNIPPET)

    def one(_i: int) -> tuple[int, int]:
        resp = client.chat.completions.create(
            model=model, messages=msgs, temperature=0.0, max_tokens=max_tokens,
        )
        usage = getattr(resp, "usage", None)
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch_size) as ex:
        results = list(ex.map(one, range(batch_size)))
    elapsed = time.perf_counter() - t0
    tin = sum(r[0] for r in results)
    tout = sum(r[1] for r in results)
    out_per_sec = tout / elapsed if elapsed > 0 else 0.0
    rec = {
        "mode": "throughput", "label": label, "endpoint": endpoint, "model": model,
        "batch_size": batch_size, "max_tokens": max_tokens,
        "tokens_in_total": tin, "tokens_out_total": tout,
        "wall_clock_seconds": elapsed,
        "completion_tokens_per_second": out_per_sec,
    }
    logger.info(
        "[throughput] %s: batch=%d -> %.1f out-tok/s, wall=%.1fs",
        label, batch_size, out_per_sec, elapsed,
    )
    return rec


# ---------------------------------------------------------------------------
# Plot + summary
# ---------------------------------------------------------------------------

def _plot(results_path: Path, chart_path: Path, logger: logging.Logger) -> int:
    if not results_path.exists():
        logger.error("results file not found: %s — run a benchmark mode first", results_path)
        return 1
    payload = json.loads(results_path.read_text())
    rows = [r for r in payload.get("results", [])
            if r.get("mode") in ("base", "tuned", "tuned_loop")]
    if not rows:
        logger.error("no base/tuned/tuned_loop rows in %s", results_path)
        return 1

    rates: dict[str, list[Any]] = {}   # mode -> [verified_count, total, sum_lat, sum_in, sum_out]
    for r in rows:
        mode = r["mode"]
        bucket = rates.setdefault(mode, [0, 0, 0.0, 0, 0])
        bucket[0] += 1 if r.get("verdict") == "verified" else 0
        bucket[1] += 1
        bucket[2] += float(r.get("latency_seconds") or 0.0)
        bucket[3] += int(r.get("tokens_in") or 0)
        bucket[4] += int(r.get("tokens_out") or 0)

    order = ["base", "tuned", "tuned_loop"]
    labels = [m for m in order if m in rates]
    fracs = [rates[m][0] / rates[m][1] for m in labels]

    print()
    print(f"{'mode':<12} {'verified':>9} {'total':>6} {'rate':>7} {'lat/id':>9} {'in/id':>8} {'out/id':>8}")
    for m in labels:
        v, n, lat, tin, tout = rates[m]
        print(f"{m:<12} {v:>9} {n:>6} {v / n:>7.1%} {lat / n:>8.1f}s {tin // n:>8d} {tout // n:>8d}")
    print()

    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        logger.warning("matplotlib unavailable (%s) — printed summary, no PNG", e)
        return 0

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, fracs)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("verified-fix rate")
    ax.set_title("PatchProof — verified-fix rate on held-out Vul4J ids")
    for b, f in zip(bars, fracs):
        ax.text(b.get_x() + b.get_width() / 2, f + 0.02,
                f"{f:.0%}", ha="center", va="bottom")
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    logger.info("wrote chart -> %s", chart_path)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_ids(ids_arg: str | None, logger: logging.Logger) -> list[str]:
    if ids_arg:
        return [s.strip() for s in ids_arg.split(",") if s.strip()]
    ids = load_eval_vul4j_ids()
    if not ids:
        logger.error(
            "eval id list is empty — populate %s or pass --ids VUL4J-10,VUL4J-12,...",
            "data/eval_vul4j_ids.txt",
        )
        sys.exit(2)
    return ids


def main() -> int:
    p = argparse.ArgumentParser(description="PatchProof Step 6 — Slide-4 benchmark.")
    p.add_argument("--mode", required=True, choices=VALID_MODES)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                   help=f"vLLM OpenAI base URL (default {DEFAULT_ENDPOINT})")
    p.add_argument("--model", default=MODEL_ID,
                   help="served-model-name. Default = config.MODEL_ID (the HF id, "
                        "appropriate for the base-model serve). For tuned/tuned_loop "
                        "pass whatever --served-name you gave serve.sh, e.g. "
                        "'patchproof-merged'.")
    p.add_argument("--api-key", default=os.environ.get("PATCHPROOF_VLLM_API_KEY", "EMPTY"))
    p.add_argument("--ids", default=None,
                   help="optional comma-separated VUL4J id override; defaults to "
                        "config.load_eval_vul4j_ids()")
    p.add_argument("--results-out", default=DEFAULT_RESULTS)
    p.add_argument("--chart-out", default=DEFAULT_CHART)
    p.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    # throughput-only knobs
    p.add_argument("--label", default="fp16",
                   help="throughput-mode label (e.g. fp16 / fp8)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="throughput-mode concurrent request count")
    args = p.parse_args()

    out_path = Path(args.results_out)
    log_dir = out_path.parent if str(out_path.parent) else Path("outputs")
    logger = _setup_logging(log_dir)
    logger.info("== benchmark mode=%s endpoint=%s model=%s ==",
                args.mode, args.endpoint, args.model)

    if args.mode == "plot":
        return _plot(out_path, Path(args.chart_out), logger)

    if args.mode == "throughput":
        rec = _run_throughput(
            endpoint=args.endpoint, model=args.model, api_key=args.api_key,
            label=args.label, batch_size=args.batch_size,
            max_tokens=args.max_tokens, timeout=args.timeout, logger=logger,
        )
        _append_results(out_path, [rec])
        return 0

    ids = _resolve_ids(args.ids, logger)
    logger.info("eval ids (%d): %s", len(ids), ", ".join(ids))

    if args.mode in ("base", "tuned"):
        results = _run_single_shot_mode(
            mode=args.mode, ids=ids, endpoint=args.endpoint, model=args.model,
            api_key=args.api_key, temperature=args.temperature,
            max_tokens=args.max_tokens, timeout=args.timeout, logger=logger,
        )
    else:
        results = _run_tuned_loop_mode(
            ids=ids, endpoint=args.endpoint, model=args.model, api_key=args.api_key,
            temperature=args.temperature, max_tokens=args.max_tokens,
            max_attempts=args.max_attempts, timeout=args.timeout, logger=logger,
        )

    _append_results(out_path, [asdict(r) for r in results])
    verified = sum(1 for r in results if r.verdict == "verified")
    logger.info("== %s done: %d/%d verified ==", args.mode, verified, len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
