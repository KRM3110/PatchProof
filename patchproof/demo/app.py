"""Step 7 — PatchProof demo UI (Streamlit). Author-time only; do not run.

Two views (sidebar selector):

  1. "Single vulnerability" — the attack-and-neutralize money shot:
       show vulnerable code -> Triage labels the CWE
       -> verifier on ORIGINAL  (RED: vulnerability confirmed, PoV failing)
       -> agent loop (triage + patch + verify + self-heal)
       -> diff between original and candidate fix
       -> verifier on FIX       (GREEN: PoV passes + regression passes -> verified)
       -> evidence panel (failing test names before, passing names after)

  2. "Batch wall" — the scale shot: read outputs/benchmark_results.json (the
     REAL Step-6 results) and render a grid of all eval ids per mode, animate
     RED -> GREEN, end on the headline verified-fix rate per mode.

  3. (optional) "Side-by-side" — base model's fix stays RED while the tuned
     model's goes GREEN for the same vuln. Replay-only: needs paired caches.

Live vs replay
--------------
Replay is the recording default. A live Maven compile+test takes tens of
seconds and can flake on camera; we don't want either during the screen-record.

Workflow:
  * On the box (once):
        streamlit run demo/app.py -- --live --save-cache --vuln-id VUL4J-10
    Click "Run live attempt". The result lands in ``demo/cache/<id>.json``.
    Commit it. Now the recording works without vLLM or Docker.

  * For the recording:
        streamlit run demo/app.py
    Replay mode is the default; the cached attempt renders instantly.

Args after ``--`` reach this script via sys.argv. Streamlit-specific args
appear before ``--`` and are filtered out by argparse below.

Nothing in this file talks to an LLM as a judge. The verifier is the only
source of truth — same rule as the rest of the project.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_RESULTS = "outputs/benchmark_results.json"
DEFAULT_CACHE_DIR = "demo/cache"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_BASE_ENDPOINT = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "patchproof-merged"
DEFAULT_BASE_MODEL = "Qwen2.5-Coder-7B-Instruct"
DEFAULT_API_KEY = os.environ.get("PATCHPROOF_VLLM_API_KEY", "EMPTY")
DEFAULT_MAX_ATTEMPTS = 3

VERDICT_GREEN = "verified"
VERDICT_REDS = {"compile_error", "still_vulnerable", "regression_broken"}


@dataclass
class CliConfig:
    live: bool = False
    save_cache: bool = False
    vuln_id: str | None = None
    results_path: str = DEFAULT_RESULTS
    cache_dir: str = DEFAULT_CACHE_DIR
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    api_key: str = DEFAULT_API_KEY
    base_endpoint: str = DEFAULT_BASE_ENDPOINT
    base_model: str = DEFAULT_BASE_MODEL
    max_attempts: int = DEFAULT_MAX_ATTEMPTS


def _parse_cli() -> CliConfig:
    """Parse args passed after ``streamlit run demo/app.py --``.

    Streamlit's own flags appear before the ``--`` so they aren't in sys.argv
    here. We still fall back gracefully if argparse trips on something unknown
    — the UI must always come up.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--live", action="store_true",
                   help="enable live runs (real graph + verifier). Replay is the default.")
    p.add_argument("--replay", action="store_true",
                   help="(default) render from cached results only")
    p.add_argument("--save-cache", action="store_true",
                   help="in live mode, expose the 'Save to cache' button so a live run "
                        "can be frozen into demo/cache/<id>.json for replay")
    p.add_argument("--vuln-id", default=None,
                   help="preselect a VUL4J id in the Single Vulnerability view")
    p.add_argument("--results", default=DEFAULT_RESULTS,
                   help=f"path to benchmark_results.json (default {DEFAULT_RESULTS})")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                   help=f"directory of per-id cache files (default {DEFAULT_CACHE_DIR})")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--base-endpoint", default=DEFAULT_BASE_ENDPOINT)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)

    try:
        args, _unknown = p.parse_known_args(sys.argv[1:])
    except SystemExit:
        # argparse called sys.exit on --help or a bad arg; degrade to defaults
        # so `streamlit run` never hangs on a parser error.
        args = argparse.Namespace(
            live=False, replay=True, save_cache=False, vuln_id=None,
            results=DEFAULT_RESULTS, cache_dir=DEFAULT_CACHE_DIR,
            endpoint=DEFAULT_ENDPOINT, model=DEFAULT_MODEL, api_key=DEFAULT_API_KEY,
            base_endpoint=DEFAULT_BASE_ENDPOINT, base_model=DEFAULT_BASE_MODEL,
            max_attempts=DEFAULT_MAX_ATTEMPTS,
        )

    return CliConfig(
        live=bool(args.live) and not bool(args.replay),
        save_cache=bool(args.save_cache),
        vuln_id=args.vuln_id,
        results_path=args.results,
        cache_dir=args.cache_dir,
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        base_endpoint=args.base_endpoint,
        base_model=args.base_model,
        max_attempts=int(args.max_attempts),
    )


# ---------------------------------------------------------------------------
# Cache (per-id JSON for the Single Vulnerability replay)
# ---------------------------------------------------------------------------

# Schema (frozen here, not imported anywhere — keep one source of truth):
#   {
#     "vuln_id":       "VUL4J-10",
#     "cwe":           "CWE-20",
#     "triage_report": "<text>",
#     "original_code": "<text>",
#     "vulnerable_verdict": "still_vulnerable",
#     "vulnerable_failing_tests": ["org.foo.Test#bar", ...],
#     "vulnerable_logs": {...verifier logs...},
#     "candidate_patch": "<text>",
#     "fixed_verdict": "verified",
#     "fixed_failing_tests": [],
#     "fixed_passing_summary": {"passed": 24, "failed": 0, "errors": 0},
#     "fixed_logs": {...verifier logs...},
#     "attempts": 2,
#     "explanation": "<text>"
#   }

def _cache_path(cache_dir: str, vuln_id: str) -> Path:
    safe = vuln_id.replace("/", "_")
    return Path(cache_dir) / f"{safe}.json"


def _load_cache(cache_dir: str, vuln_id: str) -> dict | None:
    p = _cache_path(cache_dir, vuln_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _save_cache(cache_dir: str, vuln_id: str, payload: dict) -> Path:
    p = _cache_path(cache_dir, vuln_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    return p


def _list_cached_ids(cache_dir: str) -> list[str]:
    d = Path(cache_dir)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


# ---------------------------------------------------------------------------
# Verifier-log helpers (deterministic; no LLM here)
# ---------------------------------------------------------------------------

def _testing_block(logs: dict | None) -> dict:
    """Pull the inner testing_results.tests.tests block (where Vul4J writes
    failing_tests / error_tests / passing_tests). Returns {} if not present."""
    if not logs:
        return {}
    tr = logs.get("testing_results") or {}
    tests = (tr.get("tests") or {}).get("tests") or {}
    return tests if isinstance(tests, dict) else {}


def _summary(logs: dict | None) -> dict:
    """Pull the {passing_tests, failing_tests, error_tests} overall counts."""
    if not logs:
        return {}
    tr = logs.get("testing_results") or {}
    overall = (tr.get("tests") or {}).get("overall") or {}
    return overall if isinstance(overall, dict) else {}


def _failing_test_ids(logs: dict | None) -> list[str]:
    tests = _testing_block(logs)
    failing = list(tests.get("failing_tests") or []) + list(tests.get("error_tests") or [])
    out: list[str] = []
    for t in failing:
        if isinstance(t, dict):
            cls = t.get("test_class") or ""
            method = t.get("test_method") or ""
            out.append(f"{cls}#{method}" if method else cls)
        elif isinstance(t, str):
            out.append(t)
    return out


def _passing_summary(logs: dict | None) -> dict:
    """A small {'passed': N, 'failed': N, 'errors': N} summary for the GREEN panel."""
    overall = _summary(logs)
    return {
        "passed": int(overall.get("number_passing", 0) or 0),
        "failed": int(overall.get("number_failing", 0) or 0),
        "errors": int(overall.get("number_error", 0) or 0),
    }


# ---------------------------------------------------------------------------
# UI primitives
# ---------------------------------------------------------------------------

def _badge(label: str, ok: bool) -> str:
    """Return a small inline HTML pill — colored by verdict polarity."""
    bg = "#0f6b2c" if ok else "#8b1a1a"
    return (
        f'<span style="background:{bg};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.85em;font-weight:600;">{label}</span>'
    )


def _banner_red(text: str) -> None:
    st.markdown(
        f'<div style="background:#8b1a1a;color:white;padding:10px 14px;'
        f'border-radius:6px;font-weight:600;">🛑 {text}</div>',
        unsafe_allow_html=True,
    )


def _banner_green(text: str) -> None:
    st.markdown(
        f'<div style="background:#0f6b2c;color:white;padding:10px 14px;'
        f'border-radius:6px;font-weight:600;">✅ {text}</div>',
        unsafe_allow_html=True,
    )


def _show_diff(original: str, patched: str) -> None:
    """Unified diff between original method and the candidate patch."""
    diff = difflib.unified_diff(
        original.splitlines(keepends=False),
        patched.splitlines(keepends=False),
        fromfile="vulnerable",
        tofile="patched",
        lineterm="",
    )
    text = "\n".join(diff) or "(no textual diff — patch may be identical)"
    st.code(text, language="diff")


# ---------------------------------------------------------------------------
# Live wiring (imported lazily — replay must work with NONE of these installed)
# ---------------------------------------------------------------------------

def _run_live_attempt(cfg: CliConfig, vuln_id: str) -> dict:
    """Materialize, RED-verify, agent-loop, return a cache-shaped payload.

    Imports happen inside this function on purpose — Streamlit replay must boot
    even if the project's GPU-side deps (openai, langgraph, docker) aren't
    importable on the recording laptop.
    """
    from agents.graph import build_graph
    from agents.model_client import VLLMModelClient
    from agents.state import initial_state
    from eval.benchmark import (
        _CountingClientShim, _LoopLLMClient, _make_openai, _materialize_vulnerable,
    )
    from verifier.verify import verify

    code, cwe = _materialize_vulnerable(vuln_id)

    vuln_verdict, vuln_logs = verify(vuln_id, code)

    client = _make_openai(cfg.endpoint, cfg.api_key, timeout=180.0)
    shim = _CountingClientShim(client, cfg.model)
    model_client = VLLMModelClient(
        base_url=cfg.endpoint, model=cfg.model, api_key=cfg.api_key,
    )
    llm_client = _LoopLLMClient(shim, temperature=0.2, max_tokens=1024)
    graph = build_graph(
        llm_client=llm_client, model_client=model_client, verifier_fn=verify,
    )
    final = graph.invoke(
        initial_state(vuln_id=vuln_id, code=code, max_attempts=cfg.max_attempts),
    )

    return {
        "vuln_id": vuln_id,
        "cwe": final.get("cwe") or cwe,
        "triage_report": final.get("triage_report") or "",
        "original_code": code,
        "vulnerable_verdict": vuln_verdict,
        "vulnerable_failing_tests": _failing_test_ids(vuln_logs),
        "vulnerable_logs": vuln_logs,
        "candidate_patch": final.get("candidate_patch") or "",
        "fixed_verdict": final.get("verdict"),
        "fixed_failing_tests": _failing_test_ids(final.get("logs")),
        "fixed_passing_summary": _passing_summary(final.get("logs")),
        "fixed_logs": final.get("logs"),
        "attempts": int(final.get("attempt") or 0),
        "explanation": (final.get("final_result") or {}).get("explanation") or "",
    }


# ---------------------------------------------------------------------------
# View 1: Single vulnerability
# ---------------------------------------------------------------------------

def render_single(cfg: CliConfig) -> None:
    st.subheader("Attack → Patch → Verify")
    st.caption(
        "Real Vul4J project. The verifier is the source of truth — "
        "the green badge means the exploit test actually passes."
    )

    cached_ids = _list_cached_ids(cfg.cache_dir)
    default_id = cfg.vuln_id or (cached_ids[0] if cached_ids else "VUL4J-10")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        vuln_id = st.text_input(
            "VUL4J id",
            value=default_id,
            help="Any Vul4J id with a cache file works in replay; live mode "
                 "needs the verifier + a served model.",
        )
    with col_b:
        if cached_ids:
            st.markdown("**Cached ids:** " + ", ".join(f"`{i}`" for i in cached_ids))
        else:
            st.markdown("_No cached ids yet — populate via --live --save-cache._")

    payload = _load_cache(cfg.cache_dir, vuln_id)

    if cfg.live:
        st.info("Live mode is ON — clicking 'Run live attempt' will hit the "
                "verifier (Docker) and the vLLM endpoint. Tens of seconds.")
        if st.button("▶ Run live attempt", type="primary"):
            with st.spinner("Materializing vulnerable code, then RED-verify, "
                            "then agent loop, then GREEN-verify…"):
                try:
                    payload = _run_live_attempt(cfg, vuln_id)
                    st.session_state["_last_live_payload"] = payload
                    st.session_state["_last_live_vuln_id"] = vuln_id
                    st.success("Live attempt complete.")
                except Exception as e:
                    st.error(f"Live attempt failed: {e}")
                    logging.exception("live attempt failed")
                    return
        elif st.session_state.get("_last_live_vuln_id") == vuln_id:
            payload = st.session_state.get("_last_live_payload") or payload

        if cfg.save_cache and payload is not None and st.button("💾 Save to cache"):
            saved = _save_cache(cfg.cache_dir, vuln_id, payload)
            st.success(f"Saved → {saved}")

    if payload is None:
        st.warning(
            f"No cached result for `{vuln_id}`. "
            "Populate it once with `streamlit run demo/app.py -- "
            f"--live --save-cache --vuln-id {vuln_id}` on the AMD box, "
            "then commit `demo/cache/`."
        )
        return

    # ----- Step 1: vulnerable code -----------------------------------------
    st.markdown("---")
    st.markdown("### 1. Vulnerable method")
    _banner_red(f"Vulnerability under analysis · {payload.get('vuln_id', vuln_id)}")
    st.code(payload.get("original_code", "") or "(empty)", language="java")

    # ----- Step 2: triage --------------------------------------------------
    st.markdown("### 2. Triage")
    cwe = payload.get("cwe") or "CWE-UNKNOWN"
    st.markdown(
        f"**CWE:** `{cwe}` &nbsp; {_badge(cwe, ok=False)}",
        unsafe_allow_html=True,
    )
    if payload.get("triage_report"):
        st.markdown("**Report**")
        st.write(payload["triage_report"])

    # ----- Step 3: verifier on ORIGINAL ------------------------------------
    st.markdown("### 3. Verifier on the ORIGINAL code")
    vv = payload.get("vulnerable_verdict") or "?"
    _banner_red(f"Vulnerability confirmed — verifier returned `{vv}`")
    failing = payload.get("vulnerable_failing_tests") or []
    if failing:
        st.markdown("**Failing tests (the PoV / exploit):**")
        for t in failing:
            st.markdown(f"- `{t}`")
    else:
        st.caption("(no per-test list in the cached logs; see raw logs below)")

    # ----- Step 4: agent loop diff -----------------------------------------
    st.markdown("### 4. Agent loop output — proposed fix")
    attempts = int(payload.get("attempts") or 0)
    st.caption(
        f"Loop converged in **{attempts}** patch attempt(s) "
        f"(triage → patch → verify → self-heal, budget = {cfg.max_attempts})."
    )
    _show_diff(payload.get("original_code", ""), payload.get("candidate_patch", ""))

    # ----- Step 5: verifier on FIX -----------------------------------------
    st.markdown("### 5. Verifier on the FIX")
    fv = payload.get("fixed_verdict") or "?"
    if fv == VERDICT_GREEN:
        _banner_green(
            "Verified — PoV exploit test now passes AND regression suite passes."
        )
    else:
        _banner_red(
            f"Loop did not converge to verified — final verdict `{fv}`. "
            "Showing evidence for transparency."
        )
    summary = payload.get("fixed_passing_summary") or _passing_summary(payload.get("fixed_logs"))
    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("passed", summary.get("passed", 0))
        c2.metric("failed", summary.get("failed", 0))
        c3.metric("errors", summary.get("errors", 0))

    # ----- Step 6: evidence panel ------------------------------------------
    st.markdown("### 6. Evidence")
    if payload.get("explanation"):
        st.write(payload["explanation"])
    with st.expander("Raw verifier logs (vulnerable run)"):
        st.json(payload.get("vulnerable_logs") or {})
    with st.expander("Raw verifier logs (patched run)"):
        st.json(payload.get("fixed_logs") or {})


# ---------------------------------------------------------------------------
# View 2: Batch wall
# ---------------------------------------------------------------------------

def _load_benchmark(results_path: str) -> dict[str, list[dict]]:
    """Group benchmark_results.json by mode. Empty dict if file missing."""
    p = Path(results_path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[str, list[dict]] = {}
    for r in payload.get("results", []):
        mode = r.get("mode")
        if mode in ("base", "tuned", "tuned_loop"):
            out.setdefault(mode, []).append(r)
    return out


def _grid_html(rows: list[dict], reveal_through: int) -> str:
    """Render the grid as one HTML string so the animation can update a single
    container in place instead of redrawing N columns each frame."""
    cells: list[str] = []
    for i, r in enumerate(rows):
        vid = r.get("vuln_id", "?")
        verdict = r.get("verdict") if i < reveal_through else None
        if verdict == VERDICT_GREEN:
            bg, mark = "#0f6b2c", "✓"
        elif verdict in VERDICT_REDS:
            bg, mark = "#8b1a1a", "✗"
        elif verdict is None and i < reveal_through:
            bg, mark = "#5a5a5a", "·"      # ran but no verdict (error)
        else:
            bg, mark = "#262626", "·"      # not yet revealed
        cells.append(
            f'<div style="background:{bg};color:white;padding:8px 6px;'
            f'border-radius:6px;text-align:center;font-family:monospace;'
            f'font-size:0.78em;line-height:1.2;">{vid}<br/>{mark}</div>'
        )
    return (
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,'
        'minmax(110px,1fr));gap:6px;">' + "".join(cells) + "</div>"
    )


def _mode_rate(rows: list[dict]) -> tuple[int, int, float]:
    n = len(rows)
    v = sum(1 for r in rows if r.get("verdict") == VERDICT_GREEN)
    return v, n, (v / n if n else 0.0)


def render_batch(cfg: CliConfig) -> None:
    st.subheader("Batch wall — held-out Vul4J ids")
    st.caption(
        f"Source of truth: `{cfg.results_path}`. "
        "Numbers come from the real verifier (Step 6). "
        "This view only animates them — it doesn't invent anything."
    )

    grouped = _load_benchmark(cfg.results_path)
    if not grouped:
        st.warning(
            f"No usable results at `{cfg.results_path}`. "
            "Run `eval/benchmark.py` (Step 6) first, or point --results "
            "at a known file."
        )
        return

    order = [m for m in ("base", "tuned", "tuned_loop") if m in grouped]
    cols_top = st.columns(len(order))
    for i, mode in enumerate(order):
        v, n, frac = _mode_rate(grouped[mode])
        cols_top[i].metric(f"{mode} · verified-fix rate", f"{frac:.0%}", f"{v}/{n}")

    speed = st.slider(
        "Animation speed (seconds per id)", 0.0, 0.4, 0.06, 0.02,
        help="0 = instant; the recording usually wants ~0.05–0.1s per id."
    )
    skip_anim = st.checkbox("Skip animation (render all at once)", value=False)

    tabs = st.tabs([m for m in order])
    for tab, mode in zip(tabs, order):
        with tab:
            rows = sorted(grouped[mode], key=lambda r: r.get("vuln_id", ""))
            placeholder = st.empty()
            if skip_anim or speed == 0.0:
                placeholder.markdown(_grid_html(rows, len(rows)), unsafe_allow_html=True)
            else:
                for k in range(len(rows) + 1):
                    placeholder.markdown(_grid_html(rows, k), unsafe_allow_html=True)
                    time.sleep(speed)
            v, n, frac = _mode_rate(rows)
            st.markdown(
                f"**{mode}** — verified `{v}/{n}` &nbsp; "
                + _badge(f"{frac:.0%} verified-fix rate", ok=(frac > 0)),
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Headline")
    headline_cols = st.columns(len(order))
    for i, mode in enumerate(order):
        _, _, frac = _mode_rate(grouped[mode])
        headline_cols[i].markdown(
            f"#### {mode}<br/><span style='font-size:2.2em;font-weight:700;'>"
            f"{frac:.0%}</span>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# View 3: Side-by-side beat (optional, replay-only)
# ---------------------------------------------------------------------------

def render_side_by_side(cfg: CliConfig) -> None:
    st.subheader("Side-by-side — base model vs tuned model on the same vuln")
    st.caption(
        "Both fixes drop into the same Vul4J project. The verifier verdict "
        "is the only judge. Replay-only — paired caches required."
    )

    # Look for paired caches: <id>.json (tuned) and <id>.base.json (base).
    cached_ids = _list_cached_ids(cfg.cache_dir)
    base_ids = [i.removesuffix(".base") for i in cached_ids if i.endswith(".base")]
    paired = [i for i in base_ids if i in cached_ids]

    if not paired:
        st.info(
            "No paired caches yet. To produce one for `VUL4J-10`:\n\n"
            "```bash\n"
            "# tuned cache (uses --endpoint, --model)\n"
            "streamlit run demo/app.py -- --live --save-cache --vuln-id VUL4J-10\n\n"
            "# base cache (point at the base endpoint, save as VUL4J-10.base.json)\n"
            "streamlit run demo/app.py -- --live --save-cache --vuln-id VUL4J-10 \\\n"
            "  --endpoint http://127.0.0.1:8001/v1 --model Qwen2.5-Coder-7B-Instruct\n"
            "# then rename demo/cache/VUL4J-10.json -> demo/cache/VUL4J-10.base.json\n"
            "```"
        )
        return

    vuln_id = st.selectbox("Paired id", options=paired, index=0)
    tuned = _load_cache(cfg.cache_dir, vuln_id)
    base = _load_cache(cfg.cache_dir, f"{vuln_id}.base")
    if not tuned or not base:
        st.warning("Paired cache files exist but failed to parse.")
        return

    cwe = tuned.get("cwe") or base.get("cwe") or "CWE-UNKNOWN"
    st.markdown(f"**Same input.** CWE: `{cwe}` · vuln_id: `{vuln_id}`")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Base model — single shot")
        bv = base.get("fixed_verdict") or "?"
        if bv == VERDICT_GREEN:
            _banner_green(f"verifier: `{bv}`")
        else:
            _banner_red(f"verifier: `{bv}` — plausible-looking, still broken")
        _show_diff(base.get("original_code", ""), base.get("candidate_patch", ""))

    with right:
        st.markdown("#### Tuned model + agent loop")
        tv = tuned.get("fixed_verdict") or "?"
        if tv == VERDICT_GREEN:
            _banner_green(f"verifier: `{tv}`")
        else:
            _banner_red(f"verifier: `{tv}`")
        _show_diff(tuned.get("original_code", ""), tuned.get("candidate_patch", ""))


# ---------------------------------------------------------------------------
# Sidebar + dispatch
# ---------------------------------------------------------------------------

def _sidebar(cfg: CliConfig) -> str:
    st.sidebar.title("PatchProof demo")
    st.sidebar.caption(
        "🎥 Replay mode" if not cfg.live else "🔴 LIVE — talks to vLLM + verifier"
    )

    view = st.sidebar.radio(
        "View",
        ["Single vulnerability", "Batch wall", "Side-by-side"],
        index=0,
    )

    with st.sidebar.expander("Config (overrides CLI flags)"):
        cfg.results_path = st.text_input("Results JSON", value=cfg.results_path)
        cfg.cache_dir = st.text_input("Cache dir", value=cfg.cache_dir)
        cfg.endpoint = st.text_input("Tuned endpoint", value=cfg.endpoint)
        cfg.model = st.text_input("Tuned model", value=cfg.model)
        cfg.max_attempts = int(st.number_input(
            "Max attempts", min_value=1, max_value=10, value=cfg.max_attempts,
        ))

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Verifier = source of truth (deterministic Vul4J in Docker). "
        "No LLM ever judges this demo's outcome."
    )
    return view


def main() -> None:
    cfg = _parse_cli()
    st.set_page_config(
        page_title="PatchProof — attack → fix → verify",
        page_icon="🛡️",
        layout="wide",
    )
    st.title("PatchProof")
    st.caption(
        "Self-hosted multi-agent system that fixes Java vulnerabilities — "
        "and **proves** every fix with the real exploit test + regression suite."
    )

    view = _sidebar(cfg)
    if view == "Single vulnerability":
        render_single(cfg)
    elif view == "Batch wall":
        render_batch(cfg)
    else:
        render_side_by_side(cfg)


main()
