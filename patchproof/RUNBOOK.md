# PatchProof — Runbook

One ordered checklist. Run from `patchproof/`. Do not start step N+1 until step N's gate is green.

CPU steps are free; do all of them first. GPU steps burn credits on the MI300X — only after every CPU step has passed.

```bash
cd patchproof/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## [ ] Step 0 — Restore CVEfixes DB        (CPU)

```bash
./data/download_data.sh path/to/CVEfixes_v1.0.x.sql
```

Gate: `data/cvefixes.db` exists and `sqlite3 data/cvefixes.db ".tables"` lists `cve`, `fixes`, `file_change`, `method_change`, `cwe_classification`.

---

## [ ] Step 1 — Verifier smoke test         (CPU, Docker)

Pulls `tuhhsse/vul4j:alldeps` on first run. Fill the four `VUL4J-XX` placeholders in `verifier/smoke_test.py` first.

```bash
python -m verifier.verify --smoke
```

Gate: ≥5 ids return `verified` on the human patch AND `still_vulnerable` on the original. **Nothing else runs until this is green.**

---

## [ ] Step 2 — Build dataset                (CPU)

Fill `data/held_out_cves.txt` with Vul4J + llm-vul CVE ids before `build`, or leakage filtering is a no-op.

```bash
python -m data.build_dataset inspect
python -m data.build_dataset build
```

Gate: `data/train.jsonl` and `data/val.jsonl` exist, non-empty, every line parses, every row has `cwe` / `vulnerable` / `fixed`. No held-out CVE appears in either file.

---

## [ ] Step 3 — Agent orchestrator tests     (CPU)

```bash
pytest tests/ -v
```

Gate: all of `test_router.py` + `test_orchestrator.py` pass (router branches, happy path, retry threads feedback, budget exhaustion, triage-runs-once).

---

## [ ] Step 4 — Fine-tune                    (GPU, MI300X)

Prereqs: Steps 1–3 green; ROCm torch installed; env set.

```bash
export HSA_OVERRIDE_GFX_VERSION=9.4.2
python -m train.finetune --smoke
```

Gate (smoke): `outputs/train_smoke/train.log` shows loss decreasing across the 30 steps; `train_metrics.json` has non-zero `wall_clock_seconds` and `peak_gpu_memory_gb`; `lora_adapter/` and `merged_16bit/` both exist. If flat/NaN, fix before the real run.

```bash
python -m train.finetune
```

Gate (real): `outputs/train/merged_16bit/` exists; `train_metrics.json` shows final train loss < first logged train loss; val loss recorded.

---

## [ ] Step 5 — Serve                        (GPU, MI300X)

Prereqs: Step 4 produced `outputs/train/merged_16bit/`. `serve.sh` sets the ROCm env (`HSA_OVERRIDE_GFX_VERSION=9.4.2`, `VLLM_ROCM_USE_AITER=1`).

The AMD Dev Cloud workspace = one Jupyter notebook + a JupyterLab launcher that opens fresh terminals (same layout as the airbnb MCP sample). vLLM is a long-running server, so it lives in a **terminal**, not a notebook cell. The notebook (and our agent code) is the **client** that talks to it.

### 5a. Terminal 1 — launch vLLM (blocks)

From the JupyterLab launcher → "Terminal", then:

```bash
cd patchproof/
bash serve/serve.sh --api-key abc-123
# equivalent to:
#   VLLM_USE_TRITON_FLASH_ATTN=0 \
#   HSA_OVERRIDE_GFX_VERSION=9.4.2 VLLM_ROCM_USE_AITER=1 \
#   vllm serve outputs/train/merged_16bit \
#     --served-model-name patchproof-merged \
#     --api-key abc-123 --port 8000 --host 0.0.0.0
```

Wait until vLLM prints `Application startup complete` / `Uvicorn running on http://0.0.0.0:8000`. **Leave this terminal running** for the rest of the session — closing it kills the model.

FP8 variant for the throughput slide goes on its own port so it can coexist:

```bash
bash serve/serve.sh --fp8 --served-name patchproof-merged-fp8 --port 8002 --api-key abc-123
```

### 5b. Terminal 2 — GPU watch + health check

Open a second terminal (don't reuse Terminal 1):

```bash
watch rocm-smi                                      # leave running; sanity-check VRAM + utilisation
curl -sf http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer abc-123" | grep -q patchproof-merged
```

### 5c. Notebook — point clients at the endpoint

In a notebook cell (this mirrors cell 2 of `build_airbnb_agent_mcp.ipynb` — same env-var pattern, just renamed for our client):

```python
import os
os.environ["PATCHPROOF_VLLM_BASE_URL"] = "http://127.0.0.1:8000/v1"
os.environ["PATCHPROOF_VLLM_MODEL"]    = "patchproof-merged"
os.environ["PATCHPROOF_VLLM_API_KEY"]  = "abc-123"

from serve.health_check import main as health
health()                                            # prints a fenced fix on success
```

Then wire the agent loop to the same endpoint:

```python
from agents.graph import build_graph
from agents.model_client import VLLMModelClient
from verifier.verify import verify

graph = build_graph(llm_client=..., model_client=VLLMModelClient(), verifier_fn=verify)
```

`VLLMModelClient()` reads the three `PATCHPROOF_VLLM_*` env vars you just set, so nothing else needs to change.

Gate: `curl /v1/models` returns the served model name AND `serve.health_check` prints a non-empty fenced fix using `agents.model_client.VLLMModelClient`.

---

## [ ] Step 6 — Benchmark (Slide 4)           (GPU, MI300X)

Prereqs: Step 5 tuned endpoint up + `serve.health_check` green. `data/eval_vul4j_ids.txt` populated with held-out VUL4J ids. Docker still required — the benchmark uses the REAL verifier per id.

Honest-comparison rule: `base` MUST be `Qwen2.5-Coder-7B-Instruct` (`config.MODEL_ID`), prompt-only. Do NOT substitute a frontier API — the slide's delta only means anything against the same base.

1) Serve the BASE model on a second port (own shell):

```bash
bash serve/serve.sh \
  --model-path Qwen/Qwen2.5-Coder-7B-Instruct \
  --served-name Qwen2.5-Coder-7B-Instruct \
  --port 8001
```

2) Run each mode (each call appends to `outputs/benchmark_results.json`):

```bash
# base — same Qwen2.5-Coder-7B, prompt-only, single shot
python -m eval.benchmark --mode base \
  --endpoint http://127.0.0.1:8001/v1 \
  --model Qwen2.5-Coder-7B-Instruct

# tuned — fine-tuned model, single shot
python -m eval.benchmark --mode tuned \
  --endpoint http://127.0.0.1:8000/v1 \
  --model patchproof-merged

# tuned_loop — fine-tuned model + agent loop (triage/patch/verify/self-heal)
python -m eval.benchmark --mode tuned_loop \
  --endpoint http://127.0.0.1:8000/v1 \
  --model patchproof-merged
```

3) Emit the 3-bar chart and summary table:

```bash
python -m eval.benchmark --mode plot
```

Gate: `outputs/benchmark_chart.png` exists; printed table shows `tuned >= base` AND `tuned_loop >= tuned` on verified-fix rate. Per-id `tokens_in`, `tokens_out`, `latency_seconds`, `attempts` are in `outputs/benchmark_results.json` — those are the Slide-4 fields.

Optional — FP8 throughput comparison for the optimization slide:

```bash
# third shell, FP8 endpoint:
bash serve/serve.sh --fp8 --served-name patchproof-merged-fp8 --port 8002

python -m eval.benchmark --mode throughput --label fp16 \
  --endpoint http://127.0.0.1:8000/v1 --model patchproof-merged
python -m eval.benchmark --mode throughput --label fp8  \
  --endpoint http://127.0.0.1:8002/v1 --model patchproof-merged-fp8
```

---

## [ ] Step 7 — Demo                         (Streamlit; live on box, replay off box)

`demo/app.py` is a Streamlit UI with two main views: **Single vulnerability** (attack → triage → RED → loop → diff → GREEN → evidence) and **Batch wall** (animated grid over `outputs/benchmark_results.json`, headline rate per mode). A third **Side-by-side** view is replay-only and compares a base-model fix against the tuned-model+loop fix on the same vuln.

Two modes: `--live` (hits real vLLM + verifier) and `--replay` (renders only from `outputs/benchmark_results.json` + `demo/cache/*.json`). **Replay is the default** so the screen-recording stays fast and reproducible. Live runs are tens of seconds each (Maven compile + tests in Docker) and can flake on camera.

### 7a. One-time on the box — populate the demo cache (live)

vLLM (Step 5a/T2) and Docker need to be reachable. From a new Terminal:

```bash
cd patchproof && source .venv/bin/activate
streamlit run demo/app.py -- \
  --live --save-cache --vuln-id VUL4J-10 \
  --endpoint http://127.0.0.1:8000/v1 \
  --model patchproof-merged \
  --api-key abc-123
```

In the UI: confirm "🔴 LIVE" in the sidebar → click **▶ Run live attempt** → wait for the spinner → click **💾 Save to cache**. Writes `demo/cache/VUL4J-10.json`. Repeat for any other ids you want pre-staged (3–5 is plenty).

For the optional side-by-side view, also record a base-model run on the second endpoint and rename the file:

```bash
streamlit run demo/app.py -- \
  --live --save-cache --vuln-id VUL4J-10 \
  --endpoint http://127.0.0.1:8001/v1 \
  --model Qwen2.5-Coder-7B-Instruct \
  --api-key abc-123
mv demo/cache/VUL4J-10.json demo/cache/VUL4J-10.base.json
# Now re-run the tuned recording above to re-create demo/cache/VUL4J-10.json.
```

Commit the cache so the recording laptop has the JSON:

```bash
git add demo/cache/*.json
git commit -m "Pre-stage demo cache for recording"
```

### 7b. Recording — replay only (no vLLM, no Docker needed)

```bash
streamlit run demo/app.py
# or with overrides:
streamlit run demo/app.py -- \
  --results outputs/benchmark_results.json \
  --cache-dir demo/cache \
  --vuln-id VUL4J-10
```

The Single Vulnerability view renders the cached attempt instantly. The Batch wall view reads `outputs/benchmark_results.json` (the real Step-6 output) and animates RED → GREEN per id, then shows the per-mode headline rate. The Animation-speed slider + Skip-animation checkbox let you tune the take.

Gate: with no vLLM/Docker running locally, `streamlit run demo/app.py` boots, the Single Vulnerability view shows a cached `VUL4J-10` going RED → GREEN end-to-end, and the Batch wall renders the headline rate per mode.
