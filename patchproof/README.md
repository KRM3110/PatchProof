# PatchProof

Self-hosted multi-agent system that fixes Java security vulnerabilities and
*proves* each fix by running the real exploit + regression suite.

See `RUNBOOK.md` for the canonical step-by-step run order on the AMD MI300X box
— this README only covers per-step quick starts.

## Build status (code authored)
- [x] Step 0 — CVEfixes restore script
- [x] Step 1 — verifier (CPU, Docker)
- [x] Step 2 — dataset builder
- [x] Step 3 — agent orchestrator + LangGraph wiring (stub-tested on CPU)
- [x] Step 4 — fine-tune (Unsloth + TRL, 16-bit LoRA on MI300X)
- [x] Step 5 — serve (`serve/serve.sh`, vLLM ROCm)
- [x] Step 6 — eval / Slide-4 benchmark
- [x] Step 7 — demo (Streamlit, live + replay)

All seven steps have author-time code in place. Execution happens on the AMD
Developer Cloud per `RUNBOOK.md`; nothing here has been run end-to-end yet.

## Shared config
`config.py` at the repo root holds `MODEL_ID`, `MAX_SEQ_LENGTH`, the SFT prompt
template, `build_messages()`, and `load_eval_vul4j_ids()`. Both
`data/build_dataset.py` and `train/finetune.py` import it so the dataset's
token cap and the trainer's context budget can't drift apart; `eval/benchmark.py`
and `agents/model_client.py` import the same `build_messages` so the prompt
shape at inference is byte-identical to training. Run scripts as modules from
the repo root:

```bash
python -m data.build_dataset inspect
python -m verifier.verify --smoke
```

## Dataset quick start
```bash
# Restore CVEfixes from the Zenodo SQL dump (lands at data/cvefixes.db)
./data/download_data.sh path/to/CVEfixes_v1.0.x.sql

# Confirm the schema's gotchas (before_change values, language join path)
python -m data.build_dataset inspect

# Emit data/train.jsonl + data/val.jsonl
python -m data.build_dataset build
```

Held-out lists live in `data/held_out_cves.txt` (CVE ids excluded from training)
and `data/eval_vul4j_ids.txt` (VUL4J ids used by the benchmark). Both must be
populated before `build`; they're the only way the trainer knows to skip
anything Vul4J or llm-vul might cover.

## Verifier quick start
Requires Docker. The `tuhhsse/vul4j:alldeps` image is pulled on first use.

```bash
export VUL4J_IMAGE=tuhhsse/vul4j:alldeps         # optional override
python -m verifier.verify --smoke                # the Step 1 gate
python -m verifier.verify --id VUL4J-10 --file path/to/patched.java
```

The smoke ids live in `verifier/smoke_test.py::SMOKE_IDS`.

## Agent loop quick start
The graph is dependency-injected. CPU tests use the stubs in `agents/stubs.py`;
the real run on the box wires the vLLM client (Patcher) and the loop's LLM
client (Triage / Self-Heal / Reporter).

```bash
pytest tests/ -v                                  # Step 3 gate (CPU)
```

```python
from agents.graph import build_graph
from agents.model_client import VLLMModelClient   # production Patcher
from verifier.verify import verify

graph = build_graph(
    llm_client=...,        # see eval/benchmark.py::_LoopLLMClient
    model_client=VLLMModelClient(),
    verifier_fn=verify,
)
```

## Fine-tune quick start (GPU, MI300X)
```bash
export HSA_OVERRIDE_GFX_VERSION=9.4.2
python -m train.finetune --smoke                  # ~2 min sanity run
python -m train.finetune                          # real run
# Output: outputs/train/merged_16bit/ ready for vLLM
```

## Serve quick start (GPU, MI300X)
```bash
bash serve/serve.sh --api-key abc-123             # FP16 tuned model on :8000
bash serve/serve.sh --fp8 --served-name patchproof-merged-fp8 --port 8002 --api-key abc-123
```

Health check (second terminal):

```bash
curl -sf http://127.0.0.1:8000/v1/models -H "Authorization: Bearer abc-123"
python -m serve.health_check
```

Client-side env: `PATCHPROOF_VLLM_BASE_URL`, `PATCHPROOF_VLLM_MODEL`,
`PATCHPROOF_VLLM_API_KEY`. `agents.model_client.VLLMModelClient` reads all
three; nothing else needs changing.

## Benchmark quick start (GPU, MI300X)
With the tuned model on `:8000` and the base model served separately on
`:8001`:

```bash
python -m eval.benchmark --mode base       --endpoint http://127.0.0.1:8001/v1 --model Qwen2.5-Coder-7B-Instruct
python -m eval.benchmark --mode tuned      --endpoint http://127.0.0.1:8000/v1 --model patchproof-merged
python -m eval.benchmark --mode tuned_loop --endpoint http://127.0.0.1:8000/v1 --model patchproof-merged
python -m eval.benchmark --mode plot
```

Outputs land in `outputs/benchmark_results.json` (additive across modes) and
`outputs/benchmark_chart.png`. Per-id `tokens_in`, `tokens_out`,
`latency_seconds`, `attempts` are the Slide-4 fields.

## Demo quick start (Streamlit)
Replay mode is the default — no vLLM, no Docker required.

```bash
streamlit run demo/app.py                          # replay (recording)
streamlit run demo/app.py -- --live --save-cache --vuln-id VUL4J-10
```

Three views: Single vulnerability (attack → triage → RED → loop → diff → GREEN
→ evidence), Batch wall (animated grid over `outputs/benchmark_results.json`),
Side-by-side (base vs tuned, replay-only). Cached single-vuln runs live in
`demo/cache/<id>.json`; commit them so the screen-recording laptop only needs
`streamlit`.

## Layout
```
patchproof/
├── RUNBOOK.md, README.md, requirements.txt, config.py
├── data/        dataset builder + restored CVEfixes DB + held-out lists
├── verifier/    deterministic Vul4J-in-Docker verifier (source of truth)
├── train/       Unsloth + TRL fine-tune
├── serve/       vLLM serve wrapper + health check
├── agents/      LangGraph orchestrator + per-role agents + model clients
├── eval/        Slide-4 benchmark (base / tuned / tuned_loop / plot / throughput)
├── tests/       CPU unit tests for router + orchestrator
└── demo/        Streamlit UI (live + replay) and per-id cache
```
