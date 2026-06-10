"""Step 4 — Fine-tune Qwen2.5-Coder-7B-Instruct with 16-bit LoRA on the MI300X.

Why 16-bit and not QLoRA
------------------------
bitsandbytes is unstable on ROCm. We have 192 GB of HBM3 on the MI300X, so
there's no reason to 4-bit anything — straight 16-bit LoRA is faster, more
stable, and produces a cleaner merged model for serving.

Single source of truth for the prompt format
--------------------------------------------
We import ``MODEL_ID``, ``MAX_SEQ_LENGTH``, and ``build_messages`` from the
shared ``config`` module. ``build_messages`` is the EXACT same call the
Patcher's model_client must use at inference time. If those two ever drift,
the model trained here will see one format and be prompted with another.
Do not redefine the template in this file.

Order of operations on the box
------------------------------
1. Step 1 verifier smoke green.
2. Step 2 dataset built (data/train.jsonl + data/val.jsonl exist).
3. ``export HSA_OVERRIDE_GFX_VERSION=9.4.2``
4. ``python -m train.finetune --smoke``   (~20 examples, ~30 steps)
5. ``python -m train.finetune``           (the real run)

The --smoke run exists specifically to burn ~2 minutes of GPU time validating
the full pipeline before we kick off the multi-hour real run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import MAX_SEQ_LENGTH, MODEL_ID, build_messages

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_JSONL = "data/train.jsonl"
DEFAULT_VAL_JSONL = "data/val.jsonl"
DEFAULT_OUT_DIR = "outputs/train"
SMOKE_OUT_DIR = "outputs/train_smoke"

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_R = 16
LORA_ALPHA = 16

EXPECTED_HSA_OVERRIDE = "9.4.2"


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger("patchproof.train")
    logger.info("logging to %s", log_path)
    return logger


def _check_amd_env(logger: logging.Logger) -> None:
    """Warn loudly if the MI300X env vars look wrong. Do not hard-fail —
    the same script must run on a CUDA dev box for sanity checks."""
    gfx = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if gfx is None:
        logger.warning(
            "HSA_OVERRIDE_GFX_VERSION is unset. On MI300X this must be %r — "
            "ROCm will misidentify the device without it.", EXPECTED_HSA_OVERRIDE,
        )
    elif gfx != EXPECTED_HSA_OVERRIDE:
        logger.warning(
            "HSA_OVERRIDE_GFX_VERSION=%r (expected %r for MI300X)",
            gfx, EXPECTED_HSA_OVERRIDE,
        )
    else:
        logger.info("HSA_OVERRIDE_GFX_VERSION=%s", gfx)

    rocm = os.environ.get("ROCM_PATH")
    if rocm:
        logger.info("ROCM_PATH=%s", rocm)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Build it with `python -m data.build_dataset build`."
        )
    rows = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{i} is not valid JSON: {e}") from e
    return rows


def _format_rows(rows: list[dict], tokenizer) -> list[dict]:
    """Apply the shared chat template once, eagerly, so each example's training
    text is exactly what build_messages produces.

    Storing as a flat ``text`` field lets SFTTrainer treat the dataset as a
    plain text dataset — no special formatter, no risk of TRL silently
    re-applying a chat template on top of ours.
    """
    out: list[dict] = []
    for r in rows:
        msgs = build_messages(r["cwe"], r["vulnerable"], r["fixed"])
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        out.append({"text": text})
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(logger: logging.Logger):
    """16-bit LoRA on MI300X. No 4-bit, no QLoRA — see module docstring."""
    from unsloth import FastModel  # type: ignore

    logger.info("loading base model %s (16-bit) ...", MODEL_ID)
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,           # MI300X has 192 GB; do not quantize
        load_in_8bit=False,
        dtype=None,                   # let Unsloth pick (bf16 on MI300X)
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("tokenizer pad_token unset -> using eos_token")

    logger.info("attaching LoRA adapters (r=%d, alpha=%d, targets=%s) ...",
                LORA_R, LORA_ALPHA, LORA_TARGETS)
    model = FastModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=13,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class TrainMetrics:
    wall_clock_seconds: float
    peak_gpu_memory_bytes: int
    peak_gpu_memory_gb: float
    num_train_examples: int
    num_val_examples: int
    final_train_loss: float | None
    final_val_loss: float | None
    log_history: list[dict[str, Any]]
    model_id: str
    max_seq_length: int
    smoke: bool

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        return d


def _peak_memory_bytes() -> int:
    """torch.cuda.max_memory_allocated works under ROCm too (it reports HIP
    allocations through the same API). Falls back to 0 if torch isn't built
    with GPU support — only relevant when someone dry-runs on CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return 0


def _extract_final_losses(log_history: list[dict]) -> tuple[float | None, float | None]:
    last_train = None
    last_eval = None
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            last_train = float(entry["loss"])
        if "eval_loss" in entry:
            last_eval = float(entry["eval_loss"])
    return last_train, last_eval


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir or (SMOKE_OUT_DIR if args.smoke else DEFAULT_OUT_DIR))
    logger = _setup_logging(out_dir)
    logger.info("== PatchProof training ==")
    logger.info("smoke=%s out_dir=%s", args.smoke, out_dir)
    _check_amd_env(logger)

    # ----- Data -----
    train_rows = _load_jsonl(Path(args.train_jsonl))
    val_rows = _load_jsonl(Path(args.val_jsonl))
    logger.info("loaded %d train rows, %d val rows", len(train_rows), len(val_rows))

    if args.smoke:
        train_rows = train_rows[:args.smoke_examples]
        # Keep val tiny too, just enough to confirm eval runs.
        val_rows = val_rows[: max(2, args.smoke_examples // 4)]
        logger.info("smoke subset: %d train, %d val", len(train_rows), len(val_rows))

    if not train_rows:
        logger.error("no training examples — refusing to start a no-op run")
        return 2

    # ----- Model -----
    model, tokenizer = _load_model_and_tokenizer(logger)

    train_text = _format_rows(train_rows, tokenizer)
    val_text = _format_rows(val_rows, tokenizer) if val_rows else []

    from datasets import Dataset  # type: ignore
    train_ds = Dataset.from_list(train_text)
    val_ds = Dataset.from_list(val_text) if val_text else None

    # ----- Trainer -----
    from trl import SFTConfig, SFTTrainer  # type: ignore

    sft_kwargs: dict[str, Any] = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        seed=args.seed,
        bf16=True,
        fp16=False,
        report_to=[],          # no W&B by default — keeps the box hermetic
        save_total_limit=2,
    )

    if args.smoke:
        sft_kwargs.update(
            max_steps=args.smoke_steps,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no" if val_ds is None else "steps",
            eval_steps=max(1, args.smoke_steps // 3) if val_ds is not None else None,
        )
    else:
        sft_kwargs.update(
            num_train_epochs=args.epochs,
            save_strategy="epoch",
            eval_strategy="epoch" if val_ds is not None else "no",
        )

    sft_config = SFTConfig(**{k: v for k, v in sft_kwargs.items() if v is not None})
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
    )

    # ----- Run -----
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    logger.info("starting trainer.train() ...")
    t0 = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - t0
    logger.info("trainer.train() finished in %.1fs", elapsed)

    # ----- Metrics -----
    log_history = list(trainer.state.log_history)
    final_train, final_eval = _extract_final_losses(log_history)
    metrics = TrainMetrics(
        wall_clock_seconds=elapsed,
        peak_gpu_memory_bytes=_peak_memory_bytes(),
        peak_gpu_memory_gb=_peak_memory_bytes() / (1024 ** 3),
        num_train_examples=len(train_rows),
        num_val_examples=len(val_rows),
        final_train_loss=final_train,
        final_val_loss=final_eval,
        log_history=log_history,
        model_id=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        smoke=args.smoke,
    )
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics.to_json(), indent=2))
    logger.info("wrote metrics -> %s", out_dir / "train_metrics.json")
    logger.info(
        "final_train_loss=%s final_val_loss=%s peak_gpu=%.2f GB",
        final_train, final_eval, metrics.peak_gpu_memory_gb,
    )
    if final_train is not None:
        first_train = next(
            (e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e),
            None,
        )
        if first_train is not None:
            logger.info("loss: %.4f -> %.4f (delta %.4f)",
                        first_train, final_train, final_train - first_train)
            if not args.smoke and final_train >= first_train:
                logger.warning(
                    "training loss did not decrease — check data formatting/LoRA targets",
                )

    # ----- Save -----
    adapter_dir = out_dir / "lora_adapter"
    logger.info("saving LoRA adapter -> %s", adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_dir = out_dir / "merged_16bit"
    logger.info("saving merged 16-bit model -> %s (this is what serve uses)", merged_dir)
    # Unsloth-specific helper. ``save_method`` must literally be the string
    # below for the 16-bit merged checkpoint.
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    logger.info("done.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="PatchProof Step 4 — 16-bit LoRA SFT on MI300X.")
    p.add_argument("--smoke", action="store_true",
                   help="train on ~20 examples for ~30 steps; run this FIRST on the box")
    p.add_argument("--train-jsonl", default=DEFAULT_TRAIN_JSONL)
    p.add_argument("--val-jsonl", default=DEFAULT_VAL_JSONL)
    p.add_argument("--out-dir", default=None,
                   help=f"default: {DEFAULT_OUT_DIR} (or {SMOKE_OUT_DIR} with --smoke)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--smoke-examples", type=int, default=20)
    p.add_argument("--smoke-steps", type=int, default=30)
    args = p.parse_args()
    return train_run(args)


if __name__ == "__main__":
    sys.exit(main())
