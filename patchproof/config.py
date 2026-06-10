"""Shared training / data config.

Both ``data/build_dataset.py`` and ``train/finetune.py`` import from here so
the dataset's token cap and the trainer's max_seq_length cannot drift apart.

Run scripts as modules from the repo root so this import resolves:

    python -m data.build_dataset build ...
    python -m verifier.verify --smoke
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Base model and training context budget. If you change MAX_SEQ_LENGTH, the
# dataset builder will automatically use the new cap on next run.
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LENGTH = 4096

# Headroom for the chat template's special tokens, BOS/EOS, and any role
# markers the tokenizer adds on top of message content. The dataset's
# effective per-example cap is MAX_SEQ_LENGTH - SEQ_LENGTH_MARGIN.
SEQ_LENGTH_MARGIN = 64

# Canonical SFT format. Keep prose short — most of the budget is code.
SYSTEM_PROMPT = (
    "You are an expert Java security engineer. Given a vulnerable Java method "
    "and its CWE classification, rewrite the method to remove the vulnerability "
    "while preserving behavior. Return only the fixed method inside a single "
    "```java fenced block."
)

USER_TEMPLATE = (
    "CWE: {cwe}\n\n"
    "Vulnerable method:\n"
    "```java\n{vulnerable}\n```"
)

ASSISTANT_TEMPLATE = "```java\n{fixed}\n```"


def build_messages(cwe: str, vulnerable: str, fixed: str | None = None) -> list[dict]:
    """Build the chat-messages list for one example.

    Pass ``fixed=None`` for inference-time prompts (no assistant turn). Pass
    the fixed body to get the full SFT example used for both length-measurement
    and training.
    """
    msgs: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(cwe=cwe, vulnerable=vulnerable)},
    ]
    if fixed is not None:
        msgs.append({"role": "assistant", "content": ASSISTANT_TEMPLATE.format(fixed=fixed)})
    return msgs


def example_token_length(tokenizer, cwe: str, vulnerable: str, fixed: str) -> int:
    """Token count of the *full* formatted example after chat template.

    Uses the model's actual chat template when available (the only honest
    measure). Falls back to a coarse char/4 estimate when no tokenizer is
    loaded — the build script warns loudly when that happens.
    """
    msgs = build_messages(cwe, vulnerable, fixed)
    if tokenizer is None:
        return sum(max(1, len(m["content"]) // 4) for m in msgs)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(msgs, tokenize=True)
            return len(ids)
        except Exception:
            pass
    # Last-ditch: concatenate and encode. Slightly underestimates vs. real
    # chat-template special tokens but is the right order of magnitude.
    joined = "\n".join(m["content"] for m in msgs)
    return len(tokenizer.encode(joined, add_special_tokens=False))


def effective_max_tokens() -> int:
    return MAX_SEQ_LENGTH - SEQ_LENGTH_MARGIN


def parse_cwe_list(s: str) -> set[str]:
    return {c.strip().upper() for c in s.split(",") if c.strip()}


# Held-out Vul4J ids used by eval/benchmark.py. NEVER trained on. The file
# lives in data/ alongside held_out_cves.txt (which is the *CVE*-id list used
# by the dataset builder); this one is the *VUL4J*-id list used by the
# evaluator, because the verifier checks projects out by VUL4J id.
EVAL_VUL4J_IDS_FILE = "data/eval_vul4j_ids.txt"


def load_eval_vul4j_ids(path: str | None = None) -> list[str]:
    """Read the held-out VUL4J id list. One id per line, '#'-comments allowed,
    blanks ignored. Returns [] if the file is missing — the caller decides
    whether that should be a hard error."""
    p = Path(path or EVAL_VUL4J_IDS_FILE)
    if not p.exists():
        return []
    out: list[str] = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


__all__: Iterable[str] = (
    "MODEL_ID",
    "MAX_SEQ_LENGTH",
    "SEQ_LENGTH_MARGIN",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "ASSISTANT_TEMPLATE",
    "EVAL_VUL4J_IDS_FILE",
    "build_messages",
    "example_token_length",
    "effective_max_tokens",
    "parse_cwe_list",
    "load_eval_vul4j_ids",
)
