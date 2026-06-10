"""Step 2 — Build train.jsonl + val.jsonl from the CVEfixes SQLite DB.

The CVEfixes dataset ships as a SQL dump on Zenodo. Restore it once with

    ./data/download_data.sh path/to/CVEfixes_v1.0.x.sql

which lands the DB at exactly ``data/cvefixes.db`` — the only path the
builder reads by default.

Usage
-----

    # 1. Verify the schema matches what we expect (recommended on first use)
    python -m data.build_dataset inspect

    # 2. Emit train + val JSONL
    python -m data.build_dataset build

Output format (one JSON object per line) matches train/finetune.py:

    {"cwe": "CWE-89", "vulnerable": "...method body...", "fixed": "..."}

Why the cap measures the *formatted* example
--------------------------------------------
The trainer's actual budget is ``config.MAX_SEQ_LENGTH``, and what has to fit
inside it is the chat-templated [system + user + vulnerable + assistant + fixed]
sequence — not just the method body. Capping only the body lets oversize
examples slip through and get silently truncated during SFT. We measure
``effective_max_tokens()`` (MAX_SEQ_LENGTH minus a small special-token margin)
on the full formatted sequence, using the model's own tokenizer.

We do NOT reuse ``method_change.token_count``: that field is from a code-metrics
tool and uses a different tokenizer than the base model.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from config import (
    MODEL_ID,
    MAX_SEQ_LENGTH,
    effective_max_tokens,
    example_token_length,
    parse_cwe_list,
)

DEFAULT_DB = "data/cvefixes.db"
DEFAULT_OUT = "data/train.jsonl"
DEFAULT_VAL_OUT = "data/val.jsonl"
DEFAULT_HELD_OUT = "data/held_out_cves.txt"
DEFAULT_CWES = ["CWE-89", "CWE-22", "CWE-502", "CWE-611", "CWE-78", "CWE-79"]
REQUIRED_TABLES = {"cve", "fixes", "file_change", "method_change", "cwe_classification"}


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"CVEfixes DB not found at {db_path}. Restore the Zenodo SQL dump:\n"
            f"    ./data/download_data.sh path/to/CVEfixes_v1.0.x.sql"
        )
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


# ---------------------------------------------------------------------------
# inspect mode
# ---------------------------------------------------------------------------

def inspect_db(db_path: Path) -> int:
    con = _connect_ro(db_path)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]
    print(f"# {db_path}  ({len(tables)} tables)")
    for t in tables:
        marker = "  *" if t in REQUIRED_TABLES else "   "
        cols = list(cur.execute(f"PRAGMA table_info({t})"))
        print(f"\n{marker} {t}  ({len(cols)} cols)")
        for _cid, name, ctype, notnull, _dflt, pk in cols:
            flags = []
            if pk:
                flags.append("pk")
            if notnull:
                flags.append("not null")
            flag_s = f"  [{', '.join(flags)}]" if flags else ""
            print(f"      {name:<28} {ctype}{flag_s}")

    missing = REQUIRED_TABLES - set(tables)
    if missing:
        print(f"\nWARN: required tables missing: {sorted(missing)}", file=sys.stderr)
        return 1

    rc = 0
    rc |= _inspect_method_change(cur)
    rc |= _inspect_join_path(cur)
    return rc


def _inspect_method_change(cur: sqlite3.Cursor) -> int:
    """Surface the two gotchas the build query depends on:

    (1) what type/values does ``before_change`` actually take, and
    (2) does ``method_change`` carry language info itself (it shouldn't —
        ``programming_language`` lives on ``file_change``).
    """
    print("\n# method_change — sanity checks")
    cols = {r[1] for r in cur.execute("PRAGMA table_info(method_change)")}

    if "before_change" not in cols:
        print("  FAIL: method_change.before_change column missing", file=sys.stderr)
        return 1
    distinct = [r[0] for r in cur.execute(
        "SELECT DISTINCT before_change FROM method_change LIMIT 10"
    )]
    types = sorted({type(v).__name__ for v in distinct})
    print(f"  before_change distinct values: {distinct!r}")
    print(f"  before_change value types    : {types}")
    print("    -> _is_before() handles bool/int/'True'/'False'; check the values "
          "above match one of those.")

    if "programming_language" in cols:
        print("  NOTE: method_change has a programming_language column too. The "
              "documented schema puts language on file_change; we filter there "
              "to stay consistent.")

    print("\n  sample rows (first 3):")
    for r in cur.execute("""
        SELECT file_change_id, name, signature, before_change,
               substr(code, 1, 80) AS code_preview
        FROM method_change
        LIMIT 3
    """):
        print(f"    fcid={r['file_change_id']!r} "
              f"name={r['name']!r} "
              f"before={r['before_change']!r} "
              f"sig={(r['signature'] or '')[:60]!r}")
        print(f"      code: {r['code_preview']!r}")
    return 0


def _inspect_join_path(cur: sqlite3.Cursor) -> int:
    """Confirm the cve -> fixes -> file_change -> method_change chain works
    *and* that programming_language='Java' actually filters."""
    print("\n# join-path check (cve -> fixes -> file_change -> method_change)")
    fc_cols = {r[1] for r in cur.execute("PRAGMA table_info(file_change)")}
    if "programming_language" not in fc_cols:
        print("  FAIL: file_change.programming_language missing — language filter "
              "won't work", file=sys.stderr)
        return 1

    try:
        n_java = cur.execute(
            "SELECT COUNT(*) FROM file_change WHERE programming_language = 'Java'"
        ).fetchone()[0]
        n_mc = cur.execute("""
            SELECT COUNT(*)
            FROM file_change fc
            JOIN method_change mc ON mc.file_change_id = fc.file_change_id
            WHERE fc.programming_language = 'Java'
        """).fetchone()[0]
        n_full = cur.execute("""
            SELECT COUNT(*)
            FROM cve
            JOIN cwe_classification cwe ON cwe.cve_id = cve.cve_id
            JOIN fixes                   ON fixes.cve_id = cve.cve_id
            JOIN file_change fc          ON fc.hash = fixes.hash
            JOIN method_change mc        ON mc.file_change_id = fc.file_change_id
            WHERE fc.programming_language = 'Java'
        """).fetchone()[0]
    except sqlite3.Error as e:
        print(f"  FAIL: join query errored: {e}", file=sys.stderr)
        print("        Re-run inspect and compare actual column names against "
              "the SQL in _inspect_join_path / _pull_rows.", file=sys.stderr)
        return 1

    print(f"  Java file_change rows                 : {n_java}")
    print(f"  Java method_change rows (after join)  : {n_mc}")
    print(f"  full-chain joined rows (Java + CWEs)  : {n_full}")
    print("    -> if 'full-chain' is 0, the cve/fixes side of the join is "
          "broken (commonly: fixes.hash mismatch).")
    return 0


# ---------------------------------------------------------------------------
# build mode
# ---------------------------------------------------------------------------

def _load_held_out(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.add(s.upper())
    return out


def _load_tokenizer(name: str):
    """Lazy-load the Qwen tokenizer; warn (don't crash) if unavailable.

    Authoring/runs outside the AMD box may not have transformers installed or
    HF reachable. We surface a clear warning and fall back to a char/4 estimate
    inside config.example_token_length so the build script still runs end-to-end
    for testing — the warning is the signal to re-run with a real tokenizer
    before training.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        print(f"WARN: transformers unavailable ({e}); using char/4 token estimate",
              file=sys.stderr)
        return None
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:
        print(f"WARN: could not load tokenizer {name!r} ({e}); using char/4 estimate",
              file=sys.stderr)
        return None


def _pull_rows(con: sqlite3.Connection, target_cwes: set[str]) -> list[sqlite3.Row]:
    have = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    missing = REQUIRED_TABLES - have
    if missing:
        raise RuntimeError(
            f"required tables missing from db: {sorted(missing)}. "
            f"Run `python -m data.build_dataset inspect` to see the real schema."
        )

    placeholders = ",".join("?" for _ in target_cwes)
    sql = f"""
        SELECT
            cve.cve_id            AS cve_id,
            cwe.cwe_id            AS cwe_id,
            fc.file_change_id     AS file_change_id,
            fc.filename           AS filename,
            mc.name               AS method_name,
            mc.signature          AS method_signature,
            mc.before_change      AS before_change,
            mc.code               AS code
        FROM cve
        JOIN cwe_classification cwe ON cwe.cve_id = cve.cve_id
        JOIN fixes                  ON fixes.cve_id = cve.cve_id
        JOIN file_change fc         ON fc.hash = fixes.hash
        JOIN method_change mc       ON mc.file_change_id = fc.file_change_id
        WHERE cwe.cwe_id IN ({placeholders})
          AND fc.programming_language = 'Java'
    """
    return con.execute(sql, sorted(target_cwes)).fetchall()


def _is_before(value) -> bool:
    """method_change.before_change can be 'True'/'False', 1/0, or bool.

    --inspect prints the distinct values so we can confirm against the live DB.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "t", "yes")
    return False


def build(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    target_cwes = parse_cwe_list(args.target_cwes)
    if not target_cwes:
        print("FAIL: --target-cwes is empty", file=sys.stderr)
        return 2

    cap = effective_max_tokens()
    held_out = _load_held_out(Path(args.held_out))
    print(f"model         : {MODEL_ID}")
    print(f"max_seq_len   : {MAX_SEQ_LENGTH} (cap on formatted example: {cap})")
    print(f"target CWEs   : {sorted(target_cwes)}")
    print(f"held-out CVEs : {len(held_out)} loaded from {args.held_out}")

    con = _connect_ro(db_path)
    rows = _pull_rows(con, target_cwes)
    print(f"rows pulled   : {len(rows)} method_change rows (Java + target CWEs)")

    # Group by (file_change_id, method signature) -> {before:[code,...], after:[code,...]}.
    # Pairing within the same file_change_id avoids accidentally matching a
    # method against a same-named method from a different commit.
    groups: dict[tuple, dict] = defaultdict(
        lambda: {"before": [], "after": [], "cve_id": None, "cwe_id": None},
    )
    for r in rows:
        key = (r["file_change_id"], r["method_signature"] or r["method_name"])
        slot = "before" if _is_before(r["before_change"]) else "after"
        groups[key][slot].append(r["code"] or "")
        groups[key]["cve_id"] = r["cve_id"]
        groups[key]["cwe_id"] = r["cwe_id"]

    tok = _load_tokenizer(MODEL_ID)
    using_real_tokenizer = tok is not None

    pairs: list[dict] = []
    drop_no_pair = drop_empty = drop_unchanged = drop_leakage = drop_oversize = 0
    per_cwe: Counter[str] = Counter()
    leaked_cwe: Counter[str] = Counter()

    for g in groups.values():
        before = next((b for b in g["before"] if b and b.strip()), None)
        after = next((a for a in g["after"] if a and a.strip()), None)

        if before is None or after is None:
            drop_no_pair += 1
            continue
        if not before.strip() or not after.strip():
            drop_empty += 1
            continue
        if before.strip() == after.strip():
            drop_unchanged += 1
            continue

        cve_id = (g["cve_id"] or "").upper()
        if cve_id in held_out:
            drop_leakage += 1
            leaked_cwe[g["cwe_id"]] += 1
            continue

        # Measure the full formatted example (system + user + vuln + assistant
        # + fix) against the trainer's budget — not just the method body. That
        # is what actually has to fit in max_seq_length at training time.
        if example_token_length(tok, g["cwe_id"], before, after) > cap:
            drop_oversize += 1
            continue

        per_cwe[g["cwe_id"]] += 1
        pairs.append({
            "cwe": g["cwe_id"],
            "vulnerable": before,
            "fixed": after,
        })

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * args.val_fraction))) if pairs else 0
    val_split, train_split = pairs[:n_val], pairs[n_val:]

    out_train = Path(args.out)
    out_val = Path(args.val_out)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_train, train_split)
    _write_jsonl(out_val, val_split)

    print()
    print(f"wrote {len(train_split):>6} train pairs -> {out_train}")
    print(f"wrote {len(val_split):>6} val   pairs -> {out_val}")
    print()
    print("kept per CWE:")
    for cwe, n in per_cwe.most_common():
        print(f"  {cwe:<10} {n}")
    print()
    print("dropped:")
    print(f"  no before/after pair     : {drop_no_pair}")
    print(f"  empty side               : {drop_empty}")
    print(f"  unchanged (before==after): {drop_unchanged}")
    print(f"  oversize (> {cap} formatted tokens): {drop_oversize}")
    print(f"  leakage (held-out CVE)   : {drop_leakage}")
    if leaked_cwe:
        for cwe, n in leaked_cwe.most_common():
            print(f"    {cwe:<10} {n}")
    if not using_real_tokenizer:
        print()
        print("WARN: oversize counts used a char/4 estimate, not the real Qwen "
              "tokenizer. Re-run with transformers + HF access before relying "
              "on these numbers.")
    return 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Build PatchProof training data from CVEfixes (Java, target CWEs).",
    )
    sub = p.add_subparsers(dest="mode", required=True)

    pi = sub.add_parser("inspect", help="print SQLite schema + sample method_change rows")
    pi.add_argument("--db", default=DEFAULT_DB)

    pb = sub.add_parser("build", help="emit train.jsonl + val.jsonl")
    pb.add_argument("--db", default=DEFAULT_DB)
    pb.add_argument("--out", default=DEFAULT_OUT)
    pb.add_argument("--val-out", default=DEFAULT_VAL_OUT)
    pb.add_argument("--val-fraction", type=float, default=0.05)
    pb.add_argument("--held-out", default=DEFAULT_HELD_OUT,
                    help="newline-separated CVE ids to drop (Vul4J + llm-vul)")
    pb.add_argument("--target-cwes", default=",".join(DEFAULT_CWES))
    pb.add_argument("--seed", type=int, default=13)

    args = p.parse_args()
    if args.mode == "inspect":
        return inspect_db(Path(args.db))
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
