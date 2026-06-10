#!/usr/bin/env bash
# Restore the CVEfixes SQL dump into the SQLite DB that build_dataset.py expects.
#
# Usage:
#   ./data/download_data.sh path/to/CVEfixes_v1.0.x.sql
#
# Downloading the dump itself is manual — grab it from Zenodo
# (https://zenodo.org/record/* for "CVEfixes") and point this script at it.
# We commit to a single landing path on purpose: data/build_dataset.py reads
# data/cvefixes.db by default, and that's the only restore route we support.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DB="$SCRIPT_DIR/cvefixes.db"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <path/to/CVEfixes_v1.0.x.sql>" >&2
  exit 2
fi

SQL_DUMP="$1"
if [[ ! -f "$SQL_DUMP" ]]; then
  echo "FAIL: SQL dump not found at $SQL_DUMP" >&2
  exit 1
fi

if [[ -e "$TARGET_DB" ]]; then
  echo "FAIL: $TARGET_DB already exists; remove it first if you really want to overwrite." >&2
  exit 1
fi

if ! command -v sqlite3 >/dev/null; then
  echo "FAIL: sqlite3 CLI not on PATH" >&2
  exit 1
fi

echo "restoring $SQL_DUMP -> $TARGET_DB ..."
sqlite3 "$TARGET_DB" < "$SQL_DUMP"
echo "done. size: $(du -h "$TARGET_DB" | cut -f1)"
echo
echo "next steps:"
echo "  python -m data.build_dataset inspect"
echo "  python -m data.build_dataset build"
