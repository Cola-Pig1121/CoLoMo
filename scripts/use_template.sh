#!/usr/bin/env bash
# Usage:
#   scripts/use_template.sh <template_id_or_key> <dest_path>
#
# Preferred path: use the Python tool index (tools.py) to resolve keys.
# Falls back to legacy filesystem matching when Python unavailable.

set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <template_id_or_key> <dest_path>" >&2
  exit 1
fi
TEMPLATE_ID="$1"
DEST_PATH="$2"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASE_DIR="$ROOT_DIR/rust_watchdog/templates/model_templates"
TOOLS_PY="$BASE_DIR/tools.py"

# 1) Preferred: Python tool
if command -v python >/dev/null 2>&1; then
  if python "$TOOLS_PY" fetch "$TEMPLATE_ID" "$DEST_PATH" 2>/dev/null; then
    exit 0
  fi
fi

# 2) Legacy fallback
SRC_FILE="$BASE_DIR/${TEMPLATE_ID}.py"
SRC_DIR="$BASE_DIR/${TEMPLATE_ID}"
SNIPPET_DIR="$BASE_DIR/pytorch常用代码模板"

mkdir -p "$(dirname "$DEST_PATH")" || true

if [ -f "$SRC_FILE" ]; then
  cp "$SRC_FILE" "$DEST_PATH"
  echo "Copied $SRC_FILE -> $DEST_PATH"
  exit 0
fi

if [ -d "$SRC_DIR" ]; then
  if [ -e "$DEST_PATH" ] && [ ! -d "$DEST_PATH" ]; then
    echo "ERROR: Destination exists and is not a directory: $DEST_PATH" >&2
    exit 2
  fi
  mkdir -p "$DEST_PATH"
  cp -r "$SRC_DIR" "$DEST_PATH/"
  echo "Copied directory $SRC_DIR -> $DEST_PATH/"
  exit 0
fi

if [ -d "$SNIPPET_DIR" ]; then
  # shellcheck disable=SC2207
  CANDIDATES=( $(ls "$SNIPPET_DIR" | tr '\n' '\0' | xargs -0 -I{} sh -c "echo {}" | grep -i ".*${TEMPLATE_ID}.*\.py" || true) )
  if [ ${#CANDIDATES[@]} -eq 1 ]; then
    SRC_FILE2="$SNIPPET_DIR/${CANDIDATES[0]}"
    cp "$SRC_FILE2" "$DEST_PATH"
    echo "Copied $SRC_FILE2 -> $DEST_PATH"
    exit 0
  elif [ ${#CANDIDATES[@]} -gt 1 ]; then
    echo "Multiple matches for '$TEMPLATE_ID':" >&2
    for f in "${CANDIDATES[@]}"; do echo " - $f" >&2; done
    echo "Please refine the template_id." >&2
    exit 4
  fi
fi

echo "Template not found in index or filesystem patterns" >&2
exit 3
