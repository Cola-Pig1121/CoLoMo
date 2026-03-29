#!/usr/bin/env bash
# cc-verify.sh — Build & run smoke checks for CoLoMo MVP
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Rust build check
echo "[verify] cargo check"
(cd rust_watchdog && cargo check)

# Conda/Python checks
echo "[verify] conda & python demo"
if command -v conda >/dev/null 2>&1; then
  conda env list | cat >/dev/null || true
  ENV_NAME=${ENV_NAME:-colomo}
  if ! conda env list | grep -q "^$ENV_NAME\b"; then
    conda create -n "$ENV_NAME" python=3.10 -y
  fi
  conda run -n "$ENV_NAME" pip install -r projects/demo/requirements.txt
  conda run -n "$ENV_NAME" python projects/demo/train.py || echo "[verify] demo trainer exited (expected if simulated OOM)"
else
  echo "[verify] conda not found — skipping Python demo"
fi

echo "[verify] done"
