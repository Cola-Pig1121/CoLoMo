#!/usr/bin/env bash
# cc-pipeline.sh — Sequential autonomous pipeline for CoLoMo MVP
# Language Protocol: prompts to tools are in English
# Template policy: strictly copy from templates/model_templates (no AI codegen for trainers)

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Configurable parameters
TEMPLATE_ID="${1:-pytorch}"           # e.g., pytorch | resnet18
DEST_TRAIN="projects/demo/train.py"    # destination for template copy

# Helper: ensure claude CLI exists (optional)
if ! command -v claude >/dev/null 2>&1; then
  echo "[WARN] 'claude' CLI not found on PATH. This pipeline will print intended prompts but skip execution." >&2
  RUN_CLAUDE=false
else
  RUN_CLAUDE=true
fi

# Step A: Enforce template copy policy
bash "scripts/use_template.sh" "$TEMPLATE_ID" "$DEST_TRAIN"

# Collect plan context file (optional)
PLAN_CTX_FILE=".claude-context-plan.md"
cat > "$PLAN_CTX_FILE" <<'EOF'
CoLoMo MVP phases focus:
- Phase 2: watchdog spawn via `conda run` and stdout/stderr capture to rust_watchdog/logs/<run>.log; expose get_recent_lines().
- Phase 3: GPU monitor via nvidia-smi with simulation fallback.
- Phase 4: Guided TUI editable table for core hyperparams; save to projects/demo/config.yaml.
- Phase 6: JSONL journal writer and rollback --steps N (print inverse ops; no auto-exec).
Constraints:
- Do NOT generate training scripts — when using templates, copy from rust_watchdog/templates/model_templates/* into the project.
- Minimal scope; only touch necessary files.
EOF

run_claude() {
  local PROMPT="$1"
  if [ "$RUN_CLAUDE" = true ]; then
    claude -p "$PROMPT"
  else
    printf "\n--- INTENDED PROMPT (dry-run) ---\n%s\n--- END PROMPT ---\n" "$PROMPT"
  fi
}

# Phase 2 — Watchdog spawn + log capture
run_claude "Read $PLAN_CTX_FILE. Implement rust_watchdog/src/watchdog.rs:
- Spawn training using 'conda run -n <env>' based on projects/demo/config.yaml (backend: cuda->python, tilelang->tilelang run)
- Capture stdout/stderr to rust_watchdog/logs/<run_id>.log and provide get_recent_lines(n: usize) API
- Minimal surface area; no refactor beyond what's required
- Do not generate training code; training scripts come from templates/model_templates and have already been copied"

# Phase 3 — GPU monitor (with simulation)
run_claude "Read $PLAN_CTX_FILE. Implement rust_watchdog/src/gpu_monitor.rs:
- Poll 'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits' periodically
- If command unavailable, simulate readings (flag in struct)
- Expose GpuStatus { total_mb, used_mb, simulated: bool }"

# Phase 4 — Guided TUI edit (table)
run_claude "Read $PLAN_CTX_FILE. Extend TUI (rust_watchdog/src/main.rs or split src/tui.rs):
- Render a table for batch_size, learning_rate, optimizer, dataset_path, backend, conda_env
- Support edit/save; saving writes projects/demo/config.yaml and appends a journal entry
- Add Run action: launches Watchdog to start the process and streams recent lines"

# Phase 6 — Journal + rollback
run_claude "Read $PLAN_CTX_FILE. Add a minimal JSONL journal at rust_watchdog/journal.jsonl:
- On config save, append entries with inverse actions
- Implement a CLI flag '--rollback --steps N' to print inverse actions (do not auto-exec)"

# Optional de-sloppify pass
run_claude "Review only the newly added Rust files. Remove dead code and redundant checks without changing behavior. Run 'cargo check'."

echo "Pipeline steps prepared. To verify locally, run: scripts/cc-verify.sh"
