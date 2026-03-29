# CoLoMo - ML Training Watchdog

**Colo**ral **Mo**nitor: A Rust TUI watchdog for ML training supervision.

## Purpose

CoLoMo supervises ML training runs by:
- Monitoring GPU memory and utilization
- Capturing training logs and generating summaries
- Providing intelligent recommendations (batch size, learning rate, optimizer tuning)
- Journaling all config changes for rollback capability

The goal is to catch issues early (OOM, divergence), suggest optimizations, and allow safe config changes during training.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CoLoMo TUI                            │
│  (ratatui-kit Component with imperative draw)                 │
├────────────────┬────────────────┬───────────────────────────┤
│  Config Table  │  Recent Logs  │  Summary + Recommendation  │
│  (guided mode) │               │                           │
├────────────────┴────────────────┴───────────────────────────┤
│  Status Bar: Idle | / for commands | Ctrl+C exit             │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  watchdog   │   │   gpu_monitor   │   │    advisor      │
│  (training  │   │  (nvidia-smi   │   │  (Golden Rules  │
│   process)  │   │   polling)     │   │   + LLM)        │
└──────────────┘   └─────────────────┘   └─────────────────┘
        │
        ▼
┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  journal    │   │    config       │   │    settings     │
│  (JSONL     │   │  (config.yaml) │   │ (settings.yaml) │
│   rollback) │   └─────────────────┘   └─────────────────┘
└──────────────┘
```

## Two Modes

### Guided Mode (default)
- Edit training parameters via table interface
- Commands: `/setting safety_alpha=`, `/setting acc=`, etc.
- `/apply` to apply advisor recommendations
- `/save` to persist config changes to config.yaml

### Expert Mode
- Switch with `/expert`
- Edit train.py directly via `$EDITOR` (use `/edit-own`)
- Full control over training code

## Commands

Press `/` to enter command input mode:

| Command | Description |
|---------|-------------|
| `/setting safety_alpha=0.9` | Set safety headroom for OOM detection |
| `/setting acc=` | Set accuracy weight in advisor scoring |
| `/setting lat=` | Set latency weight |
| `/setting mem=` | Set memory weight |
| `/setting thr=` | Set throughput weight |
| `/setting energy=` | Set energy efficiency weight |
| `/language` | Toggle Chinese/English UI |
| `/create <name> <template>` | Create project (templates: resnet18, pytorch, tensorflow, lora, full-finetune) |
| `/open <project>` | Open existing project |
| `/plan <requirement>` | Generate implementation plan via LLM (saves to `<project>/plan.md`) |
| `/execute` | Autonomous implementation: reads plan.md, writes files, creates conda env |
| `/auto-complete <requirement>` | Auto-run full pipeline: plan→summary→execute→summary→runner→summary→tester→summary |
| `/runner` | Run training in Conda environment, stream output to output panel |
| `/checker` | Validate plan feasibility and code correctness via LLM |
| `/summary` | Generate structured summary (keywords, files, requirement, plan_path, conda_env, error_code) |
| `/tester` | Run tests and report model_overview + task_completed |
| `/teacher <topic>` | Explain ML algorithm (intro, pros, cons, use cases, variants) |
| `/status` | Show pipeline status (plan_exists, running, last_summary) |
| `/save` | Save config to config.yaml (journaled) |
| `/run` | Launch training under watchdog |
| `/apply` | Apply last recommendation |
| `/stop` | Stop training |
| `/expert` | Switch to expert mode |
| `/guided` | Switch to guided mode |
| `/rollback [n]` | Undo last n config changes |
| `/edit-own` | Open train.py in $EDITOR (expert mode) |
| `A` (capital) | Apply recommendation |
| `Ctrl+C` | Exit (once from normal mode, twice from command input) |

## Design Decisions

### Why ratatui-kit?
ratatui-kit provides a component-based reactive framework with state management similar to React hooks (`use_state`, `use_events`). The `#[component]` macro simplifies component authoring while the imperative `draw()` approach gives full control over rendering.

### Why Golden Rules RAG?
Rather than requiring an LLM for every recommendation, CoLoMo uses pre-encoded formulas:
- **Batch size**: `BS = floor(GPU_memory * alpha / model_size)` with alpha safety headroom
- **Learning rate**: Linear scaling `LR = base_lr * (BS / 32)`
- **Optimizer selection**: By parameter count (SGD < 1M < Adam < 1B < AdamW)
- **Gradient accumulation**: `steps = target_BS / actual_BS`

This gives instant, deterministic recommendations without API calls or rate limits.

### Why journal-based rollback?
Training configs change frequently. A JSONL journal records every `modify_config` and `apply_recommendation` event with old config snapshots. `/rollback` reads the inverse operations from journal.jsonl, enabling safe experimentation.

### Why Conda isolation?
All training runs execute under `conda run -n <env> ...` ensuring dependency isolation without shell activation complexity. Supports both `cuda` (python train.py) and `tilelang` (tilelang run) backends.

## Project Structure

```
rust_watchdog/
├── src/
│   ├── main.rs          # ratatui TUI + command routing
│   ├── lib.rs           # library exports
│   ├── config.rs        # config.yaml load/save
│   ├── settings.rs      # settings.yaml (weights, safety_alpha)
│   ├── journal.rs       # JSONL journal + rollback
│   ├── advisor.rs       # Golden Rules recommendation engine
│   ├── gpu_monitor.rs   # nvidia-smi polling
│   ├── watchdog.rs      # Training process supervisor (Conda spawn)
│   ├── summarizer.rs    # Log summarization
│   ├── monitor.rs       # System resource monitor
│   ├── env_check.rs     # Conda environment validation
│   ├── llm.rs           # LLM HTTP client (iflow API)
│   ├── llm_agent.rs     # Multi-agent: plan/execute/checker/summary/tester/runner/teacher
│   ├── agent_runner.rs  # Async agent spawn utilities
│   ├── project_creator.rs # Project scaffolding from templates
│   ├── tui.rs           # Legacy TUI module
│   └── tools/           # Agent tool definitions (file, bash, api tools)
├── projects/             # ML project workspaces
├── templates/            # Golden Rules + model templates
├── Cargo.toml
└── .env.example
```

## Getting Started

```bash
# Build
cargo build

# Run
cargo run

# In the TUI:
# - Press / to enter command mode
# - Type /setting acc=0.8 and press Enter
# - Press Ctrl+C to exit (once from normal mode)
```

## Multi-Agent Pipeline

CoLoMo implements a 6-stage autonomous pipeline:

| Stage | Command | Output |
|-------|---------|--------|
| Plan | `/plan <requirement>` | `.claude/plan/plan.md` |
| Execute | `/execute` | Writes files, creates conda env |
| Runner | `/runner` | Streams training output |
| Checker | `/checker` | Validates plan + code feasibility |
| Summary | `/summary` | Structured `summary:` line |
| Tester | `/tester` | Test results + `task_completed` |

Agents share context only via the `summary:` line — no inter-agent state.

## Future

- [ ] LLM tool use in execute agent (currently FILE:/CMD: parsing is basic)
- [ ] Template system integration with project creation
- [ ] gRPC/HTTP backend for web UI reuse
- [ ] Web UI backend (gRPC/HTTP from same watchdog)
