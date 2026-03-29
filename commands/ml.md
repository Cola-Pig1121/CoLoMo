---
description: ML training assistant — plan, run, test, and explain PyTorch / LLM training workflows. Delegates to the CoLoMo subagent for full ML workflows: Golden Rules hyperparameter tuning, LoRA/QLoRA fine-tuning, batch size optimization (10% VRAM reserved).
---

# /ml — ML Training Command

Invoke the **CoLoMo** agent for autonomous ML training assistance.

## Usage

```
/ml <subcommand> [args...]
```

## Subcommands

| Subcommand | Description |
|-----------|-------------|
| `detect` | Detect GPU/CPU config, show Golden Rules recommendations |
| `plan <requirement>` | Detect → Calculate → Generate plan.md + config.yaml |
| `execute` | **Check env → Install deps → Generate code from plan** |
| `advise` | GPU-based hyperparameter recommendations |
| `run` | Run training in the project's Conda environment |
| `test` | Run pytest in the project's Conda environment |
| `explain <topic>` | Explain an ML algorithm (teacher mode) |
| `rollback [n]` | Undo the last N config changes |

## Workflow

```
Requirement
    ↓
/ml plan       → Detect GPU → Calculate params → Generate plan.md + config.yaml
    ↓
/ml execute    → Check env → Install deps → Generate code from plan
    ↓
/ml run        → conda run -n <env> python train.py
    ↓
/ml test       → conda run -n <env> pytest
```

## 1. Detect

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
nproc && free -h | grep Mem
```

## 2. Plan (`/ml plan`)

1. Detect GPU/CPU config
2. Calculate batch_size, lr, optimizer via Golden Rules
3. Generate `plan.md` with tasks marked `[ ]`
4. Generate `config.yaml`
5. Return summary

## 3. Execute (`/ml execute`)

**Pre-execution checks (stop on first failure):**

```
1. Check plan.md exists?       → Error if missing, prompt /ml plan first
2. Detect conda location?        → Ask user: install conda or use system python?
3. Check conda env exists?       → Ask user: create env or use existing?
4. Check Python deps installed?  → pip install -r requirements.txt
5. Execute plan tasks in order
```

### Conda Not Found → Ask User

If `conda` is not in PATH, ask user:

```
[ERROR] Conda not found in PATH.
  Detected Python: <path>
  Miniconda recommended for ML projects.

  Options:
    [1] Install Miniconda (recommended) — I'll run the installer
    [2] Use system Python — continue without conda
    [3] Cancel

  Your choice:
```

If user chooses `[1]`: Run Miniconda install commands.
If user chooses `[2]`: Skip conda env creation, use `python` directly.

### Conda Env Missing → Ask User

If `conda env list` does not show the env from `config.yaml`:

```
[ERROR] Conda environment "mnist" not found.
  config.yaml specifies: conda_env: mnist

  Options:
    [1] Create environment "mnist" — conda create -n mnist python=3.11
    [2] Use existing environment — specify name
    [3] Cancel

  Your choice:
```

## 4. Advise (Golden Rules)

- `BS = 0.90 × VRAM_Available / (param_mem + activation_mem)`
- `LR_new = LR_old × (BS_new / BS_old)`
- Optimizer: SGD (<10M), Adam (10M–100M), AdamW (>100M)

## Examples

```
/ml plan fine-tune Llama2 with QLoRA on a single GPU
/ml execute       # execute plan.md step by step
/ml detect
/ml advise
/ml run
/ml test
/ml rollback 2
```

## Related

- Skill: `ml-training` (Golden Rules, 17 PyTorch snippets, LoRA/QLoRA, RAG)
- Agent: `colomo` (full autonomous training subagent)
- Rules: `rules/ml/` (PyTorch conventions, Golden Rules patterns)
