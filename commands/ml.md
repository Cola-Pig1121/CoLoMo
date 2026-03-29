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
| `plan <requirement>` | **Detect GPU → Calculate params → Generate plan** |
| `detect` | Detect GPU/CPU config and show Golden Rules recommendations |
| `advise` | GPU-based hyperparameter recommendations (Golden Rules) |
| `run` | Run training in the project's Conda environment |
| `test` | Run pytest in the project's Conda environment |
| `explain <topic>` | Explain an ML algorithm (teacher mode) |
| `rollback [n]` | Undo the last N config changes |

## Workflow: `/ml plan`

**Before planning, always detect system config first:**

```
1. Detect → nvidia-smi (GPU name, VRAM, utilization)
2. Calculate → Golden Rules formula
3. Plan → Generate plan.md with computed parameters
```

**Golden Rules formula (safety_alpha = 0.90):**
```
RecommendedBatchSize = 0.90 × (GPU_Memory_MB / (ParamMem_MB + ActivationPerSample_MB))
```

## Examples

```
# Full workflow: detect + plan
/ml plan fine-tune Llama2 with QLoRA on a single GPU

# Just see GPU info and recommendations
/ml detect

# Ask for advice on current config
/ml advise

# Explain an algorithm
/ml explain LoRA

# Rollback config changes
/ml rollback 2
```

## What This Does

### 1. Detect (`/ml detect` or automatic on `/ml plan`)
- Runs `nvidia-smi` to get GPU name, VRAM total/used, utilization
- Runs `nproc` and `free -h` for CPU/RAM info
- Shows all detected hardware

### 2. Calculate (Golden Rules)
- `BS = 0.90 × VRAM_Available / (param_mem + activation_mem)`
- `LR_new = LR_old × (BS_new / BS_old)`
- Optimizer: SGD (<10M params), Adam (10M–100M), AdamW (>100M)

### 3. Plan (`/ml plan`)
1. Detect system config
2. Calculate hyperparameters using Golden Rules
3. Generate `plan.md` with computed `batch_size`, `lr`, `optimizer`
4. Generate `config.yaml` with those parameters
5. Return structured summary

## Related

- Skill: `ml-training` (Golden Rules, 17 PyTorch snippets, LoRA/QLoRA, RAG)
- Agent: `colomo` (full autonomous training subagent)
- Rules: `rules/ml/` (PyTorch conventions, Golden Rules patterns)
