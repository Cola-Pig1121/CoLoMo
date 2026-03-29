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
| `plan <requirement>` | Generate an implementation plan from a training requirement |
| `run` | Run training in the project's Conda environment |
| `test` | Run pytest in the project's Conda environment |
| `advise` | GPU-based hyperparameter recommendations (Golden Rules) |
| `explain <topic>` | Explain an ML algorithm (teacher mode) |
| `rollback [n]` | Undo the last N config changes |

## Examples

```
/ml plan fine-tune Llama2 with QLoRA on a single GPU

/ml advise

/ml explain LoRA

/ml rollback 2
```

## What This Does

The CoLoMo agent:

1. **Plan** (`/ml plan`) — analyzes requirements, generates a structured markdown plan with tasks
2. **Advise** (`/ml advise`) — applies Golden Rules:
   - `BS = 0.90 × GPU_mem / (param_mem + activation_mem)` → auto batch size
   - `LR_new = LR_old × (BS_new / BS_old)` → linear LR scaling
   - Optimizer: SGD (<10M), Adam (10M–100M), AdamW (>100M)
3. **Run** (`/ml run`) — streams training stdout/stderr via `conda run`
4. **Test** (`/ml test`) — runs pytest in the project's Conda environment
5. **Explain** (`/ml explain`) — teacher mode with formulas, pros/cons, variants
6. **Rollback** (`/ml rollback`) — restores `config.yaml` from journal snapshots

## Related

- Skill: `ml-training` (Golden Rules, 17 PyTorch snippets, LoRA/QLoRA, RAG)
- Agent: `colomo` (full autonomous training subagent)
- Rules: `rules/ml/` (PyTorch conventions, Golden Rules patterns)
