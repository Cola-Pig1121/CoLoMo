# CoLoMo Configuration

CoLoMo is a Claude Code plugin — no runtime configuration required. This document describes the configuration schemas referenced by the plugin's knowledge base.

## Plugin Configuration

CoLoMo plugin is enabled via Claude Code's settings. No additional configuration needed.

## Hyperparameter Configuration Reference

These schemas are referenced by the Golden Rules and advice in the plugin.

### Batch Size (Golden Rule)

```
RecommendedBatchSize = α × (GPU_Memory_MB / (ParamMem_MB + ActivationPerSample_MB × FeatureSize))
```

| Parameter | Description |
|-----------|-------------|
| `α` (safety_alpha) | Headroom factor, default 0.85 (15% reserved) |
| `GPU_Memory_MB` | Total GPU VRAM in MB |
| `ParamMem_MB` | `param_count × 4` (FP32) or `× 2` (BF16/FP16) |
| `ActivationPerSample_MB` | Activation memory per sample |

### Optimizer Selection

| Parameter Count | Optimizer | Weight Decay |
|----------------|-----------|-------------|
| < 10M | SGD + Momentum (0.9) | 5e-4 |
| 10M – 100M | Adam | 1e-4 |
| > 100M | AdamW | 0.01 |

### Learning Rate Linear Scaling

When batch size changes: `LR_new = LR_old × (BS_new / BS_old)`

### Gradient Accumulation

```python
effective_batch = batch_size × grad_accum_steps
grad_accum_steps = ceil(target_batch / actual_batch)
```

### LoRA Configuration

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `r` | 4–16 | Rank of low-rank matrices |
| `lora_alpha` | `2 × r` | Scaling factor |
| `lora_dropout` | 0.05–0.1 | Dropout probability |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Layers to adapt |

### Mixed Precision

| Precision | Memory | Speed | Accuracy Loss |
|-----------|--------|-------|-------------|
| FP32 | 1x | 1x | None |
| FP16 | 0.5x | ~2x | Possible |
| BF16 | 0.5x | ~2x | Near zero |

**BF16 is preferred for LLM training** — same exponent range as FP32.

### DeepSpeed ZeRO Stages

| Stage | What is Sharded | Memory Savings |
|-------|----------------|----------------|
| ZeRO-1 | Optimizer states | ~4x |
| ZeRO-2 | Optimizer states + gradients | ~8x |
| ZeRO-3 | All (params + grads + optimizer) | ~64x |

## Directory Structure

```
CoLoMo/
├── .claude-plugin/
│   └── plugin.json
├── skills/ml-training/
│   └── SKILL.md              # ML knowledge base
├── agents/
│   └── colomo.md            # CoLoMo agent
├── rules/ml/
│   ├── coding-style.md      # PyTorch conventions
│   └── patterns.md          # Golden Rules + patterns
└── docs/
    ├── CONTRIBUTING.md       # Development guide
    └── SETUP.md             # This file
```
