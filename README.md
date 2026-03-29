# CoLoMo — ML Training Plugin for Claude Code

**CoLoMo** is a Claude Code plugin that provides ML training knowledge: Golden Rules for hyperparameter tuning, PyTorch code patterns, LLM fine-tuning guidance (LoRA/QLoRA/SFT/RLHF), and autonomous training agent workflows.

---

## What This Plugin Provides

### 1. ML Training Skill (`skills/ml-training/SKILL.md`)

ML knowledge triggered automatically when discussing training topics:

- **Golden Rules**: Batch size formulas, learning rate linear scaling, optimizer selection by parameter count
- **PyTorch Patterns**: Training loops, gradient clipping, label smoothing, mixup, checkpointing, fine-tuning
- **Distributed Training**: DDP, DeepSpeed ZeRO stages, BF16/FP16 mixed precision, Flash Attention
- **LLM Fine-tuning**: 3-stage pipeline (pretrain → SFT → RLHF), LoRA/QLoRA configuration
- **RAG**: Chunking strategies, vector DB selection, hybrid retrieval

### 2. CoLoMo Agent (`agents/colomo.md`)

Autonomous ML training subagent — use it when the user wants to go from a requirement to a trained model.

Commands:
- `/ml plan <requirement>` — generate implementation plan
- `/ml run` — run training in Conda environment
- `/ml test` — run pytest in Conda environment
- `/ml advise` — GPU-based hyperparameter recommendations
- `/ml explain <topic>` — explain ML algorithm
- `/ml rollback [n]` — undo config changes

### 3. ML Rules (`rules/ml/`)

- **`coding-style.md`**: PyTorch conventions (device management, gradient handling, no in-place ops on pretrained weights, mixed precision)
- **`patterns.md`**: Golden Rules formulas, fine-tuning patterns (frozen backbone / differential LR / LoRA / QLoRA), augmentation, distributed training

---

## Installation

### Via Claude Code Marketplace (Recommended)

```bash
# 1. Add this GitHub repo as a marketplace
/plugin marketplace add Cola-Pig1121/CoLoMo

# 2. Install the plugin from that marketplace
/plugin install colomo@Cola-Pig1121/CoLoMo
```

> Requires Claude Code **v1.0.33+**. Alternatively, open `/plugin` in Claude Code and browse the **Marketplaces** tab to add this repo.

---

## Golden Rules Quick Reference

| Signal | Rule | Action |
|--------|------|--------|
| CUDA OOM | `BS = α × GPU_mem / (param_mem + activation_mem)` | Halve batch_size |
| GPU util < 50% | — | Increase batch by 25% |
| Batch changed | `LR_new = LR_old × (BS_new / BS_old)` | Scale LR linearly |
| Params > 100M | — | Use AdamW |
| Params 10M–100M | — | Use Adam |
| Params < 10M | — | Use SGD |
| Batch reduced | — | Set `grad_accum_steps = ceil(old / new)` |

Default `α` (safety_alpha) = **0.90** — reserve 10% VRAM headroom.

---

## PyTorch Snippet Index

| Snippet | Category | Use when |
|---------|----------|----------|
| `label_smoothing` | Loss | Noisy labels, calibration |
| `mixup` | Augmentation | Generalization, adversarial robustness |
| `grad_clip` | Training | Deep transformers, RNNs |
| `lr_decay` | Scheduler | Cosine or step decay |
| `checkpoint_save_load` | IO | Resuming after interruption |
| `finetune_fc` | Finetune | Frozen backbone + train head |
| `finetune_fc_high_lr_conv_low_lr` | Finetune | Differential LR (BERT style) |
| `no_weight_decay_bias` | Optimizer | Standard fine-tuning practice |
| `extract_imagenet_layer_feature` | Feature | Transfer learning |
| `train_visualization` | Vis | Loss/accuracy plots |

---

## Architecture

```
CoLoMo/
├── .claude-plugin/
│   ├── plugin.json          # Plugin manifest
│   └── marketplace.json      # Marketplace discovery manifest
├── skills/
│   └── ml-training/
│       └── SKILL.md        # ML knowledge base (17 snippets + Golden Rules)
├── agents/
│   └── colomo.md           # CoLoMo subagent
├── rules/
│   └── ml/
│       ├── coding-style.md # PyTorch conventions
│       └── patterns.md      # Golden Rules + patterns
├── templates/
│   ├── pytorch-snippets/    # 17 standalone PyTorch snippets
│   ├── model_templates/     # Full project templates (pytorch-template)
│   └── docs/               # Algorithm references (LoRA, RAG, etc.)
├── docs/
│   ├── CONTRIBUTING.md     # Development guide
│   └── SETUP.md           # Configuration reference
└── CLAUDE.md              # Claude Code guidance
```

---

## Fine-tuning Decision Tree

```
Is model > 1B params and single GPU?
├─ YES → QLoRA (4-bit NF4 base + LoRA adapters)
└─ NO
   ├─ Want maximum quality?
   │  ├─ YES → Full fine-tune or RLHF
   │  └─ NO
   │     └─ LoRA (r=8–16, single GPU, ~0.1–1% trainable params)
   └─ Have pretrained backbone?
      ├─ YES
      │  ├─ Small dataset → Frozen backbone + train head only
      │  └─ Medium dataset → Differential LR (high head / low backbone)
      └─ NO → Train from scratch
```

---

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for development guidelines.
