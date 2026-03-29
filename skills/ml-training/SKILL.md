---
name: ml-training
description: This skill should be used when the user asks about ML training, PyTorch, hyperparameter tuning, batch size, learning rate, GPU memory, model fine-tuning, LoRA, QLoRA, LLM training (pretrain/SFT/RLHF), RAG, distributed training (DDP/ZeRO), mixed precision, Flash Attention, or algorithm explanations. Provides Golden Rules, code templates, and best practices.
version: 1.0.0
---

# ML Training Skill

Comprehensive guidance for machine learning training — from single-GPU PyTorch loops to distributed LLM fine-tuning.

---

## Golden Rules: Hyperparameter Tuning

### Rule 1 — GPU Memory Overflow
**Condition:** CUDA OOM error OR `nvidia-smi` shows memory_used / memory_total > 0.9

**Formula:**
```
RecommendedBatchSize = α × (GPU_Memory_MB / (ParamMem_MB + ActivationPerSample_MB × FeatureSize))
```
- Default `α` (safety_alpha) = **0.85** — reserves 15% headroom
- Reduce if still OOM; increase if GPU utilization < 50%

**Recommendation:** Halve batch_size, then set `grad_accum_steps = ceil(old_batch / new_batch)` to preserve effective batch.

### Rule 2 — Learning Rate Scaling
**When batch_size changes:** `LR_new = LR_old × (BS_new / BS_old)`

**Effective batch:**
```
effective_batch_size = batch_size × gradient_accumulation_steps × num_gpus
```

### Rule 3 — Optimizer Selection
| Parameter Count | Optimizer | Why |
|---|---|---|
| < 10M | SGD + Momentum | Simple tasks, less prone to overfitting |
| 10M – 100M | Adam | Good for moderate-scale models |
| > 100M | AdamW | Decoupled weight decay prevents instability at scale |

### Rule 4 — Grad Accumulation
When GPU can't fit a larger batch:
```python
effective_batch = batch_size × grad_accum_steps
# optimizer.step() called every grad_accum_steps forward passes
```

### Rule 5 — Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Prevents exploding gradients, especially in transformers/RNNs
```

### Quick Reference

| Situation | Action |
|---|---|
| CUDA OOM | Halve batch_size; set grad_accum_steps |
| GPU util < 50% | Increase batch_size by 25% |
| Batch changed | Linearly scale learning rate |
| Params > 100M | Use AdamW |
| Deep transformer | Apply gradient clipping + warmup |
| Fine-tuning | Use bias no-decay + lower LR for backbone |

---

## PyTorch Training Patterns

### Standard Classification Loop
```python
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        x, y = batch
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

### Gradient Accumulation
```python
model.train()
for i, batch in enumerate(dataloader):
    x, y = batch
    loss = criterion(model(x), y) / grad_accum_steps
    loss.backward()
    if (i + 1) % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Checkpoint Save/Load
```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

### Label Smoothing Loss
```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
```

### No Weight Decay on Bias (standard fine-tuning practice)
```python
decay_params = [p for n, p in model.named_parameters() if 'bias' not in n]
no_decay_params = [p for n, p in model.named_parameters() if 'bias' in n]
optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=lr)
```

### Fine-tune: Frozen Backbone + Train Head
```python
for param in model.backbone.parameters():
    param.requires_grad = False
# Only classifier head is trainable (~1-2% of params)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```

### Fine-tune: Differential LR (backbone low / head high)
```python
backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
head_params = [p for n, p in model.named_parameters() if 'head' in n]
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},   # lower LR for pretrained backbone
    {'params': head_params, 'lr': 1e-3},        # higher LR for new head
])
```

### Mixup Augmentation
```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

### LR Scheduler: Cosine Annealing with Warmup
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=1e-6)
# Or manual warmup:
if global_step < warmup_steps:
    lr = base_lr * global_step / warmup_steps
```

---

## Distributed Training

### DDP (Data Distributed Parallel)
```python
# Launch: torchrun --nproc_per_node=N train.py
import torch.distributed as dist
dist.init_process_group(backend='nccl')
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
# Each GPU runs a full copy; gradients synced every step
```

### DeepSpeed ZeRO Stages
| Stage | What is Sharded | Memory Savings |
|-------|----------------|----------------|
| ZeRO-1 | Optimizer states | ~4x |
| ZeRO-2 | Optimizer states + gradients | ~8x |
| ZeRO-3 | All (params + grads + optimizer) | ~64x (linear) |

```python
# ds_config.json
{
  "zero_optimization": {"stage": 2},
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0
}
# In code:
training_args = TrainingArguments(deepspeed="ds_config.json", ...)
```

### Mixed Precision
| Precision | Memory | Speed | Accuracy |
|-----------|--------|-------|----------|
| FP32 | 1x | 1x | No loss |
| FP16 | 0.5x | ~2x | May degrade |
| BF16 | 0.5x | ~2x | Nearly lossless |

**BF16 is preferred for LLM training** — same exponent range as FP32, avoids gradient explosion/NaN.

### Flash Attention
```python
# Transformers
model = AutoModel.from_pretrained(
    "model_name",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
# Reduces attention memory from O(N²) to O(N) — critical for long sequences
```

---

## LLM Training

### 3-Stage Pipeline
1. **Pretraining** — Causal LM on massive raw text (T级别 token); goal: learn language
2. **SFT** — Supervised Fine-Tuning on instruction-response pairs (10K-100K); goal: follow instructions
3. **RLHF** — Reward Model + PPO on human feedback; goal: align with human preferences

### LoRA Fine-tuning
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                          # rank (4-16; higher = more params, less compression)
    lora_alpha=16,                # scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # typically 0.1-1% of total
```

### QLoRA (Quantized LoRA)
- Quantize base model to NF4 (4-bit NormalFloat)
- Apply LoRA on the quantized model
- 65B model fits on single 48GB GPU
- Use `bitsandbytes` + `peft`

### When to use what
| Scenario | Approach |
|---|---|
| Single GPU, < 7B params | Full fine-tune or LoRA |
| Single GPU, > 7B params | QLoRA |
| Multi-GPU, > 70B params | DeepSpeed ZeRO-3 + LoRA |
| Fast iteration, many tasks | LoRA (switch adapters per task) |
| Maximum quality | Full fine-tune or RLHF |

---

## RAG (Retrieval-Augmented Generation)

### Core Pipeline
```
Document → Chunk → Embed → Vector DB
Query → Embed → ANN Search → Top-K Chunks → LLM Generate
```

### Chunking Strategies
| Strategy | When to use |
|---|---|
| Fixed length (512 tokens) | Simple, fast; may cut mid-sentence |
| Semantic (by paragraph) | Preserves meaning; more complex |
| Overlapping | Captures cross-chunk context |

### Vector DB Selection
| DB | Best for |
|---|---|
| ChromaDB | Fast prototyping, embedded |
| Faiss | Single-node, large scale |
| Milvus | Production, distributed |
| Qdrant | Hybrid search, high precision |

---

## Algorithm Explanations (Teacher Mode)

When `/ml explain <topic>` is invoked, provide:

### Structure
1. **What it is** — one-sentence definition
2. **How it works** — key mechanism, with formulas
3. **When to use** — appropriate scenarios
4. **Pros** — advantages over alternatives
5. **Cons** — limitations, tradeoffs
6. **Variants** — related approaches

### Example: Mixup
**What:** Data augmentation by linearly interpolating pairs of samples and their labels.

**Formula:** `x̂ = λ·xᵢ + (1-λ)·xⱼ`, `ŷ = λ·yᵢ + (1-λ)·yⱼ` where `λ ~ Beta(α, α)`

**Pros:** Improves generalization, robustness to adversarial examples, reduces memorization.

**Cons:** Soft labels don't align with hard-label metrics; requires tuning α.

**Variants:** CutMix (paste patches instead of blending), Cutout, Manifold Mixup.

### Example: LoRA
**What:** Low-rank adapter that freezes pretrained weights and adds trainable `B·A` matrices.

**Formula:** `W = W₀ + BA` where `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×k)`, `r << min(d,k)`

**Pros:** Train 0.1-1% of params; zero inference overhead (merged into W); hot-swappable adapters.

**Cons:** Lower rank limits complexity of learnable transformations; less expressive than full fine-tune.

**Variants:** QLoRA (quantized base), DoRA (weight decomposition direction), AdaLoRA (adaptive rank).
