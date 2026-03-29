---
name: ml-training
description: This skill should be used when the user asks about ML training, PyTorch, hyperparameter tuning, batch size, learning rate, GPU memory, model fine-tuning, LoRA, QLoRA, LLM training (pretrain/SFT/RLHF), RAG, distributed training (DDP/ZeRO), mixed precision, Flash Attention, or algorithm explanations. Provides Golden Rules, 17 directly-usable code templates, and best practices.
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
RecommendedBatchSize = 0.90 × (GPU_Memory_MB / (ParamMem_MB + ActivationPerSample_MB × FeatureSize))
```
- `0.90` = **10% VRAM reserved** — prevents OOM from activation spikes, gradient buffers, multi-worker DataLoader
- Reduce if still OOM; increase if GPU utilization < 50%

**Recommendation:** Halve batch_size, then set `grad_accum_steps = ceil(old_batch / new_batch)` to preserve effective batch.

### Rule 2 — Learning Rate Scaling
**When batch_size changes:** `LR_new = LR_old × (BS_new / BS_old)`

**Effective batch:**
```
effective_batch_size = batch_size × gradient_accumulation_steps × num_gpus
```

### Rule 3 — Optimizer Selection
| Parameter Count | Optimizer | Weight Decay |
|---|---|---|
| < 10M | SGD + Momentum (0.9) | 5e-4 |
| 10M – 100M | Adam | 1e-4 |
| > 100M | AdamW | 0.01 |

### Rule 4 — Gradient Accumulation
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

## Templates as Tools (17 PyTorch Snippets)

All snippets are in `templates/pytorch-snippets/`. Use the Read tool to fetch the full code, then Write tool to add to the user's project.

### Training

#### `classification_train` — Standard PyTorch Training Loop
**When to use:** Starting a new image classification project; the minimal end-to-end skeleton.
**Key parameters:** `device`, `num_epochs`, `learning_rate`

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")
```

#### `grad_clip` — Gradient Clipping
**When to use:** Deep transformers, RNNs, or any training prone to gradient explosion.
**Key parameters:** `max_norm` (typically 0.5–2.0; 1.0 for transformers)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### Loss

#### `label_smoothing` — Label Smoothing Cross-Entropy
**When to use:** Noisy labels, multi-class classification where calibration matters.
**Key parameters:** `e` (smoothing factor, typically 0.1)

```python
import torch
import torch.nn as nn

class LSR(nn.Module):
    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        one_hot = torch.zeros(labels.size(0), classes)
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / (length - 1)
        return one_hot.to(target.device)

    def forward(self, x, target):
        if x.size(0) != target.size(0):
            raise ValueError(f"Expected batch size {x.size(0)} to match target batch {target.size(0)}")
        if x.dim() < 2:
            raise ValueError(f"Expected tensor with ≥2 dims, got {x.dim()}")
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(-x * smoothed_target, dim=1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        raise ValueError("reduction must be none, mean, or sum")
```

#### `label_smoothing_in_model` — In-Model Label Smoothing
**When to use:** Cleaner integration with custom models without changing the dataset.
**Key parameters:** `C` (num classes), smoothing factor 0.1

```python
# Inside your training loop over images/labels:
N = labels.size(0)
C = num_classes  # number of classes
smoothed_labels = torch.full(size=(N, C), fill_value=0.1 / (C - 1)).to(images.device)
smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=0.9)

score = model(images)
log_prob = torch.nn.functional.log_softmax(score, dim=1)
loss = -torch.sum(log_prob * smoothed_labels) / N
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

#### `custom_loss` — Custom nn.Module Loss
**When to use:** Complex training objectives, stateful losses (e.g., class-frequency-weighted CE).
**Key parameters:** Define your own `forward` logic.

```python
import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss = torch.mean((x - y) ** 2)
        return loss
```

---

### Augmentation

#### `mixup` — Mixup Data Augmentation
**When to use:** Improving generalization, reducing overfitting, adversarial robustness.
**Key parameters:** `alpha` (β distribution parameter, typically 0.2)

```python
beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()

    lambda_ = beta_distribution.sample([]).item()
    index = torch.randperm(images.size(0)).cuda()
    mixed_images = lambda_ * images + (1 - lambda_) * images[index, :]
    label_a, label_b = labels, labels[index]

    scores = model(mixed_images)
    loss = (lambda_ * loss_function(scores, label_a)
            + (1 - lambda_) * loss_function(scores, label_b))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

### Optimizer

#### `no_weight_decay_bias` — Exclude Bias from Weight Decay
**When to use:** Standard fine-tuning practice for BERT, ViT, and other pretrained models.
**Key parameters:** None — filters by name pattern `'bias'`

```python
bias_list = (p for n, p in model.named_parameters() if n[-4:] == 'bias')
others_list = (p for n, p in model.named_parameters() if n[-4:] != 'bias')
parameters = [
    {'params': bias_list, 'weight_decay': 0},
    {'params': others_list}
]
optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
```

#### `optimizer_chained_update` — Multi-Group Optimizer
**When to use:** Different layers need different learning rates or schedules.
**Key parameters:** `gamma` (decay factor), `step_size` (epochs between decay)

```python
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR, StepLR

optimizer = SGD(model.parameters(), lr=0.1)
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(4):
    optimizer.step()
    scheduler1.step()
    scheduler2.step()
    # Schedulers can be chained — alternating between them
```

#### `l1_regularization` — L1 Regularization
**When to use:** Feature selection, sparse models, model compression.
**Key parameters:** `lambda_` (regularization strength)

```python
l1_loss = torch.nn.L1Loss(reduction='sum')
loss = ...  # your task loss (e.g., cross-entropy)
for param in model.parameters():
    loss += lambda_ * torch.sum(torch.abs(param))
loss.backward()
```

---

### Scheduler

#### `lr_decay` — Learning Rate Decay
**When to use:** Prevent optimizer from overshooting near convergence.
**Key parameters:** `T_max` (total epochs), `patience` (ReduceLROnPlateau)

```python
# Reduce on plateau (monitors validation accuracy)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, verbose=True)
for epoch in range(num_epochs):
    train()
    val_acc = validate()
    scheduler.step(val_acc)

# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Step decay (decay 10× at epochs 50 and 70)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[50, 70], gamma=0.1)

# Warmup (linear increase over first 10 epochs)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda epoch: epoch / 10)
```

#### `get_current_lr` — Inspect Current Learning Rate
**When to use:** Logging current LR, warmup decisions, adaptive scheduling.
**Key parameters:** None — reads optimizer state directly

```python
# Single global LR:
lr = next(iter(optimizer.param_groups))['lr']

# Multiple LRs (different layers):
all_lr = [pg['lr'] for pg in optimizer.param_groups]
```

---

### IO

#### `checkpoint_save_load` — Save and Resume Training State
**When to use:** Long-running training, interruption recovery, best-model selection.
**Key parameters:** `best_acc` (tracked metric), `resume` (bool to load checkpoint)

```python
import os
import shutil
import torch

start_epoch = 0
best_acc = 0.0

if resume:
    model_path = os.path.join('model', 'best_checkpoint.pth.tar')
    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

for epoch in range(start_epoch, num_epochs):
    train()
    val_acc = validate()

    is_best = val_acc > best_acc
    best_acc = max(val_acc, best_acc)

    checkpoint = {
        'best_acc': best_acc,
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join('model', 'checkpoint.pth.tar'))
    if is_best:
        shutil.copy(
            os.path.join('model', 'checkpoint.pth.tar'),
            os.path.join('model', 'best_checkpoint.pth.tar'))
```

---

### Finetune

#### `finetune_fc` — Frozen Backbone + Train Head
**When to use:** Small dataset, strong pretrained backbone (ImageNet), fast iteration.
**Key parameters:** `lr` for head (typically 1e-2 to 1e-3)

```python
import torchvision
import torch.nn as nn

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, num_classes)  # Replace classifier head

optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
```

#### `finetune_fc_high_lr_conv_low_lr` — Differential Learning Rates
**When to use:** Medium dataset, backbone should adapt slightly to new domain. Standard BERT fine-tuning.
**Key parameters:** head LR (e.g., 1e-3), backbone LR (e.g., 1e-5)

```python
import torchvision

model = torchvision.models.resnet18(pretrained=True)
finetuned_params = list(map(id, model.fc.parameters()))
conv_params = (p for p in model.parameters() if id(p) not in finetuned_params)

parameters = [
    {'params': conv_params, 'lr': 1e-5},       # lower LR — preserve pretrained
    {'params': model.fc.parameters(), 'lr': 1e-3},  # higher LR — learn new task
]
optimizer = torch.optim.SGD(parameters, momentum=0.9, weight_decay=1e-4)
```

---

### Feature Extraction

#### `extract_imagenet_layer_feature` — Single-Layer Feature Hook
**When to use:** Transfer learning, similarity search, frozen feature extraction.
**Key parameters:** `layer` (e.g., `'layer4'` for ResNet, `'features[-1]'` for VGG)

```python
import collections
import torchvision

# VGG-16 relu5-3 feature (before final pooling):
model = torchvision.models.vgg16(pretrained=True).features[:-1]

# VGG-16 pool5 feature:
model = torchvision.models.vgg16(pretrained=True).features

# VGG-16 fc7 feature:
model = torchvision.models.vgg16(pretrained=True)
model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])

# ResNet GAP feature (remove final fc):
model = torchvision.models.resnet18(pretrained=True)
model = torch.nn.Sequential(collections.OrderedDict(list(model.named_children())[:-1]))

with torch.no_grad():
    model.eval()
    conv_representation = model(image)
```

#### `extract_imagenet_multi_conv_features` — Multi-Layer Feature Extraction
**When to use:** Fine-grained recognition, multi-scale representations.
**Key parameters:** `layers_to_extract` (set of layer names)

```python
import collections
import torchvision
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model, layers_to_extract):
        super().__init__()
        self._model = pretrained_model
        self._model.eval()
        self._layers_to_extract = set(layers_to_extract)

    def forward(self, x):
        with torch.no_grad():
            conv_representation = []
            for name, layer in self._model.named_children():
                x = layer(x)
                if name in self._layers_to_extract:
                    conv_representation.append(x)
            return conv_representation

model = torchvision.models.resnet152(pretrained=True)
model = nn.Sequential(collections.OrderedDict(list(model.named_children())[:-1]))

extractor = FeatureExtractor(
    pretrained_model=model,
    layers_to_extract={'layer1', 'layer2', 'layer3', 'layer4'}
)
features = extractor(image)  # list of tensors from each layer
```

---

### Visualization

#### `train_visualization` — TensorBoard Live Plots
**When to use:** Any training — catches divergence early, compares hyperparameters.
**Key parameters:** `log_dir` (TensorBoard log directory)

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter(log_dir='runs')

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test',  np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test',  np.random.random(), n_iter)
# Launch tensorboard: tensorboard --logdir=runs
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
| ZeRO-3 | All (params + grads + optimizer) | ~64x |

### Mixed Precision
| Precision | Memory | Speed | Accuracy Loss |
|-----------|--------|-------|-------------|
| FP32 | 1x | 1x | None |
| FP16 | 0.5x | ~2x | Possible |
| BF16 | 0.5x | ~2x | Near zero |

**BF16 is preferred for LLM training** — same exponent range as FP32.

### Flash Attention
```python
model = AutoModel.from_pretrained(
    "model_name",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
```

---

## LLM Training

### 3-Stage Pipeline
```
Pretraining (raw text, T tokens) → SFT (instruction pairs, 10K-100K) → RLHF (human feedback)
```

### LoRA Fine-tuning
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                           # rank (4-16)
    lora_alpha=16,                 # scaling = lora_alpha / r
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
- Base model: 4-bit NF4 quantization via `bitsandbytes`
- LoRA adapters on top
- 65B model fits on single 48GB GPU
- Use `AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)`

### When to Use What
| Scenario | Approach |
|---|---|
| Single GPU, < 7B params | Full fine-tune or LoRA |
| Single GPU, > 7B params | QLoRA |
| Multi-GPU, > 70B params | DeepSpeed ZeRO-3 + LoRA |
| Fast iteration, many tasks | LoRA |
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
| Semantic (by paragraph) | Preserves meaning |
| Overlapping | Captures cross-chunk context |

### Vector DB Selection
| DB | Best for |
|---|---|
| ChromaDB | Fast prototyping |
| Faiss | Single-node, large scale |
| Milvus | Production, distributed |
| Qdrant | Hybrid search, high precision |

---

## Algorithm Explanations

When asked to explain an ML algorithm, provide:
1. **What it is** — one-sentence definition
2. **How it works** — key mechanism with formulas
3. **When to use** — appropriate scenarios
4. **Pros** — advantages
5. **Cons** — limitations and tradeoffs
6. **Variants** — related approaches

### Example: Mixup
**What:** Data augmentation by linearly interpolating pairs of samples and their labels.

**Formula:** `x̂ = λ·xᵢ + (1-λ)·xⱼ`, `ŷ = λ·yᵢ + (1-λ)·yⱼ` where `λ ~ Beta(α, α)`

**When:** Image classification, generalization improvement, adversarial robustness.

**Pros:** Improves generalization, robustness to adversarial examples, reduces memorization.

**Cons:** Soft labels don't align with hard-label metrics; requires tuning α.

**Variants:** CutMix (paste patches), Manifold Mixup (interpolate hidden features).

### Example: LoRA
**What:** Low-rank adapter that freezes pretrained weights and adds trainable `B·A` matrices.

**Formula:** `W = W₀ + BA` where `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×k)`, `r << min(d,k)`

**When:** Single GPU, large models (>1B params), multiple task adapters.

**Pros:** Train 0.1-1% of params; zero inference overhead; hot-swappable adapters.

**Cons:** Lower rank limits expressiveness; less expressive than full fine-tune.

**Variants:** QLoRA (quantized base), DoRA (weight decomposition), AdaLoRA (adaptive rank).
