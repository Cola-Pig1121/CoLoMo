# ML Training Patterns

> This file extends [common/patterns.md](../common/patterns.md) with ML-specific patterns.

## Golden Rules: Hyperparameter Formulas

### GPU Memory Budget
```
RecommendedBatchSize = α × (GPU_Memory_MB / (ParamMem_MB + ActivationPerSample_MB × FeatureSize))
```
- Default `α = 0.85` (safety headroom)
- `ParamMem_MB ≈ param_count × 4 bytes` (FP32) or `× 2 bytes` (FP16/BF16)
- `ActivationPerSample_MB ≈ (2 × layers × hidden_size² + 4 × layers × hidden_size × vocab_size) / batch_size`

### Learning Rate Linear Scaling
```
LR_new = LR_old × (BS_new / BS_old)
```
When changing batch size, scale LR proportionally. Effective batch:
```
effective_batch = batch_size × grad_accum_steps × num_gpus
```

### Optimizer Selection by Scale
| Params | Optimizer | Weight Decay | Notes |
|--------|-----------|-------------|-------|
| < 10M | SGD + Momentum(0.9) | 5e-4 | Simple tasks |
| 10M – 100M | Adam | 1e-4 | General |
| > 100M | AdamW | 0.01 | LLM scale |

## Fine-tuning Patterns

### Frozen Backbone + Train Head
```python
# Freeze all backbone layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Only classifier is trainable
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
```
**When:** Small dataset, strong pretrained features, fast iteration.

### Differential LR (Backbone Low / Head High)
```python
backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
head_params = [p for n, p in model.named_parameters() if 'head' in n]
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},   # low LR — preserve pretrained
    {'params': head_params, 'lr': 1e-3},        # high LR — learn task
], weight_decay=0.01)
```
**When:** Medium dataset, backbone should adapt slightly to new domain. Standard BERT fine-tuning practice.

### LoRA (Low-Rank Adaptation)
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                           # rank: 4-16 (higher = more expressive)
    lora_alpha=16,                 # scaling = lora_alpha / r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
```
**When:** Single GPU, large models (>1B params), multiple tasks needing different adapters.

### QLoRA
- Base model: 4-bit NF4 quantization via `bitsandbytes`
- LoRA adapters applied on top
- 65B model fits on 48GB GPU
- Use `AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)`

## Augmentation Patterns

### Mixup
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
**When:** Improve generalization, reduce overfitting on image classification.

### Label Smoothing
```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return confidence * nll_loss + self.smoothing * smooth_loss
```
**When:** Noisy labels, classification tasks where calibration matters.

## Distributed Training Patterns

### DDP Wrapper Pattern
```python
# Spawn with: torchrun --nproc_per_node=N train.py
model = MyModel()
model = model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

# In training loop: model() returns loss; DDP averages gradients automatically
output = model(input)
loss.backward()  # gradients synced across GPUs
```

### ZeRO Stage Selection
| Stage | Use when |
|-------|----------|
| ZeRO-1 | Memory-constrained single node |
| ZeRO-2 | Multi-node, moderate memory savings needed |
| ZeRO-3 | Multi-node, 70B+ models, can tolerate higher communication |

## LLM Training Patterns

### 3-Stage Pipeline
```
Pretraining (raw text, T tokens) → SFT (instruction pairs, 10K-100K) → RLHF (human feedback)
```

### Attention Mechanism Variants
| Type | K/V Heads | Memory | Quality |
|------|-----------|--------|---------|
| MHA | full | High | Best |
| GQA | shared across Q groups | Medium | Near MHA |
| MQA | single K/V | Low | Degraded |

**LLaMA2 uses GQA** — balances quality and KV-cache efficiency.

### Position Encoding
- **RoPE** (LLaMA, Mistral): Rotary, encodes relative position via Q/K rotation
- **ALiBi**: Linear bias on attention scores, no embedding needed
- **Sinusoidal**: Fixed, learnable absolute positions (GPT-2)

## RAG Patterns

### Chunking Strategy
```python
# Fixed-length (simple, fast)
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Overlapping (reduces context cuts)
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

# Semantic (by sentence/paragraph boundary — use NLP library)
import nltk; paragraphs = nltk.sent_tokenize(text)
```

### Hybrid Retrieval
```python
# Combine dense (embedding) + sparse (BM25) retrieval
dense_results = vector_db.similarity_search(query, k=5)
sparse_results = bm25.search(query, k=5)
# Rerank combined results with cross-encoder
```
