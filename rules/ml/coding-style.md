# ML Training Coding Style

> This file extends [common/coding-style.md](../common/coding-style.md) with ML-specific content.

## PyTorch Conventions

### Device Management
```python
# ALWAYS use .to(device) explicitly — never rely on default CPU placement
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)

# For multi-GPU: use DDP wrapper AFTER .to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

### Model Structure
```python
# GOOD — nn.Module with clear forward
class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.head = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

# BAD — in-place modification of pretrained backbone without freezing
for param in model.backbone.parameters():
    param.data -= param.data * weight_decay  # never in-place!
```

### Gradient Handling
```python
# GOOD — zero_grad before backward
optimizer.zero_grad()
loss.backward()
optimizer.step()

# GOOD — clip before step
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# BAD — forget to zero_grad (gradient accumulation across steps)
loss.backward()  # without zero_grad → wrong gradients
optimizer.step()
```

### No In-Place Operations on Pretrained Weights
```python
# BAD — in-place ReLU changes the pretrained model behavior
self.conv = nn.Conv2d(3, 64, 3)
self.relu = nn.ReLU(inplace=True)  # modifies original weights in-place

# GOOD — non-in-place (standard practice)
self.relu = nn.ReLU(inplace=False)
```

### Checkpointing
```python
# GOOD — save everything needed to resume
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_metric': best_metric,
}, f'checkpoint_epoch_{epoch}.pth')

# Load with strict=False when model architecture may change
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

## Data Loading

### Always Use Reproducible Seeds
```python
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

DataLoader(dataset, worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(42))
```

### Pin Memory for GPU Training
```python
# Pin memory speeds up host→GPU transfers
DataLoader(dataset, pin_memory=True)  # always for GPU training
```

## Mixed Precision
```python
# Use BF16 for LLM training (safer than FP16)
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

## Logging & Monitoring

```python
# GOOD — log both loss and learning rate
if global_step % log_every == 0:
    writer.add_scalar('train/loss', loss.item(), global_step)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

# Validate every N epochs, not every step
if epoch % val_every == 0:
    val_metrics = evaluate(model, val_loader)
    writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
```

## Hyperparameter Conventions

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| batch_size | GPU-dependent | Use Golden Rules to compute |
| learning_rate | 1e-4 to 3e-4 (Adam) | Scale with batch_size |
| weight_decay | 0.01 (AdamW) | Exclude bias from decay |
| grad_clip | 0.5 to 1.0 | 1.0 for transformers |
| warmup_steps | 2-10% of total steps | Linear warmup |
| label_smoothing | 0.05 to 0.1 | Prevents overconfidence |
| dropout | 0.1 to 0.3 | Task-dependent |

## Error Handling

```python
# ALWAYS check for NaN in gradients
def check_nan_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            raise RuntimeError(f"NaN gradient in {name}")

# Handle GPU OOM gracefully
try:
    output = model(input)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Halve batch and retry once
        if batch_size > 1:
            raise RuntimeError(f"OOM at batch_size={batch_size}; reduce batch or enable grad accum")
    raise
```
