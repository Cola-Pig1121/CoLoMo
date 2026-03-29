# 模型训练实践

## Transformers Trainer

Transformers 的 Trainer 类封装了分布式训练的核心逻辑：

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # 等效 batch_size = 8*4=32
    learning_rate=3e-4,
    num_train_epochs=3,
    warmup_ratio=0.03,       # 预热步数比例
    logging_steps=10,
    save_steps=500,
    bf16=True,               # 使用 BF16 混合精度
    deepspeed="ds_config.json",  # DeepSpeed 配置
)
```

**Trainer 封装能力：**
- 分布式训练：DDP、DeepSpeed、Megatron-LM
- 混合精度：FP16/BF16
- 日志集成：wandb、SwanLab
- Checkpoint 保存与恢复

---

## Batch Size 与学习率的关系

**线性缩放规则（Linear Scaling）：**
当 batch size 增大 k 倍时，学习率也应线性增大 k 倍。

```
lr_new = lr_old × (batch_size_new / batch_size_old)
```

**Grad Accumulation 等效 batch：**
```
effective_batch_size = batch_size × gradient_accumulation_steps × num_gpus
```

**Q: gradient_accumulation_steps 的作用？**
A: 当单卡显存不足以容纳大 batch 时，通过累计多个小 batch 的梯度来模拟大 batch 更新，在不增加显存占用的前提下达到等效大 batch 的训练效果。

---

## 分布式训练策略

| 策略 | 适用场景 | 通信开销 |
|------|---------|---------|
| Data Parallel（DDP） | 数据并行，多卡同步梯度 | 高（每步同步） |
| ZeRO Stage 1 | 分片优化器状态 | 中等 |
| ZeRO Stage 2 | 分片优化器状态 + 梯度 | 中等 |
| ZeRO Stage 3 | 分片全部状态 | 较低 |
| Pipeline Parallel | 层间并行，大模型跨卡 | 较低（流水线气泡） |
| Tensor Parallel | 张量切分，单层跨多卡 | 高（每层通信） |

**Q: DeepSpeed ZeRO-2 和 ZeRO-3 如何选择？**
A: ZeRO-2 适合显存不足但计算资源充足；ZeRO-3 适合超大规模模型，需要极致显存优化。ZeRO-3 通信开销最大。

---

## 学习率与 Warmup

**常见学习率策略：**
- **固定学习率**：简单但需要手动调节
- **余弦退火（Cosine Annealing）**：学习率按余弦曲线衰减，收敛更平滑
- **阶梯衰减（Step Decay）**：每隔固定步数减半学习率
- **指数衰减（Exponential Decay）**：学习率指数下降

**Warmup 作用：**
训练初期权重更新幅度大、梯度分布不稳定。warmup 让学习率从 0 逐渐增加到目标值，避免早期训练不稳定。

```python
# 余弦退火 + Warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=500, T_mult=2, eta_min=1e-6
)
```

---

## 混合精度训练

| 精度 | 显存占用 | 计算速度 | 精度损失 |
|------|---------|---------|---------|
| FP32 | 1x | 1x | 无 |
| FP16 | 1/2 | ~2x | 可能较大 |
| BF16 | 1/2 | ~2x | 几乎无损失 |

**Q: 为什么 BF16 比 FP16 更适合 LLM 训练？**
A: BF16 有更大的指数范围（和 FP32 相同），动态范围更大，避免了 FP16 在训练中期可能出现的梯度爆炸或 NaN 问题，同时显存节省与 FP16 相当。

---

## Flash Attention

Flash Attention 将注意力计算分块加载到 SRAM、避免生成完整的注意力矩阵（N×N），将显存复杂度从 O(N²) 降至 O(N)。

**优势：**
- 长序列训练显存显著降低
- 计算速度更快（减少 HBM 访问）
- 训练稳定性和精度不受影响

```python
# Transformers 中启用
model = AutoModel.from_pretrained(
    "model_name",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
```
