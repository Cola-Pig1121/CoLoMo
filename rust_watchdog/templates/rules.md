# CoLoMo Golden Rules

## Rule: GPU Memory Overflow
### Condition
- Error contains "CUDA out of memory" or "CUDA OOM"
- nvidia-smi shows memory_used / memory_total > 0.9
### Hard Fields
- error_code: regex "CUDA out of memory|CUDA OOM|cuDNN"
- gpu_memory_mb: from EnvInfo
### Formula
RecommendedBatchSize = α * (GPU_Memory_MB / (ParamMem_MB + ActivationPerSample_MB * FeatureSize))
α = 0.8
### Recommendation
- Reduce batch_size to RecommendedBatchSize
- Optionally set accum_steps = ceil(TargetBatchSize / ActualBatchSize)
### Example
BatchSize=128, GPU=8192MB, ParamMem=2000MB, ActivationPerSample=20MB -> RecommendedBatchSize ≈ 80

---

## Rule: Optimizer Selection
### Condition
- ParamCount thresholds
### Formula
Optimizer =
- SGD+Momentum if ParamCount < 1e7
- Adam if 1e7 ≤ ParamCount < 1e8
- AdamW if ParamCount ≥ 1e8
### Recommendation
Switch optimizer accordingly

---

## Rule: Dependency Conflict
### Condition
- pip check reports conflicts
### Recommendation
- Suggest pinning versions in requirements.txt
- Suggest recreating conda env from requirements.txt snapshot

---

## CoLoMo 推荐规则速查

| 场景 | 推荐动作 | 参考公式/规则 |
|------|---------|--------------|
| OOM 错误 | batch_size 减半 | batch_new = batch_old / 2 |
| GPU 利用率 < 50% | batch_size 增加 25% | batch_new = batch_old × 1.25 |
| batch size 变化 | 线性缩放学习率 | lr_new = lr_old × (batch_new / batch_old) |
| 参数 > 1亿 | 建议使用 AdamW | - |
| 参数 1000万~1亿 | 建议使用 Adam | - |
| 参数 < 1000万 | 建议使用 SGD | - |
| batch 减少 | 设置 grad_accum_steps | accum = ceil(old_batch / new_batch) |
| GPU 显存安全系数 | safety_alpha = 0.85 | 预留 15% headroom |

---

## 参考知识库

详见 `docs/` 目录下的分片知识文件：

| 文件 | 内容 |
|------|------|
| `docs/transformer.md` | Transformer 架构、注意力机制、RoPE、RMSNorm、GQA |
| `docs/pretrain.md` | 预训练语言模型、三种架构、BERT |
| `docs/llm.md` | LLM 涌现能力、RLHF、训练三阶段 |
| `docs/lora.md` | LoRA、QLoRA 原理与实战配置 |
| `docs/training.md` | 分布式训练、batch-LR 关系、Flash Attention |
| `docs/rag.md` | RAG 核心流程、优化技巧、向量数据库 |
| `docs/qa.md` | 常见问题与解答（训练、显存、评测等） |

*知识库基于 Happy-LLM (Datawhale) 教程构建，版本 2025-10。*
