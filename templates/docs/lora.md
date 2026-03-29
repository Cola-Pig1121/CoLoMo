# LoRA 高效微调

## LoRA 原理

LoRA（Low-Rank Adaptation）通过低秩矩阵分解来高效微调大模型。

**核心思想：**
- 冻结预训练权重 W₀
- 添加可训练的低秩分解 W = W₀ + BA
- 其中 B ∈ R^(d×r), A ∈ R^(r×k)，r << min(d, k)
- 仅训练 A、B 两个低秩矩阵

**优势：**
- 参数量大幅减少（通常 r 在 4~16 之间）
- 可合并为单一大矩阵，推理零额外开销
- 可为不同任务训练不同 LoRA 权重，随时切换

**Q: LoRA 的秩 r 如何选择？**
A: r 的选择取决于任务复杂度。简单任务 r=4~8 足够；复杂任务 r=8~16。更大的 r 收益递减，计算成本线性增长。

**Q: LoRA 训练时底层模型需要梯度吗？**
A: 不需要。LoRA 冻结预训练权重，只训练 LoRA 的低秩矩阵 A 和 B，这大幅降低了显存占用。

---

## QLoRA 原理

QLoRA = Quantization + LoRA，在 LoRA 的基础上引入了量化：

1. **NF4 量化**：4-bit NormalFloat 量化，更适合正态分布的权重
2. **双重量化**：对量化常数也做量化
3. **分页注意力**：处理梯度累积时的内存峰值

**显存节省：**
- 65B 模型在单卡 48GB 上可微调（原本需要 > 500GB）

**Q: 为什么 QLoRA 能大幅降低显存？**
A: 将预训练权重从 FP16/BF16 量化为 NF4（4-bit），在保持性能的同时将权重显存降低 4 倍；配合 LoRA 的低秩更新，合计可实现单卡微调百亿参数模型。

---

## LoRA 变体

| 变体 | 核心改进 | 适用场景 |
|------|---------|---------|
| LoRA | 基础低秩适应 | 通用微调 |
| QLoRA | NF4 量化 + LoRA | 极致显存优化 |
| DoRA | 权重分解方向微调 | 更高精度需求 |
| AdaLoRA | 自适应秩调整 | 动态资源分配 |
| LoRA+ | 单独学习率 | 加速收敛 |

---

## LoRA 实战配置

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                        # 秩
    lora_alpha=16,              # 缩放因子
    target_modules=[            # 目标模块
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# 可训练参数: 0.1%（以 7B 模型为例）
```
