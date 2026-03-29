# Transformer 架构

## 注意力机制三要素

Query（查询值）、Key（键值）、Value（真值）是注意力机制的三个核心变量。

**计算流程：**
1. Query 与 Key 通过点积计算注意力分数
2. Softmax 归一化得到注意力权重
3. 注意力权重与 Value 加权求和得到输出

**公式：**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**为什么需要缩放因子 √d_k？**
当 d_k 较大时，点积的方差会增大，导致 Softmax 输出趋于极端（梯度变小）。缩放因子可防止这一问题。

**Q: 为什么 RNN 难以处理长序列？**
A: RNN 依序计算限制并行能力；距离越远的信息越难捕捉；需要将整个序列读入内存，限制了序列长度。LSTM 通过门机制有所改善，但对远距离依赖仍不理想。

**Q: Transformer 为什么能解决 RNN 的缺陷？**
A: 完全使用注意力机制，允许任意位置的 token 直接相互作用，消除序列依序计算的限制，可充分并行化。

---

## 多头注意力机制

将注意力分成多个头（head），分别学习不同的注意力模式，然后拼接输出。

**参数共享变体（Grouped-Query Attention, GQA）：**
- 标准 MHA：每个头有独立的 Q、K、V
- GQA：K、V 在多个 Q 头之间共享，减少显存占用
- MQA：所有 Q 头共享同一组 K、V

**Q: LLaMA2 使用了什么注意力变体？**
A: LLaMA2 使用 GQA（Grouped-Query Attention），n_kv_heads < n_heads，在保持性能的同时减少 KV 缓存开销。

---

## 位置编码

Transformer 本身不感知序列顺序，需要额外注入位置信息。

**主流方案：**
- 绝对位置编码：Sinusoidal（GPT-2 使用）、可学习的位置编码（BERT 使用）
- 相对位置编码：RoPE（Rotary Position Embedding，LLaMA、Mistral 使用）
- ALiBi：线性偏置注意力，不添加位置嵌入

**Q: RoPE 的核心思想是什么？**
A: RoPE 通过旋转矩阵将位置信息融入 Q、K，使 attention score 自然包含相对位置关系。数学上等价于在复数空间旋转向量。

---

## 残差连接与层归一化

每个 Transformer 层都有残差连接（Add）和层归一化（Norm）：

```
output = LayerNorm(x + Sublayer(x))
```

**常见归一化方案：**
- Post-LN：归一化在残差分支之后（LLaMA 使用）
- Pre-LN：在残差分支之前进行归一化，收敛更稳定
- RMSNorm：仅计算 RMS（Root Mean Square），去掉均值 centering，计算效率更高

**Q: RMSNorm 的计算公式？**
A: RMSNorm(x) = (x / √(mean(x²) + ε)) × γ，其中 γ 是可学习的缩放参数。
