# 预训练语言模型

## 三种架构对比

| 架构 | 代表模型 | 预训练任务 | 适用场景 |
|------|---------|-----------|---------|
| Encoder-Only | BERT | MLM（掩码语言模型） | NLU（分类、序列标注、问答） |
| Encoder-Decoder | T5、BART | Seq2Seq（条件生成） | 文本生成、翻译、摘要 |
| Decoder-Only | GPT 系列、LLaMA | CLM（因果语言模型） | 开放式文本生成 |

**Q: 为什么 GPT 选择 Decoder-Only 而非 Encoder-Decoder？**
A: Decoder-Only 架构更简洁，Scaling 性好，且通过下一 token 预测的预训练任务可直接用于对话、代码等生成任务。ChatGPT 的成功证明了这条路线的有效性。

---

## BERT 关键设计

- **MLM 任务**：随机掩码 15% 的 token，预测被掩码的词
- **NSP 任务**（Next Sentence Prediction）：判断句子对是否连续（已被后续工作证明价值有限）
- **WordPiece 分词**：将单词拆解为子词（"playing" -> ["play", "##ing"]）
- **激活函数**：GELU（非线性激活，比 ReLU 更平滑）

**Q: BERT 的 MLM 和 GPT 的 CLM 区别？**
A: MLM 允许看到上下文（双向），适用于理解任务；CLM 只能看到前文（单向），适用于生成任务。

---

## BERT 的 Encoder 结构

- 输入：文本序列通过 tokenizer 转化为 input_ids
- Embedding 层：转化为特定维度的 hidden_states
- Encoder Layer：堆叠 N 层（base: 12层/768维/110M参数；large: 24层/1024维/340M参数）
- prediction_heads：线性层 + Softmax 输出分类概率

**Q: BERT 中 GELU 激活函数的特点？**
A: GELU (x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))，将随机正则思想引入激活函数，通过输入自身的概率分布决定神经元抛弃或保留。
