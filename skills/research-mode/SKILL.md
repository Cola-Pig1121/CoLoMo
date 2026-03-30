---
name: research-mode
description: This skill should be used when the user has enabled research mode (colomo_config.yaml mode=research) or asks about academic papers, algorithm research, hallucination prevention, paper citations, or Semantic Scholar/arXiv search.
version: 1.0.0
---

# Research Mode Skill

Research mode provides academic paper-grounded answers, hallucination prevention, and algorithm logic lookup via Semantic Scholar and arXiv APIs.

---

## API Setup

### Semantic Scholar API (Recommended)
- **Endpoint**: `https://api.semanticscholar.org/graph/v1`
- **Free tier**: 100 requests/day without key
- **With key**: Higher rate limits
- **Get key**: https://api.semanticscholar.org
- **Search endpoint**: `GET /paper/search?query={query}&limit=5&fields=title,abstract,authors,openAccessPdf,year`

### arXiv API (Free fallback)
- **Endpoint**: `https://export.arxiv.org/api/query`
- **Rate limit**: 1 request per 3 seconds
- **No API key required**
- **Search syntax**: `search_query=all:term+AND+ti:term`

---

## Hallucination Prevention

### When generating explanations or code:

1. **Search first** — query Semantic Scholar for relevant papers before explaining an algorithm
2. **Cite sources** — include inline citations with paper titles and URLs
3. **Mark unverified claims** — if no supporting paper found, mark as `[unverified]`
4. **Confidence scoring** — based on number and quality of supporting sources

### Citation Format
```markdown
According to [Paper Title](https://arxiv.org/abs/XXXXX) by Authors (Year):

Key claim explanation here.

[Confidence: High/Medium/Low based on citation count]
```

### Unverified Claims
```markdown
CLAIM: Transformers use self-attention mechanism
STATUS: [verified] — Vaswani et al. (2017) "Attention Is All You Need"
SOURCE: https://arxiv.org/abs/1706.03762

CLAIM: Some additional claim
STATUS: [unverified] — no supporting paper found in Semantic Scholar
```

---

## Paper Search Workflow

### For algorithm explanations:

```
1. User asks: "Explain LoRA fine-tuning"
2. Search Semantic Scholar: "LoRA low-rank adaptation"
3. Search arXiv: "LoRA fine-tuning language models"
4. Read top 3 paper abstracts
5. Synthesize answer with citations
6. Mark any claims without support as [unverified]
```

### Search Query Examples

| Topic | Semantic Scholar Query | arXiv Query |
|-------|----------------------|-------------|
| LoRA | `LoRA low-rank adaptation fine-tuning` | `all:LoRA+AND+ti:fine-tuning` |
| RAG | `retrieval augmented generation hallucination` | `all:RAG+AND+ti:retrieval` |
| Attention | `attention mechanism transformer` | `all:attention+AND+ti:transformer` |
| Gradient clipping | `gradient clipping transformer training` | `all:gradient+AND+ti:clipping` |

---

## API-Gated Features

When `colomo_config.yaml` has `mode: research` but no API key:

```
[WARNING] Research mode requires a Semantic Scholar API key.
Without it, paper search and hallucination prevention are unavailable.

Options:
  [1] Set API key via /ml setting → API Keys → Semantic Scholar Key
  [2] Continue in normal mode (no paper search)
  [3] Use arXiv only (free, no key needed)

Your choice:
```

---

## Response Structure for Research Mode

### Algorithm Explanation
```markdown
# LoRA: Low-Rank Adaptation

**What it is**: A parameter-efficient fine-tuning method that adds trainable rank-decomposition matrices to pretrained weights.

**How it works** [Paper: "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)
- Freezes pretrained weights W₀
- Adds trainable matrices B and A where W = W₀ + BA
- Rank r << min(d,k) keeps trainable params to 0.1-1%

**When to use**: Single GPU, large models (>1B params), multiple task adapters.

**Pros**: [verified] — 0.1-1% trainable params, zero inference overhead [Hu et al. (2022)]
**Cons**: [unverified] — Lower rank may limit expressiveness for some tasks

**Variants**:
- QLoRA (quantized base + LoRA) [verified] — Dettmers et al. (2023)
- DoRA (weight decomposition) [unverified]
- AdaLoRA (adaptive rank) [unverified]
```

### Code Generation with Citations
```python
# LoRA Configuration
# Based on: "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
# https://arxiv.org/abs/2305.14314
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                           # rank: 4-16 (higher = more expressive)
    lora_alpha=16,                 # scaling = lora_alpha / r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
# Trainable params: ~0.1-1% of total [verified: Dettmers et al.]
```

---

## Mode Detection

Research mode is active when:
1. `.colomo/colomo_config.yaml` exists
2. `mode.default` is `"research"`
3. At least one API key is configured (Semantic Scholar or arXiv fallback)

Normal mode: No paper search, no citations, no hallucination prevention markers.

---

## Graceful Degradation

| API Key Status | Research Mode Available |
|----------------|------------------------|
| Semantic Scholar key set | Full research mode |
| arXiv key set (future) | arXiv-only search |
| No key | Falls back to normal mode with warning |

---

## Confidence Scoring

| Citation Count | Confidence |
|----------------|------------|
| 3+ papers support | High |
| 1-2 papers support | Medium |
| No papers found | Low (mark as [unverified]) |

Also consider:
- Publication venue (arXiv vs peer-reviewed)
- Citation count in Semantic Scholar
- Recency (prefer last 2-3 years for rapidly evolving topics)
