---
description: ML training assistant — plan, run, test, verify, and explain PyTorch / LLM training workflows. Also supports community templates, research mode, and auto-complete workflows. Delegates to the CoLoMo subagent for full ML workflows.
---

# /ml — ML Training Command

Invoke the **CoLoMo** agent for autonomous ML training assistance.

## Usage

```
/ml <subcommand> [args...]
```

## Subcommands

| Subcommand | Description |
|-----------|-------------|
| `detect` | Detect GPU/CPU config, show Golden Rules recommendations |
| `plan <requirement>` | Detect → Calculate → Generate .colomo/plan.md + colomo_config.yaml |
| `execute` | Check env → Install deps → Generate code from plan (reads config.yaml) |
| `advise` | GPU-based hyperparameter recommendations (fills config.yaml) |
| `run` | Run training in the project's Conda environment |
| `test` | Run pytest in the project's Conda environment |
| `verify` | Generate and run verification script for trained model |
| `auto <requirement>` | Full auto-complete: advise → plan → execute → tester → runner → verify (asks at each step) |
| `explain <topic>` | Explain an ML algorithm (teacher mode, research mode adds paper citations) |
| `rollback [n]` | Undo the last N config changes |
| `add <template_repo>` | Add community template from GitHub repo |
| `setting` | Configure mode (normal/research), API keys, defaults |
| `status` | Show current mode, API key status, config.yaml summary |

## Workflows

### Standard Workflow
```
Requirement
    ↓
/ml plan       → Detect GPU → Calculate params → Generate .colomo/plan.md + .colomo/colomo_config.yaml
    ↓
/ml execute    → Check env → Install deps → Generate code from plan
    ↓
/ml run        → conda run -n <env> python train.py
    ↓
/ml test       → conda run -n <env> pytest
```

### Auto-Complete Workflow (with confirmation at each step)
```
Requirement
    ↓
/ml auto       → ADVISE → [confirm] → PLAN → [confirm] → EXECUTE → [confirm] → TESTER → [confirm] → RUNNER → [confirm] → VERIFY
```

### Research Mode (requires API key)
```
/ml setting    → Switch to "research" mode
/ml explain    → Papers from Semantic Scholar/arXiv + hallucination-prevented answers
```

## 1. Config Initialization

Every `/ml` call first checks for `.colomo/colomo_config.yaml`. If missing, creates a template and prompts:
```
[CoLoMo] First time setup. Created .colomo/colomo_config.yaml
  [1] Run /ml advise — detect GPU, auto-fill hyperparameters (recommended)
  [2] Set manually via /ml setting
  [3] Continue without advising
```

## 2. Detect

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
nproc && free -h | grep Mem
```

## 3. Plan (`/ml plan`)

1. Detect GPU/CPU config
2. Calculate batch_size, lr, optimizer via Golden Rules
3. Generate `.colomo/plan.md` with tasks marked `[ ]`
4. Generate `.colomo/colomo_config.yaml`
5. Return summary

## 4. Execute (`/ml execute`)

**Pre-execution checks (stop on first failure):**

```
1. Check .colomo/plan.md exists?   → Error if missing, prompt /ml plan first
2. Detect conda location?            → Ask user: install conda or use system python?
3. Check conda env exists?           → Ask user: create env or use existing?
4. Check Python deps installed?      → pip install -r requirements.txt
5. Execute plan tasks in order
```

**All generated code reads from `.colomo/colomo_config.yaml`:**
```python
import yaml
with open('.colomo/colomo_config.yaml') as f:
    config = yaml.safe_load(f)
# use config['training']['batch_size'], etc.
```

### Conda Not Found → Ask User

```
[ERROR] Conda not found in PATH.
  Detected Python: <path>
  Miniconda recommended for ML projects.

  Options:
    [1] Install Miniconda (recommended) — I'll run the installer
    [2] Use system Python — continue without conda
    [3] Cancel

  Your choice:
```

### Conda Env Missing → Ask User

```
[ERROR] Conda environment "colomo-ml" not found.
  config.yaml specifies: conda_env: colomo-ml

  Options:
    [1] Create environment "colomo-ml" — conda create -n colomo-ml python=3.11
    [2] Use existing environment — specify name
    [3] Cancel

  Your choice:
```

## 5. Tester (auto after execute in auto-complete)

Tests generated code. If errors:
- Writes `.colomo/examine.md` with error traceback, failed files, user requirement
- Prompts: "Restart workflow with examine report?"

## 6. Verify (`/ml verify`)

Generates `.colomo/verify_<task>.py` and runs it:
- Model loads from checkpoint
- Inference runs without error
- Metrics within expected range

## 7. Advise (Golden Rules)

- `BS = 0.90 × VRAM_Available / (param_mem + activation_mem)`
- `LR_new = LR_old × (BS_new / BS_old)`
- Optimizer: SGD (<10M), Adam (10M–100M), AdamW (>100M)
- Fills results into `.colomo/colomo_config.yaml`

## 8. Mode Settings (`/ml setting`)

```
=== CoLoMo Settings ===

Current Mode: normal
API Keys:
  - Semantic Scholar: [not set] ← research mode requires this
  - arXiv: [not set]

Options:
  [1] Switch mode: normal ↔ research
  [2] Set Semantic Scholar API key
  [3] Set arXiv API key
  [4] Set default conda environment
  [5] View current colomo_config.yaml
  [6] Reset to defaults
```

**Research mode**: requires Semantic Scholar API key. Falls back to normal if not set.

## 9. Community Templates (`/ml add`)

```
/ml add user/my-template-repo
```

Fetches template from GitHub, validates `.claude-plugin/template.json`, adds to local index.

## 10. Status (`/ml status`)

```
=== CoLoMo Status ===
Mode: normal
.colomo/colomo_config.yaml: exists
  - GPU: RTX 4090, 24GB
  - batch_size: 64
  - lr: 1e-4
  - optimizer: AdamW
API Keys:
  - Semantic Scholar: [not set] (research mode unavailable)
  - arXiv: [not set]
```

## Examples

```
/ml plan fine-tune Llama2 with QLoRA on a single GPU
/ml execute
/ml detect
/ml advise
/ml run
/ml test
/ml verify
/ml auto fine-tune BERT for sentiment classification
/ml add user/my-lora-template
/ml setting
/ml status
/ml rollback 2
```

## Mode: Research

When research mode is active:
- `/ml explain <topic>` includes citations from Semantic Scholar/arXiv
- Hallucination prevention: claims marked `[unverified]` if no supporting paper
- Algorithm logic from latest papers, not just training data

**Requires**: Semantic Scholar API key (free at api.semanticscholar.org)

## Related

- Skill: `ml-training` (Golden Rules, 17 PyTorch snippets, LoRA/QLoRA, RAG)
- Skill: `research-mode` (paper search, hallucination prevention, citation grounding)
- Agent: `colomo` (full autonomous training subagent)
- Rules: `rules/ml/` (PyTorch conventions, Golden Rules patterns)
