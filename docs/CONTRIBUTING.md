# CoLoMo Contributing Guide

CoLoMo is a Claude Code plugin. Contributions are primarily knowledge content (skills, rules, agents) rather than application code.

## Plugin Structure

| Directory | Purpose |
|-----------|---------|
| `.claude-plugin/plugin.json` | Plugin manifest |
| `.claude-plugin/marketplace.json` | Marketplace discovery manifest |
| `skills/ml-training/SKILL.md` | ML training knowledge base (17 snippets + Golden Rules) |
| `agents/colomo.md` | CoLoMo subagent definition |
| `rules/ml/coding-style.md` | PyTorch code conventions |
| `rules/ml/patterns.md` | Golden Rules + ML design patterns |
| `templates/pytorch-snippets/` | 17 standalone PyTorch snippet files |
| `templates/model_templates/` | Full project templates (e.g. pytorch-template) |
| `templates/docs/` | Algorithm references (LoRA, RAG, Transformer, etc.) |

## Adding ML Knowledge

### Extend the Skill

Edit `skills/ml-training/SKILL.md` to add:
- New algorithm explanations (follow the `### Example: <topic>` format)
- Additional PyTorch patterns with code snippets
- Distributed training strategies
- LLM training techniques

### Extend the Rules

- **`rules/ml/coding-style.md`**: PyTorch conventions — follow the existing section format
- **`rules/ml/patterns.md`**: Design patterns — follow the `### Pattern Name` section format

### Extend the Agent

Edit `agents/colomo.md`:
- New agent commands (add to the Available Commands table)
- New tools (add to Tool Reference)
- New pipeline stages

## Content Guidelines

### Golden Rules

Formulas must include:
1. **Condition** — when this rule applies
2. **Formula** — in plain text math notation
3. **Recommendation** — what to do
4. **Example** — concrete numbers

### Algorithm Explanations (Teacher Mode)

Follow this structure:
1. **What it is** — one-sentence definition
2. **How it works** — key mechanism with formulas
3. **When to use** — appropriate scenarios
4. **Pros** — advantages
5. **Cons** — limitations and tradeoffs
6. **Variants** — related approaches

### PyTorch Code Snippets

- Use standard PyTorch idioms
- Include type hints on function signatures
- Comment non-obvious lines
- Follow `rules/ml/coding-style.md`

## Claude Code Plugin Conventions

### Skill Frontmatter

```yaml
---
name: skill-name
description: This skill should be used when the user asks about <topics>.
version: 1.0.0
---
```

### Agent Frontmatter

```yaml
---
name: agent-name
description: When to use this agent
tools: ["Read", "Glob", "Grep", "Bash", "Write", "Edit"]
model: sonnet
---
```

### Marketplace Convention

```json
{
  "$schema": "https://anthropic.com/claude-code/marketplace.schema.json",
  "name": "<owner>-<plugin>",
  "owner": { "name": "...", "email": "..." },
  "plugins": [{
    "name": "<plugin>",
    "source": "./",
    "version": "1.0.0",
    "keywords": [...],
    "category": "workflow",
    "tags": [...]
  }]
}
```

### Rule Files

Each rule file extends a common rule:
```markdown
> This file extends [common/xxx.md](../common/xxx.md) with <domain>-specific content.
```

## Pull Request Checklist

- [ ] Frontmatter fields are complete and accurate
- [ ] Golden Rule formulas are mathematically correct
- [ ] `safety_alpha` = **0.90** (10% VRAM reserved) used in all formulas and references
- [ ] `marketplace.json` `$schema`, `name`, `owner`, `plugins` fields present
- [ ] `plugin.json` has `agents` and `skills` arrays
- [ ] Code snippets are idiomatic PyTorch
- [ ] Algorithm explanations follow the prescribed structure
- [ ] No hardcoded values — use named constants
- [ ] No placeholder content (TODO, TBD, etc.)
