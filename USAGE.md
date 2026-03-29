# CoLoMo 使用指南

> ⚠️ **Legacy document** — CoLoMo has been converted to a Claude Code plugin. The TUI application (`rust_watchdog/`) has been removed. See README.md for the new plugin-based usage.

本文档为历史文档，CoLoMo 已转型为 Claude Code 插件，原 TUI 应用已移除。

---

## 前置条件

- Rust 1.75+ (`rustup update`)
- Miniconda 或 Anaconda
- NVIDIA GPU + nvidia-smi（watchdog 支持无 GPU 降级模拟）
- Claude Code CLI（用于 AI Agent 功能，`claude --print`）
- （可选）iflow API key（用于 LLM 摘要功能）

---

## 编译与首次启动

```bash
# 编译 release 版本
cd rust_watchdog
cargo build --release

# 复制环境变量模板
cp .env.example .env
# 编辑 .env，填入 IFLOW_API_BASE 和 IFLOW_API_KEY（可选）
```

---

## TUI 操作手册

运行 `cargo run` 后进入 TUI：

```
┌─────────────────────────┬─────────────────────────────────────┐
│  引导配置（Guided）     │  Plan/Todo │ 最近日志 │ 摘要+推荐   │
│  可编辑配置表           │  [>>]进行中 │            │ A 键应用    │
│                         │  [DONE]完成 │            │            │
│                         │  [ ]待处理  │            │            │
├─────────────────────────┴─────────────────────────────────────┤
│  Command Input + Status (Idle | Agent状态 | 进度 3/5)       │
└───────────────────────────────────────────────────────────────┘
```

### 界面布局

| 区域 | 说明 |
|------|------|
| 左上 | Guided Mode 配置表 / Expert Mode 文件列表 |
| 右上-Plan/Todo | 当前计划任务，带颜色状态标记 |
| 右上-最近日志 | 训练日志输出 |
| 右下-摘要 | Advisor 推荐内容 |
| 底部 | 命令输入 + 状态栏（合并显示） |

### 状态标记说明

| 标记 | 颜色 | 含义 |
|------|------|------|
| `[>>]` | 青色 | 正在执行 |
| `[DONE]` | 绿色 | 已完成 |
| `[ ]` | 黄色 | 待处理 |
| `[!]` | 红色 | 错误 |

### 快捷键（常规模式）

| 快捷键 | 功能 |
|--------|------|
| `/` | 进入**命令模式**（底部弹出输入框） |
| `A` | 应用 Advisor 推荐（更新 batch_size / learning_rate / optimizer） |
| `鼠标左键` | 在输入框中点击定位光标 |
| `鼠标右键` | 粘贴剪贴板内容 |
| `Ctrl+C` | 退出（常规模式下直接退出） |

### 命令模式快捷键

| 快捷键 | 功能 |
|--------|------|
| `Esc` | 取消命令模式 |
| `↑` / `↓` | 在命令列表中上下导航 |
| `Tab` | 补全当前选中的命令并关闭提示 |
| `Enter` | 确认执行命令 |
| `Ctrl+V` | 粘贴剪贴板内容 |
| `Ctrl+C` | **按两次**退出（命令输入模式下） |
| `←` / `→` | 光标左右移动（按字符） |
| `Backspace` | 删除光标前字符 |
| `Delete` | 删除光标后字符 |

### 命令模式指令

按 `/` 进入命令模式后，可输入以下指令：

#### /plan

基于需求生成实现计划（调用 AI Agent）：

```bash
# 用法
/plan implement batch size auto-tuning

# 输出示例
Planning: implement batch size auto-tuning
Starting planner agent...
  [1/5] Understanding requirement
  [2/5] Analyzing project context
  [3/5] Generating implementation plan
  [4/5] Validating plan
  [5/5] Plan ready
✓ Planner: Plan generated at .claude/plan/plan.md
```

计划文件保存在项目目录的 `.claude/plan/plan.md`。

#### /new

交互式创建新项目：

```bash
# 在 TUI 中按 /new，然后按提示输入：
# 1. 项目名称（如 my_model）
# 2. 项目路径（如 projects/）
```

创建的项目包含：
- `config.yaml` - 训练配置文件
- `train.py` - 训练脚本模板
- `requirements.txt` - Python 依赖
- `logs/` - 日志目录
- `saved/` - 检查点目录
- `plan.md` - 项目计划文档

#### /setting

调整运行参数，无需重启 TUI：

```bash
# 调整安全系数（显存预留比例）
/setting safety_alpha=0.9

# 调整评分权重（用于多候选评估）
/setting acc=0.5 lat=0.2 mem=0.2 thr=0.1 energy=0.0

# 开启学习模式
/setting learning_mode=true

# 批量设置
/setting safety_alpha=0.85 acc=0.6 lat=0.2 mem=0.2
```

#### /rollback

回滚最近 N 次配置变更：

```bash
# 回滚最近 1 次配置变更
/rollback 1
```

> 回滚会从 `journal.jsonl` 读取快照，将 `config.yaml` 恢复至历史版本。

---

## 配置文件说明

### 项目配置（config.yaml）

每个项目目录下应有独立的 `config.yaml`：

```yaml
conda_env: colomo          # Conda 环境名称
backend: cuda             # 运行后端：cuda | tilelang | jupyter
train_script: projects/demo/train.py   # 训练脚本（.py 或 .ipynb）
tile_script: null          # TileLang 脚本（仅 backend=tilelang 时使用）
batch_size: 32
learning_rate: 0.0003
optimizer: AdamW
dataset_path: ./data
param_count: 110000000     # 模型参数量（Advisor 用来选择优化器）
```

### 全局设置（settings.yaml）

位于 `rust_watchdog/settings.yaml`，影响所有项目：

```yaml
safety_alpha: 0.85   # GPU 显存安全系数（0.5-0.95）
                      # 较低值 = 保守，留更多显存余量
                      # 较高值 = 激进，可设更大 batch_size
learning_mode: false # true 时每次应用推荐会触发 teacher agent 解释算法原理
weights:             # 多候选评估时的加权评分
  acc: 0.6           # 准确率权重
  lat: 0.2           # 延迟权重
  mem: 0.2           # 显存权重
  thr: 0.0           # 吞吐量权重
  energy: 0.0        # 能耗权重
```

---

## Advisor 推荐逻辑

每次按 `A` 时，Advisor 会根据以下信号给出推荐：

| 信号 | 条件 | 推荐动作 |
|------|------|---------|
| OOM 日志 | 训练日志中检测到 "out of memory" | batch_size 减半 |
| GPU 空闲 | 已用显存 < 总显存的 50% | batch_size 增加 25% |
| batch 变化 | 旧 batch != 新 batch | 按比例缩放 learning_rate | 
| 参数量 | >1 亿参数 | 建议切换到 AdamW |
| 参数量 | 1000 万 - 1 亿参数 | 建议切换到 Adam |
| 参数量 | < 1000 万参数 | 建议切换到 SGD |
| batch 减少 | 新 batch < 旧 batch | 自动设置 grad_accum_steps 以保持有效 batch |

推荐会记录在 `journal.jsonl` 中，包含完整的推理过程（rationale）。

---

## 项目模板

### 查看可用模板

```bash
python rust_watchdog/templates/model_templates/tools.py list
```

示例输出：

```
pytorch_template  full      PyTorch Trainer | 完整训练脚手架（trainer、data_loader、utils）
mixup             snippet    Augmentation     | Mixup 数据增强实现
label_smoothing   snippet    Loss            | 标签平滑交叉熵损失
grad_clip         snippet    Training        | 梯度裁剪
lr_decay          snippet    Scheduler       | 余弦退火学习率调度
finetune_fc       snippet    Finetune        | 仅微调最后分类层
```

### 使用模板

```bash
# 获取完整 PyTorch 模板项目
python rust_watchdog/templates/model_templates/tools.py \
  fetch pytorch_template projects/my_model/

# 获取单个代码片段
python rust_watchdog/templates/model_templates/tools.py \
  fetch mixup projects/my_model/mixup.py

# 或用 Bash 快捷方式
bash scripts/use_template.sh pytorch_template projects/my_model/
```

---

## 日志与回滚

### journal.jsonl 格式

每次配置变更都会追加一条 JSONL 记录：

```json
{"timestamp":"2026-03-28T10:00:00Z","action":"apply_recommendation","file":"projects/demo/config.yaml","applied":true,"grad_accum_steps":2,"rationale":"Detected OOM -> halve batch 64 -> 32\nScale LR: 0.001 -> 0.0005\nReduce batch 64->32 -> set grad_accum_steps=2 to preserve effective batch","old_config":{"batch_size":64,"learning_rate":0.001,"optimizer":"AdamW"}}
```

### 查看历史

```bash
cat journal.jsonl | jq '.action, .timestamp'
```

---

## 常见问题

### Q: 训练启动失败

检查项：
1. `conda_env` 是否与已创建的 Conda 环境名一致
2. `train_script` 路径是否正确（相对于项目根目录）
3. Conda 是否在 PATH 中（`conda --version`）

### Q: Advisor 没有推荐

- 确保 `config.yaml` 中设置了 `batch_size`（Advisor 需要基准值）
- `param_count` 有值时才会触发优化器推荐
- GPU 使用率正常（50-90%）且无 OOM 日志时，Advisor 保持静默（"No strong signals"）

### Q: nvidia-smi 不可用

watchdog 内置降级模拟：GPU 状态返回 `simulated: true`，显存显示为 8192 MB / 2048 MB 已用。仅影响 Advisor 的 batch 调整，不会阻止训练运行。

---

## 环境变量

| 变量 | 必填 | 说明 | 示例 |
|------|------|------|------|
| `IFLOW_API_BASE` | 否 | LLM API 地址 | `https://api.iflow.cn/v1` |
| `IFLOW_API_KEY` | 否 | API 密钥 | `sk-xxx` |
| `IFLOW_MODEL` | 否 | 模型名称 | `qwen-max-preview` |
| `IFLOW_TIMEOUT_MS` | 否 | 请求超时（毫秒） | `15000` |

---

## 扩展指南

### 新增后端

在 `rust_watchdog/src/watchdog.rs` 的 `run` 函数中添加 match 分支：

```rust
"my_backend" => {
    Command::new("my_backend")
        .arg("run")
        .arg(&self.train_script)
        ...
}
```

### 接入新 LLM

修改 `rust_watchdog/src/llm.rs`：将 `reqwest` 请求目标改为 OpenAI / Anthropic / Groq 等 API，保持 `summarizer.rs` 的接口不变即可。
