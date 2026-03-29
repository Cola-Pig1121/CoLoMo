#!/usr/bin/env python3
"""
Lightweight demo training script that avoids heavy deps.
- Reads config.yaml if present
- Prints steps to stdout
- Simulates OOM when batch_size is large and backend=="cuda"
Requires: pyyaml
"""
import sys, time, yaml, os

CFG_PATHS = [
    os.environ.get("COLOMO_CONFIG"),
    "config.yaml",
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "demo", "config.yaml"),
]
CFG_PATHS = [p for p in CFG_PATHS if p and os.path.exists(p)] or [p for p in ["config.yaml"] if os.path.exists(p)]

cfg = {"batch_size": 64, "learning_rate": 1e-3, "backend": "cuda"}
if CFG_PATHS:
    try:
        with open(CFG_PATHS[0], "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    except Exception as e:
        print(f"WARN: failed to read config: {e}")

bs = int(cfg.get("batch_size", 64))
lr = float(cfg.get("learning_rate", 1e-3))
backend = str(cfg.get("backend", "cuda"))

print(f"CONFIG batch_size={bs} learning_rate={lr} backend={backend}")
for step in range(1, 301):
    print(f"step={step} loss={(1000/(step+10)):.4f}")
    time.sleep(0.01)
    if backend == "cuda" and bs >= 120 and step == 120:
        raise RuntimeError("CUDA out of memory: simulated for demo")

print("training complete")
