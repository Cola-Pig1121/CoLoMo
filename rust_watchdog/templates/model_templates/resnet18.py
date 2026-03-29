#!/usr/bin/env python3
"""
Minimal ResNet18 demo (requires torch + torchvision + pyyaml).
This is optional; use only if PyTorch is installed.
"""
import time, yaml, os
try:
    import torch
    import torchvision
except Exception as e:
    print("PyTorch not available; exiting:", e)
    raise SystemExit(1)

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

bs = int(cfg.get("batch_size", 32))
device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("backend","cuda")=="cuda" else "cpu")
model = torchvision.models.resnet18(weights=None).to(device)

for step in range(1, 201):
    # fake workload
    x = torch.randn(bs, 3, 224, 224, device=device)
    y = model(x)
    print(f"step={step} out_mean={y.mean().item():.6f}")
    time.sleep(0.01)
    if device.type == "cuda" and bs >= 128 and step == 120:
        raise RuntimeError("CUDA out of memory (demo)")

print("resnet18 demo complete")
