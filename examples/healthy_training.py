"""
Baseline — healthy training loop.

All tensors are properly released each step.
Expected: MemScope reports HEALTHY — flat memory.
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from memscope import MemoryProfiler

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

model = nn.Linear(1000, 1000).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []

with MemoryProfiler(output_dir="./memscope-report/healthy", model=model) as prof:
    for step in range(200):
        optimizer.zero_grad()
        x = torch.randn(64, 1000, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # .item() extracts scalar — computation graph freed
        prof.step()
        if step % 20 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
            print(f"  step {step:3d} | GPU allocated: {allocated:.1f} MB | loss: {losses[-1]:.4f}")

prof.report()
print("\nMemory should be flat — this is a correctly written training loop.")
