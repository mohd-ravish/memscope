"""
Leak Type 1: Computational graph accumulation.

The loss tensor is appended to a list WITHOUT calling .item().
This keeps the entire computation graph alive in memory for every step.
Memory grows linearly — never freed until the list is garbage collected.

Expected: MemScope flags LINEAR_LEAK.
Fix: losses.append(loss.item())
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
losses = []

with MemoryProfiler(output_dir="./memscope-report/leaky_graph", model=model) as prof:
    for step in range(200):
        x = torch.randn(64, 1000, device=device)
        out = model(x)
        loss = out.sum()
        losses.append(loss)  # BUG: holds entire computation graph — never freed
        prof.step()
        if step % 20 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
            print(f"  step {step:3d} | GPU allocated: {allocated:.1f} MB | losses list len: {len(losses)}")

prof.report()
print("\nFix: change `losses.append(loss)` to `losses.append(loss.item())`")
