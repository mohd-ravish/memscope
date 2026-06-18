"""
Leak Type 2: Tensor list accumulation.

Model output tensors are stored directly in a Python list.
GPU tensors remain on the device — memory grows with every step.

Expected: MemScope flags LINEAR_LEAK.
Fix: results.append(out.detach().cpu())
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
results = []

with MemoryProfiler(output_dir="./memscope-report/leaky_tensor_list", model=model) as prof:
    for step in range(200):
        x = torch.randn(64, 1000, device=device)
        out = model(x)
        results.append(out)  # BUG: GPU tensor held by Python list forever
        prof.step()
        if step % 20 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
            print(f"  step {step:3d} | GPU allocated: {allocated:.1f} MB | results list len: {len(results)}")

prof.report()
print("\nFix: change `results.append(out)` to `results.append(out.detach().cpu())`")
