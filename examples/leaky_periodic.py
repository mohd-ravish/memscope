"""
Leak Type 3: Periodic accumulation — validation loop pattern.

val_outputs accumulates across every batch of every epoch but is never cleared.
Memory spikes each epoch and never fully releases.

Expected: MemScope flags PERIODIC_ACCUMULATION.
Fix: val_outputs.clear() after each epoch.
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from memscope import MemoryProfiler

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

model = nn.Linear(500, 500).to(device)
val_outputs = []

with MemoryProfiler(output_dir="./memscope-report/leaky_periodic", model=model) as prof:
    for epoch in range(20):
        # Simulated training — fine
        for _ in range(5):
            x = torch.randn(32, 500, device=device)
            out = model(x)
            loss = out.sum()
            loss.item()  # detach via .item()
            prof.step()

        # Simulated validation — leaky
        for _ in range(5):
            with torch.no_grad():
                x = torch.randn(32, 500, device=device)
                out = model(x)
                val_outputs.append(out)  # BUG: grows every epoch, never cleared

        allocated = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
        print(f"  epoch {epoch:2d} | GPU allocated: {allocated:.1f} MB | val_outputs len: {len(val_outputs)}")

prof.report()
print("\nFix: add `val_outputs.clear()` after each validation loop")
