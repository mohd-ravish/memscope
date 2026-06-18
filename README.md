# MemScope

A pip-installable memory leak profiler for PyTorch training. Wrap your training loop in one `with` block and get a self-contained HTML report showing exactly which line of code is leaking memory, how fast it's growing, and when your job will OOM.

---

## The problem

When a training job crashes with `CUDA out of memory`, engineers typically:

1. Add `print(torch.cuda.memory_allocated())` manually everywhere
2. Comment out layers randomly to isolate the issue
3. Restart the job 5–10 times, burning GPU hours each time

There is no open-source tool that gives you **stack traces per leaked tensor + pattern detection + OOM prediction** in one shot. PyTorch Profiler tracks speed — not memory leak root causes. Weights & Biases tracks metrics — not allocations. MemScope fills that gap.

---

## What you get

- **Memory timeline** — allocated vs reserved GPU memory and CPU RAM across every training step
- **Leak pattern detection** — automatically classifies as `linear_leak`, `periodic_accumulation`, `over_allocation`, or `healthy`
- **Exact allocation sites** — the Python file and line number that allocated each suspicious tensor (from PyTorch's CUDA snapshot API)
- **OOM prediction** — "at current growth rate, you will OOM at step 312"
- **Per-layer breakdown** — average output tensor size for every named layer in your model
- **Self-contained HTML report** — one file you can share with your team, no server needed

---

## Installation

```bash
pip install memscope
```

Or from source:

```bash
git clone https://github.com/mohd-ravish/memscope
cd memscope
pip install -e .
```

**Requirements:** Python 3.10+, PyTorch 2.0+, NumPy, SciPy, Jinja2

---

## Quick start

```python
# Before — your existing training code, unchanged
for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# After — add 3 lines
from memscope import MemoryProfiler

with MemoryProfiler(output_dir="./memscope-report", model=model) as prof:
    for epoch in range(100):
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            prof.step()

prof.report()  # generates report, opens browser
```

That is the entire user-facing API.

---

## API reference

### `MemoryProfiler(output_dir, model, sample_interval_ms, track_layers)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `str` | `"./memscope-report"` | Directory to write the HTML report |
| `model` | `nn.Module` | `None` | Model to register per-layer forward hooks on |
| `sample_interval_ms` | `int` | `500` | How often the background thread polls memory |
| `track_layers` | `bool` | `True` | Whether to enable layer-level tracking |

### Methods

```python
prof.step()              # call once per training step — advances the step counter
prof.track_model(model)  # register layer hooks after construction
prof.report()            # generate and open the HTML report, returns the file path
```

---

## Detected patterns

| Pattern | Severity | Cause | Fix |
|---|---|---|---|
| `linear_leak` | HIGH | Tensors accumulating in a list without `.item()` or `.detach()` | `losses.append(loss.item())` instead of `loss` |
| `periodic_accumulation` | MEDIUM | Tensors collected across batches, never cleared | Add `.detach().cpu()` before storing; clear lists between epochs |
| `over_allocation` | LOW | Memory is high but stable — model too large for GPU | Reduce batch size or use gradient checkpointing |
| `healthy` | NONE | Memory is flat and stable | No action needed |

---

## Examples

Three runnable leak scripts are in `examples/`. Each one intentionally leaks in a different way so you can see exactly what MemScope catches.

```bash
# Leak type 1: computation graph held alive by a Python list
python examples/leaky_graph.py

# Leak type 2: GPU tensors stored directly in a list
python examples/leaky_tensor_list.py

# Leak type 3: validation outputs never cleared between epochs
python examples/leaky_periodic.py

# Baseline: correctly written training loop — should show HEALTHY
python examples/healthy_training.py
```

Each script generates a report in `./memscope-report/` and opens it in your browser.

---

## Report sections

The HTML report contains six sections:

1. **Summary banner** — pattern, growth rate, OOM prediction, severity, peak memory
2. **Memory timeline chart** — allocated (GPU), reserved (GPU), and CPU RAM over training steps
3. **Leak diagnosis card** — pattern classification, R² fit, plain-English explanation, fix recommendation
4. **Top tensor allocation sites** — table of the 20 largest live tensors with file, line, and function name (requires GPU with CUDA snapshot)
5. **Per-layer memory breakdown** — horizontal bar chart of average output tensor size per named layer
6. **Raw samples table** — every polled sample with CSV download

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

25 tests covering the sampler, analyzer, hook manager, and full end-to-end report generation. All tests run on CPU — no GPU required.

---

## Project structure

```
memscope/
├── pyproject.toml
├── README.md
├── examples/
│   ├── leaky_graph.py          # leak: computation graph accumulation
│   ├── leaky_tensor_list.py    # leak: GPU tensors in a Python list
│   ├── leaky_periodic.py       # leak: validation outputs never cleared
│   └── healthy_training.py     # baseline — flat memory
├── memscope/
│   ├── context.py              # MemoryProfiler context manager
│   ├── sampler.py              # background thread — polls memory every 500ms
│   ├── analyzer.py             # pattern detection via linear regression
│   ├── snapshot.py             # parses torch.cuda.memory._snapshot()
│   ├── hooks.py                # per-layer forward hooks
│   ├── reporter.py             # renders HTML report via Jinja2
│   └── templates/
│       └── report.html.j2      # Chart.js report template
└── tests/
    ├── test_sampler.py
    ├── test_analyzer.py
    ├── test_hooks.py
    └── test_integration.py
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Memory tracking | `torch.cuda` memory APIs + `tracemalloc` |
| Pattern detection | NumPy + SciPy linear regression |
| Background sampling | Python `threading` |
| Report generation | Jinja2 + Chart.js |
| Packaging | `pyproject.toml` + setuptools |
| Testing | pytest |

---

## License

MIT
