import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class LeakAnalysis:
    pattern: str
    severity: str
    growth_mb_per_step: float
    oom_predicted_at_step: int | None
    message: str
    recommendation: str


def analyze(samples) -> LeakAnalysis:
    if len(samples) < 10:
        return LeakAnalysis(
            pattern="insufficient_data",
            severity="none",
            growth_mb_per_step=0.0,
            oom_predicted_at_step=None,
            message="Not enough samples to analyze. Run for more steps.",
            recommendation="Ensure training runs for at least 10 sampler intervals (~5 seconds at default settings).",
        )

    gpu_vals = np.array([s.allocated_mb for s in samples])
    cpu_vals = np.array([s.cpu_mb for s in samples])
    # Use CPU RAM when no GPU is present (all GPU values are zero)
    y = cpu_vals if gpu_vals.max() == 0.0 else gpu_vals
    x = np.arange(len(y))

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)

    # Pattern 1: Linear growth — retained computation graph or growing tensor list
    if slope > 0.5 and r_value**2 > 0.85 and p_value < 0.01:
        gpu_total = _get_gpu_total_mb()
        steps_to_oom = None
        if gpu_total and slope > 0:
            remaining = gpu_total - y[-1]
            if remaining > 0:
                steps_to_oom = int(remaining / slope) + len(y)

        return LeakAnalysis(
            pattern="linear_leak",
            severity="high",
            growth_mb_per_step=round(slope, 2),
            oom_predicted_at_step=steps_to_oom,
            message=(
                f"Memory growing at {slope:.1f} MB/step (R²={r_value**2:.2f}). "
                "Classic linear leak — memory is not being freed between steps."
            ),
            recommendation=(
                "Check for:\n"
                "  1. Loss accumulation — replace `losses.append(loss)` with `losses.append(loss.item())`\n"
                "  2. Tensors stored without `.detach()` — use `tensor.detach().cpu()` before storing\n"
                "  3. Model outputs appended to a list inside the training loop without detaching"
            ),
        )

    # Pattern 2: Periodic spikes — list appended every N steps, cleared sometimes
    diffs = np.diff(y)
    big_drops = np.where(diffs < -50)[0]
    big_spikes = np.where(diffs > 50)[0]
    if len(big_drops) >= 2 and len(big_spikes) >= 2:
        period = int(np.mean(np.diff(big_drops))) if len(big_drops) > 1 else 0
        return LeakAnalysis(
            pattern="periodic_accumulation",
            severity="medium",
            growth_mb_per_step=round(slope, 2),
            oom_predicted_at_step=None,
            message=(
                f"Periodic memory spikes every ~{period} steps. "
                "Tensors accumulating then partially releasing — typical of validation loops."
            ),
            recommendation=(
                "Check for lists or dicts that collect tensors across batches.\n"
                "  - Call `.detach().cpu()` before storing any tensor\n"
                "  - Clear accumulation buffers after each epoch: `val_outputs.clear()`\n"
                "  - Use `torch.no_grad()` during validation"
            ),
        )

    # Pattern 3: High but flat — over-allocated, not a leak (GPU only)
    gpu_total = _get_gpu_total_mb()
    if gpu_total and gpu_vals.max() > 0 and np.mean(y) > 0.85 * gpu_total:
        return LeakAnalysis(
            pattern="over_allocation",
            severity="low",
            growth_mb_per_step=round(slope, 2),
            oom_predicted_at_step=None,
            message=(
                f"Memory is high ({np.mean(y):.0f} MB / {gpu_total:.0f} MB) but not growing. "
                "Not a leak — model or batch size is too large for this GPU."
            ),
            recommendation=(
                "Reduce batch size or enable gradient checkpointing:\n"
                "  `from torch.utils.checkpoint import checkpoint`\n"
                "  Use mixed precision: `torch.cuda.amp.autocast()`"
            ),
        )

    return LeakAnalysis(
        pattern="healthy",
        severity="none",
        growth_mb_per_step=round(slope, 2),
        oom_predicted_at_step=None,
        message="Memory is stable. No leak pattern detected.",
        recommendation="No action needed.",
    )


def _get_gpu_total_mb() -> float | None:
    try:
        import torch
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / 1024**2
    except Exception:
        return None
