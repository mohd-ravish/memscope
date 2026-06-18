import torch
import torch.nn as nn
from .sampler import MemorySampler
from .hooks import HookManager
from .snapshot import parse_snapshot
from .analyzer import analyze
from .reporter import generate_report


class MemoryProfiler:
    def __init__(
        self,
        output_dir: str = "./memscope-report",
        sample_interval_ms: int = 500,
        track_layers: bool = True,
        model: nn.Module | None = None,
    ):
        self.output_dir = output_dir
        self.track_layers = track_layers

        self.sampler = MemorySampler(interval_ms=sample_interval_ms)
        self.hook_manager = HookManager()
        self._final_snapshot: dict | None = None

        if model is not None:
            self.hook_manager.register(model)

    def __enter__(self):
        self.sampler.start()
        if torch.cuda.is_available():
            try:
                torch.cuda.memory._record_memory_history(max_entries=100_000)
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sampler.stop()
        if torch.cuda.is_available():
            try:
                self._final_snapshot = torch.cuda.memory._snapshot()
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
        self.hook_manager.remove()
        return False

    def step(self):
        """Call once per training step to track step-level memory changes."""
        self.sampler.increment_step()

    def track_model(self, model: nn.Module):
        """Register forward hooks on a model for per-layer memory tracking."""
        self.hook_manager.register(model)

    def report(self) -> str:
        """Generate and open the HTML report. Returns the path to the report file."""
        analysis = analyze(self.sampler.samples)
        path = generate_report(
            samples=self.sampler.samples,
            analysis=analysis,
            snapshot=self._final_snapshot,
            layer_stats=self.hook_manager.stats,
            output_dir=self.output_dir,
        )
        return path
