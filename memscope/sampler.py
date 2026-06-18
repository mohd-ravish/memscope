import os
import threading
import time
import torch
from dataclasses import dataclass

try:
    import psutil
    _PROCESS = psutil.Process(os.getpid())
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class MemorySample:
    timestamp: float
    step: int
    allocated_mb: float   # GPU: torch.cuda.memory_allocated()
    reserved_mb: float    # GPU: torch.cuda.memory_reserved()
    cpu_mb: float         # CPU: process RSS via psutil (includes PyTorch C++ tensors)


class MemorySampler:
    def __init__(self, interval_ms: int = 500):
        self.interval = interval_ms / 1000.0
        self.samples: list[MemorySample] = []
        self._step = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="memscope-sampler")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def increment_step(self):
        with self._lock:
            self._step += 1

    @property
    def current_step(self) -> int:
        with self._lock:
            return self._step

    def _run(self):
        while self._running:
            allocated = 0.0
            reserved = 0.0
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2

            # psutil RSS captures real process memory including PyTorch C++ allocations.
            # tracemalloc only sees Python objects and misses tensor data entirely.
            if _HAS_PSUTIL:
                try:
                    cpu_mb = _PROCESS.memory_info().rss / 1024**2
                except Exception:
                    cpu_mb = 0.0
            else:
                cpu_mb = 0.0

            with self._lock:
                step = self._step

            self.samples.append(MemorySample(
                timestamp=time.time(),
                step=step,
                allocated_mb=round(allocated, 2),
                reserved_mb=round(reserved, 2),
                cpu_mb=round(cpu_mb, 2),
            ))
            time.sleep(self.interval)
