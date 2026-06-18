import threading
import time
import torch
import tracemalloc
from dataclasses import dataclass, field


@dataclass
class MemorySample:
    timestamp: float
    step: int
    allocated_mb: float
    reserved_mb: float
    cpu_mb: float


class MemorySampler:
    def __init__(self, interval_ms: int = 500):
        self.interval = interval_ms / 1000.0
        self.samples: list[MemorySample] = []
        self._step = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self):
        tracemalloc.start()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="memscope-sampler")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        tracemalloc.stop()

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

            try:
                current, _ = tracemalloc.get_traced_memory()
                cpu_mb = current / 1024**2
            except Exception:
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
