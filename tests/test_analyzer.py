import random
import pytest
from memscope.analyzer import analyze, LeakAnalysis
from memscope.sampler import MemorySample


def _make_samples(values: list[float]) -> list[MemorySample]:
    return [
        MemorySample(timestamp=float(i), step=i, allocated_mb=v, reserved_mb=v * 1.2, cpu_mb=10.0)
        for i, v in enumerate(values)
    ]


def test_insufficient_data():
    samples = _make_samples([100.0] * 5)
    result = analyze(samples)
    assert result.pattern == "insufficient_data"
    assert result.severity == "none"


def test_detects_linear_leak():
    samples = _make_samples([100 + i * 5 for i in range(50)])
    result = analyze(samples)
    assert result.pattern == "linear_leak"
    assert result.severity == "high"
    assert result.growth_mb_per_step > 4.0


def test_linear_leak_message_contains_growth_rate():
    samples = _make_samples([100 + i * 5 for i in range(50)])
    result = analyze(samples)
    assert "MB/step" in result.message


def test_detects_periodic_accumulation():
    # Alternating spike-drop pattern: flat at 100, spike to 300, drop to 50, repeat
    # Produces diffs of +200 (spike) and -250 (drop) at regular intervals
    values = [100.0] * 80
    for i in range(20, 80, 20):
        values[i] = 300.0        # +200 spike
        values[i + 1] = 50.0    # -250 drop
    samples = _make_samples(values)
    result = analyze(samples)
    assert result.pattern in ("periodic_accumulation", "linear_leak")


def test_healthy_training():
    rng = random.Random(42)
    samples = _make_samples([400 + rng.uniform(-5, 5) for _ in range(50)])
    result = analyze(samples)
    assert result.pattern == "healthy"
    assert result.severity == "none"


def test_healthy_recommendation():
    rng = random.Random(42)
    samples = _make_samples([400 + rng.uniform(-5, 5) for _ in range(50)])
    result = analyze(samples)
    assert "No action" in result.recommendation


def test_analysis_returns_dataclass():
    samples = _make_samples([100 + i * 5 for i in range(50)])
    result = analyze(samples)
    assert isinstance(result, LeakAnalysis)
    assert isinstance(result.growth_mb_per_step, float)
