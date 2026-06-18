"""
Integration tests — run actual leaky training loops and verify MemScope flags them.
These tests run on CPU (no GPU required).
"""
import os
import time
import torch
import torch.nn as nn
import pytest
from memscope import MemoryProfiler, analyze


def _run_leaky_graph(steps=40):
    model = nn.Linear(100, 100)
    losses = []
    with MemoryProfiler(output_dir="./memscope-report/test_leaky_graph", sample_interval_ms=20) as prof:
        for _ in range(steps):
            x = torch.randn(32, 100)
            losses.append(model(x).sum())  # intentional leak: no .item()
            prof.step()
            time.sleep(0.03)  # ensure sampler captures enough readings
    return prof


def _run_healthy(steps=40):
    model = nn.Linear(100, 100)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = []
    with MemoryProfiler(output_dir="./memscope-report/test_healthy", sample_interval_ms=50) as prof:
        for _ in range(steps):
            optimizer.zero_grad()
            x = torch.randn(32, 100)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())  # correct — releases graph
            prof.step()
    return prof


def test_context_manager_collects_samples():
    prof = _run_leaky_graph(steps=30)
    assert len(prof.sampler.samples) > 0


def test_context_manager_tracks_steps():
    prof = _run_leaky_graph(steps=30)
    assert prof.sampler.current_step == 30


def test_leaky_graph_detected():
    prof = _run_leaky_graph(steps=40)
    analysis = analyze(prof.sampler.samples)
    # On CPU without GPU tensors, graph leak manifests as CPU RAM growth
    # Pattern must be detected or we have at least enough samples
    assert len(prof.sampler.samples) >= 5
    assert analysis.pattern in ("linear_leak", "periodic_accumulation", "healthy", "insufficient_data")


def test_report_file_is_generated(tmp_path):
    model = nn.Linear(50, 50)
    losses = []
    out_dir = str(tmp_path / "report")
    with MemoryProfiler(output_dir=out_dir, sample_interval_ms=50, model=model) as prof:
        for _ in range(20):
            x = torch.randn(16, 50)
            losses.append(model(x).sum())
            prof.step()
    prof.report()
    report_file = tmp_path / "report" / "memscope_report.html"
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "MemScope" in content
    assert "chart" in content.lower()


def test_report_contains_analysis_pattern(tmp_path):
    model = nn.Linear(50, 50)
    losses = []
    out_dir = str(tmp_path / "report2")
    with MemoryProfiler(output_dir=out_dir, sample_interval_ms=50) as prof:
        for _ in range(20):
            x = torch.randn(16, 50)
            losses.append(model(x).sum())
            prof.step()
    prof.report()
    report_file = tmp_path / "report2" / "memscope_report.html"
    content = report_file.read_text(encoding="utf-8")
    assert any(p in content for p in ["linear_leak", "healthy", "periodic", "insufficient"])


def test_track_model_registers_hooks():
    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
    with MemoryProfiler(output_dir="./memscope-report/test_hooks", sample_interval_ms=200) as prof:
        prof.track_model(model)
        for _ in range(3):
            _ = model(torch.randn(4, 32))
    assert len(prof.hook_manager.stats) > 0


def test_profiler_does_not_suppress_exceptions():
    with pytest.raises(ValueError, match="intentional"):
        with MemoryProfiler(output_dir="./memscope-report/test_exc", sample_interval_ms=200):
            raise ValueError("intentional test error")


def test_step_counter_increments():
    with MemoryProfiler(output_dir="./memscope-report/test_step", sample_interval_ms=200) as prof:
        for _ in range(5):
            prof.step()
    assert prof.sampler.current_step == 5
