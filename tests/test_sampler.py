import time
import pytest
from memscope.sampler import MemorySampler, MemorySample


def test_sampler_starts_and_stops():
    sampler = MemorySampler(interval_ms=100)
    sampler.start()
    time.sleep(0.35)
    sampler.stop()
    assert len(sampler.samples) >= 2


def test_sampler_collects_memory_sample_fields():
    sampler = MemorySampler(interval_ms=100)
    sampler.start()
    time.sleep(0.25)
    sampler.stop()
    assert len(sampler.samples) > 0
    s = sampler.samples[0]
    assert isinstance(s, MemorySample)
    assert isinstance(s.timestamp, float)
    assert isinstance(s.allocated_mb, float)
    assert isinstance(s.reserved_mb, float)
    assert isinstance(s.cpu_mb, float)
    assert s.cpu_mb >= 0


def test_increment_step():
    sampler = MemorySampler(interval_ms=200)
    assert sampler.current_step == 0
    sampler.increment_step()
    sampler.increment_step()
    assert sampler.current_step == 2


def test_sampler_step_recorded_in_samples():
    sampler = MemorySampler(interval_ms=50)
    sampler.start()
    time.sleep(0.08)
    sampler.increment_step()
    time.sleep(0.08)
    sampler.stop()
    steps = [s.step for s in sampler.samples]
    assert 0 in steps
    assert 1 in steps


def test_sampler_stop_is_idempotent():
    sampler = MemorySampler(interval_ms=100)
    sampler.start()
    time.sleep(0.15)
    sampler.stop()
    sampler.stop()  # should not raise
