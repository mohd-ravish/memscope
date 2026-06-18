import torch
import torch.nn as nn
import pytest
from memscope.hooks import HookManager


def test_hook_manager_registers_and_captures():
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
    manager = HookManager()
    manager.register(model)

    x = torch.randn(8, 64)
    _ = model(x)

    assert len(manager.stats) > 0
    # At least one layer should have recorded output size
    all_values = [v for vals in manager.stats.values() for v in vals]
    assert len(all_values) > 0
    assert all(isinstance(v, float) and v >= 0 for v in all_values)


def test_hook_manager_remove_clears_handles():
    model = nn.Linear(32, 16)
    manager = HookManager()
    manager.register(model)
    manager.remove()

    # After removal, no handles remain
    assert len(manager._handles) == 0


def test_hook_manager_stats_accumulate_over_forward_passes():
    model = nn.Linear(64, 32)
    manager = HookManager()
    manager.register(model)

    for _ in range(5):
        _ = model(torch.randn(4, 64))

    all_counts = sum(len(v) for v in manager.stats.values())
    assert all_counts >= 5


def test_hook_manager_no_crash_on_empty_model():
    model = nn.Module()  # no submodules
    manager = HookManager()
    manager.register(model)
    manager.remove()


def test_hook_captures_linear_layer_output_size():
    model = nn.Linear(100, 50)
    manager = HookManager()
    manager.register(model)

    batch_size = 8
    _ = model(torch.randn(batch_size, 100))

    all_values = [v for vals in manager.stats.values() for v in vals]
    # Output is (8, 50) float32 = 8*50*4 bytes = 1600 bytes = ~0.00153 MB
    assert any(v > 0 for v in all_values)
