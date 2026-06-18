import torch
import torch.nn as nn
from collections import defaultdict


class HookManager:
    def __init__(self):
        self.stats: dict[str, list[float]] = defaultdict(list)
        self._handles = []

    def register(self, model: nn.Module):
        for name, module in model.named_modules():
            handle = module.register_forward_hook(self._make_hook(name or "root"))
            self._handles.append(handle)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                mb = output.element_size() * output.nelement() / 1024**2
                self.stats[name].append(round(mb, 3))
            elif isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        mb = item.element_size() * item.nelement() / 1024**2
                        self.stats[name].append(round(mb, 3))
                        break
        return hook
