"""ModelProbe: register forward_hooks to capture per-layer metrics."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .metrics import compute_cosine_sim, compute_relative_delta, compute_sparsity


class ModelProbe:
    """Attach forward hooks to Transformer blocks and buffer per-step metrics."""

    def __init__(
        self,
        model: nn.Module,
        hook_prefixes: list[str],
        metric_fns: dict[str, Callable[..., float]] | None = None,
    ) -> None:
        self.model = model
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []
        self._step_buffer: list[dict] = []
        self._metric_fns = metric_fns or {
            'relative_delta': lambda inp, out: compute_relative_delta(inp, out),
            'cosine_sim': lambda inp, out: compute_cosine_sim(inp, out),
            'sparsity': lambda inp, out: compute_sparsity(out),
        }
        self._register_hooks(hook_prefixes)

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self, prefixes: list[str]) -> None:
        for name, module in self.model.named_modules():
            if self._is_block_module(name, prefixes):
                idx = self._extract_index(name)
                handle = module.register_forward_hook(self._make_hook(name, idx))
                self._hook_handles.append(handle)

    @staticmethod
    def _is_block_module(name: str, prefixes: list[str]) -> bool:
        for prefix in prefixes:
            if name.startswith(prefix):
                remainder = name[len(prefix):]
                # Accept exact index like "transformer.h.3"
                if remainder.isdigit():
                    return True
        return False

    @staticmethod
    def _extract_index(name: str) -> int:
        return int(name.rsplit('.', 1)[-1])

    def _make_hook(self, layer_name: str, layer_idx: int) -> Callable:
        def hook_fn(
            module: nn.Module,
            inp: tuple[torch.Tensor, ...],
            output: tuple | torch.Tensor,
        ) -> None:
            hidden_in = inp[0]
            hidden_out = output[0] if isinstance(output, tuple) else output
            # Move to CPU once, compute all metrics
            cpu_in = hidden_in.detach().cpu()
            cpu_out = hidden_out.detach().cpu()

            entry: dict = {'idx': layer_idx, 'name': layer_name}
            for metric_name, fn in self._metric_fns.items():
                entry[metric_name] = fn(cpu_in, cpu_out)
            self._step_buffer.append(entry)

        return hook_fn

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def flush_step(self, token_index: int) -> list[dict]:
        """Return buffered layer metrics for the current step and reset."""
        data = list(self._step_buffer)
        self._step_buffer.clear()
        return data

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    @property
    def num_hooks(self) -> int:
        return len(self._hook_handles)
