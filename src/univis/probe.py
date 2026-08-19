"""ModelProbe: register forward_hooks to capture per-layer metrics."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .metrics import cosine_sim_tensor, relative_delta_tensor, sparsity_tensor


class ModelProbe:
    """Attach forward hooks to Transformer blocks and buffer per-step metrics.

    Hooks compute metric tensors on the activation's device (no host sync);
    flush_step moves one step's scalars to CPU in a single batched transfer.
    """

    def __init__(
        self,
        model: nn.Module,
        hook_prefixes: list[str],
        metric_fns: dict[str, Callable[..., torch.Tensor]] | None = None,
    ) -> None:
        self.model = model
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []
        self._step_buffer: list[dict] = []
        self._metric_fns = metric_fns or {
            'relative_delta': relative_delta_tensor,
            'cosine_sim': cosine_sim_tensor,
            'sparsity': sparsity_tensor,
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

            entry: dict = {'idx': layer_idx, 'name': layer_name}
            for metric_name, fn in self._metric_fns.items():
                entry[metric_name] = fn(hidden_in, hidden_out)
            self._step_buffer.append(entry)

        return hook_fn

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def flush_step(self, token_index: int) -> list[dict]:
        """Materialize buffered layer metrics for this step as Python floats."""
        data = list(self._step_buffer)
        if data:
            metric_keys = [k for k in data[0] if k not in ('idx', 'name')]
            for key in metric_keys:
                vals = [d[key] for d in data]
                if isinstance(vals[0], torch.Tensor):
                    on_cpu = torch.stack(vals).cpu()
                    for d, v in zip(data, on_cpu.tolist()):
                        d[key] = v
            for entry in data:
                entry['token_idx'] = token_index
        self._step_buffer.clear()
        return data

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    @property
    def num_hooks(self) -> int:
        return len(self._hook_handles)
