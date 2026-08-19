"""Pilot mode: automatic inference intervention — skip redundant layers + early-exit.

Pilot applies a PilotPolicy (derived from Monitor data) to a model:
- layer-skip: redundant layers (high cosine similarity) get their forward patched
  to pass input through, skipping the layer's computation.
- early-exit: terminate generation when prediction entropy drops below threshold.

Constraints (enforced/documented):
- layer-skip is incompatible with KV cache (skipped layers produce no present
  key/value) -> generation MUST use use_cache=False.
- Pilot is incompatible with output_attentions / output_hidden_states (the
  pass-through only returns the hidden state).
- early-exit is validated for greedy + multinomial sampling; beam search is
  NOT validated.

Usage (context manager — restores forwards even on exception):
    policy = PilotPolicy.from_layer_summary(summary, cos_threshold=0.95)
    with Pilot(model, policy) as pilot:
        out = model.generate(input_ids, max_new_tokens=50, use_cache=False)
    # forwards restored here, even if generate() raised
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

# HF changed decoder-layer return API: <5 returns a tuple (model unpacks [0]);
# >=5 returns the tensor directly. Pilot's pass-through must match.
try:
    import transformers as _tf
    _WRAPS_TUPLE = int(_tf.__version__.split('.')[0]) < 5
except Exception:
    _WRAPS_TUPLE = True

# Match standard decoder-layer paths: model.layers.N / transformer.h.N / blocks.N.
# Narrower than "any trailing digit" to avoid patching non-decoder modules.
_LAYER_RE = re.compile(r'(?:layers|h|blocks)\.(\d+)$')


@dataclass
class PilotPolicy:
    """Intervention policy — which layers to skip, when to early-exit.

    skip_layers is snapshot-at-apply: mutating it after Pilot.apply() has no
    effect on already-patched layers.

    entropy_window: consecutive low-entropy steps required before forcing EOS.
    1 = fire on the first low-entropy step (naive). Chat models emit ultra-
    confident tokens at answer openings, so a window > 1 is usually needed to
    exit on completion rather than on the opening.
    """

    skip_layers: set[int] = field(default_factory=set)
    entropy_threshold: float = 0.1
    early_exit_enabled: bool = True
    entropy_window: int = 1

    @classmethod
    def from_layer_summary(
        cls,
        layer_summary: list[dict],
        cos_threshold: float = 0.95,
        max_skip_ratio: float = 0.3,
        entropy_threshold: float = 0.1,
    ) -> 'PilotPolicy':
        """Build policy from Monitor's per-layer summary.

        Marks layers with avg_cosim >= cos_threshold as skippable, capped at
        max_skip_ratio of total layers to avoid over-aggressive skipping.
        """
        if not layer_summary:
            return cls(entropy_threshold=entropy_threshold)
        candidates = [
            l for l in layer_summary
            if l.get('avg_cosim', 0) >= cos_threshold
        ]
        candidates.sort(key=lambda l: l.get('avg_cosim', 0), reverse=True)
        cap = max(1, int(len(layer_summary) * max_skip_ratio))
        skip = {l['idx'] for l in candidates[:cap]}
        return cls(
            skip_layers=skip,
            entropy_threshold=entropy_threshold,
        )


class Pilot:
    """Apply layer-skip + early-exit intervention to a model.

    Use as a context manager to guarantee forwards are restored even if
    generation raises. Original forwards are stored keyed by id(module) with a
    module reference, so restore does not depend on re-enumerating named_modules.
    """

    def __init__(self, model: nn.Module, policy: PilotPolicy) -> None:
        self._model = model
        self._policy = policy
        # id(module) -> (module, original_forward)
        self._original_forwards: dict[int, tuple[nn.Module, Callable]] = {}
        self._patched = False
        self.early_exit_count = 0

    # ------------------------------------------------------------------
    # context manager — exception-safe restore
    # ------------------------------------------------------------------
    def __enter__(self) -> 'Pilot':
        self.apply()
        return self

    def __exit__(self, *exc) -> bool:
        self.restore()
        return False

    # ------------------------------------------------------------------
    # layer-skip
    # ------------------------------------------------------------------
    def apply(self) -> int:
        """Patch forward of skip_layers to pass-through. Return count patched."""
        if self._model.training:
            warnings.warn(
                'Pilot.apply() on a model in training mode: skipped layers '
                'receive no gradient. Call model.eval() first.',
                stacklevel=2,
            )
        if self._patched:
            return len(self._original_forwards)
        count = 0
        for name, module in self._model.named_modules():
            idx = self._extract_layer_idx(name)
            if idx is not None and idx in self._policy.skip_layers:
                self._patch_forward(module)
                count += 1
        self._patched = True
        return count

    def _patch_forward(self, module: nn.Module) -> None:
        """Replace module.forward with a pass-through. Idempotent per module."""
        if id(module) in self._original_forwards:
            return
        self._original_forwards[id(module)] = (module, module.forward)
        wraps = _WRAPS_TUPLE

        def pass_through(*args, **kwargs):
            hidden = args[0] if args else kwargs.get('hidden_states')
            return (hidden,) if wraps else hidden

        module.forward = pass_through

    def restore(self) -> None:
        """Restore original forward methods from stored references."""
        for module, orig_fwd in self._original_forwards.values():
            module.forward = orig_fwd
        self._original_forwards.clear()
        self._patched = False

    # ------------------------------------------------------------------
    # early-exit
    # ------------------------------------------------------------------
    def logits_processor(self, eos_token_id: int | None = None) -> Callable:
        """Return a logits processor that forces EOS on low-entropy steps.

        Validated for greedy + multinomial sampling. Beam search NOT supported.
        With policy.entropy_window > 1, EOS is forced only after that many
        consecutive steps below the threshold.
        """
        policy = self._policy
        pilot = self
        streak = 0

        def processor(input_ids, scores):
            nonlocal streak
            if not policy.early_exit_enabled or eos_token_id is None:
                return scores
            if eos_token_id < 0 or eos_token_id >= scores.shape[-1]:
                return scores  # invalid eos id — bail out safely
            with torch.no_grad():
                probs = torch.softmax(scores.float(), dim=-1)
                entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)
                if entropy.mean().item() < policy.entropy_threshold:
                    streak += 1
                    if streak >= policy.entropy_window:
                        pilot.early_exit_count += 1
                        streak = 0
                        masked = scores.new_full(scores.shape, float('-inf'))
                        masked[:, eos_token_id] = 0.0
                        return masked
                else:
                    streak = 0
            return scores

        return processor

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_layer_idx(name: str) -> int | None:
        """Extract layer index from decoder-layer paths like 'model.layers.5'."""
        m = _LAYER_RE.search(name)
        return int(m.group(1)) if m else None

    @property
    def skipped_layer_count(self) -> int:
        """Number of layers currently patched (skipped)."""
        return len(self._original_forwards)
