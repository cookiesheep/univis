"""Auto-detect Transformer block structure in a model."""

from __future__ import annotations

import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


# Known block name prefixes for common architectures
_BLOCK_PREFIXES = [
    'transformer.h.',       # GPT-2
    'model.layers.',        # LLaMA, Qwen, Mistral
    'encoder.layer.',       # BERT encoder
    'decoder.block.',       # some decoder architectures
    'transformer.blocks.',  # some custom models
]


def detect_block_prefixes(model: nn.Module) -> list[str]:
    """Detect Transformer block prefixes by scanning named_modules.

    Returns a list of matching prefixes (usually just one).
    Returns empty list and logs a warning if no known pattern matches.
    """
    names = [name for name, _ in model.named_modules()]
    for prefix in _BLOCK_PREFIXES:
        matching = [n for n in names if n.startswith(prefix)]
        # Filter to exact block-level (no deeper dots after the index)
        block_names = [
            n for n in matching
            if n[len(prefix):].isdigit()
            or (n[len(prefix):].split('.')[0].isdigit()
                and '.' not in n[len(prefix):])
        ]
        if block_names:
            return [prefix]

    # Fallback: look for modules containing 'block' or 'layer' at depth 1-2
    fallback = []
    for name, _ in model.named_modules():
        parts = name.split('.')
        if 1 <= len(parts) <= 2:
            lower = parts[-1].lower()
            if lower.startswith(('block', 'layer', 'h')):
                prefix_candidate = name.rsplit('.', 1)[0] + '.'
                if prefix_candidate not in fallback:
                    fallback.append(prefix_candidate)
    if fallback:
        return fallback

    logger.warning(
        'No known Transformer block pattern detected in model. '
        'Top-level modules: %s. '
        'Pass hook_prefixes manually, e.g. hook_prefixes=["model.layers."]',
        [n for n, _ in list(model.named_modules())[:20]],
    )
    return []


def get_layer_count(model: nn.Module, prefix: str) -> int:
    """Count transformer blocks matching the given prefix."""
    count = 0
    for name, _ in model.named_modules():
        if name.startswith(prefix):
            remainder = name[len(prefix):]
            if remainder.isdigit() and '.' not in remainder:
                count += 1
    return count
