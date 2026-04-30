"""Metric computation functions for UniVis.

All functions accept CPU tensors and return Python floats.
For batch>1 inputs, metrics are computed per-item then averaged across the batch.
"""

from __future__ import annotations

import torch


def compute_relative_delta(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> float:
    """Relative L2 change: ||output - input|| / ||input||.

    Small value means the layer barely changed the representation (redundant).
    For batch>1, returns mean across batch items.
    """
    delta = output_tensor - input_tensor
    if delta.dim() == 3:
        delta = delta[:, -1, :]
        input_tensor = input_tensor[:, -1, :]
    if delta.dim() == 2:
        norm_delta = delta.float().norm(dim=-1)
        norm_input = input_tensor.float().norm(dim=-1)
        safe = norm_input > 1e-10
        ratios = torch.where(
            safe, norm_delta / norm_input.clamp(min=1e-10), torch.zeros_like(norm_delta),
        )
        return ratios.mean().item()
    norm_delta = delta.float().norm().item()
    norm_input = input_tensor.float().norm().item()
    if norm_input < 1e-10:
        return 0.0
    return norm_delta / norm_input


def compute_cosine_sim(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> float:
    """Cosine similarity between input and output representations.

    Close to 1.0 means similar direction. Note: residual connections inflate
    this metric, so relative_delta is preferred as the primary indicator.
    For batch>1, returns mean across batch items.
    """
    if input_tensor.dim() == 3:
        input_tensor = input_tensor[:, -1, :]
        output_tensor = output_tensor[:, -1, :]
    if input_tensor.dim() == 2:
        return torch.nn.functional.cosine_similarity(
            input_tensor.float(), output_tensor.float(), dim=-1,
        ).mean().item()
    a = input_tensor.float().flatten()
    b = output_tensor.float().flatten()
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0),
    ).item()


def compute_sparsity(
    tensor: torch.Tensor,
    threshold: float = 1e-6,
) -> float:
    """Fraction of activations near zero.

    High value means many neurons are inactive.
    For batch>1, returns mean across batch items.
    """
    if tensor.dim() == 3:
        tensor = tensor[:, -1, :]
    if tensor.dim() == 2:
        return (tensor.abs() < threshold).float().mean(dim=-1).mean().item()
    return (tensor.abs() < threshold).float().mean().item()


def compute_entropy(logits: torch.Tensor) -> float:
    """Prediction entropy from final-layer logits.

    Low value means the model is confident about the next token.
    For batch>1, returns mean across batch items.
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean().item()
