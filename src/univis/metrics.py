"""Metric computation functions for UniVis.

All functions accept CPU tensors and return Python floats.
"""

from __future__ import annotations

import torch


def compute_relative_delta(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> float:
    """Relative L2 change: ||output - input|| / ||input||.

    Small value means the layer barely changed the representation (redundant).
    """
    delta = output_tensor - input_tensor
    # For 3D tensors (batch, seq, dim), take last token position
    if delta.dim() == 3:
        delta = delta[:, -1, :]
        input_tensor = input_tensor[:, -1, :]
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
    """
    if input_tensor.dim() == 3:
        input_tensor = input_tensor[:, -1, :]
        output_tensor = output_tensor[:, -1, :]
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
    """
    if tensor.dim() == 3:
        tensor = tensor[:, -1, :]
    return (tensor.abs() < threshold).float().mean().item()


def compute_entropy(logits: torch.Tensor) -> float:
    """Prediction entropy from final-layer logits.

    Low value means the model is confident about the next token.
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).item()
