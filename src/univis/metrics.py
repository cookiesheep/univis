"""Metric computation functions for UniVis.

Tensor variants return 0-dim tensors (computed on the input device, no host
sync) — ModelProbe batches them into a single CPU transfer per step. Public
float wrappers are kept for direct use and tests.
For batch>1 inputs, metrics are computed per-item then averaged across the batch.
"""

from __future__ import annotations

import torch


def _last_token(t: torch.Tensor) -> torch.Tensor:
    """Reduce [batch, seq, hidden] to the last token [batch, hidden]."""
    return t[:, -1, :] if t.dim() == 3 else t


def relative_delta_tensor(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> torch.Tensor:
    """Relative L2 change as 0-dim tensor: ||output - input|| / ||input||."""
    delta = _last_token(output_tensor - input_tensor).float()
    inp = _last_token(input_tensor).float()
    norm_delta = delta.norm(dim=-1)
    norm_input = inp.norm(dim=-1)
    safe = norm_input > 1e-10
    ratios = torch.where(
        safe, norm_delta / norm_input.clamp(min=1e-10), torch.zeros_like(norm_delta),
    )
    return ratios.mean()


def cosine_sim_tensor(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity between input and output as 0-dim tensor."""
    a = _last_token(input_tensor).float()
    b = _last_token(output_tensor).float()
    if a.dim() == 1:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()


def sparsity_tensor(tensor: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    """Fraction of near-zero activations as 0-dim tensor."""
    return (_last_token(tensor).abs() < threshold).float().mean()


def entropy_tensor(logits: torch.Tensor) -> torch.Tensor:
    """Prediction entropy of the next-token distribution as 0-dim tensor."""
    logits = _last_token(logits).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


def compute_relative_delta(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> float:
    """Relative L2 change: ||output - input|| / ||input||.

    Small value means the layer barely changed the representation (redundant).
    For batch>1, returns mean across batch items.
    """
    return relative_delta_tensor(input_tensor, output_tensor).item()


def compute_cosine_sim(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> float:
    """Cosine similarity between input and output representations.

    Close to 1.0 means similar direction. Note: residual connections inflate
    this metric, so relative_delta is preferred as the primary indicator.
    For batch>1, returns mean across batch items.
    """
    return cosine_sim_tensor(input_tensor, output_tensor).item()


def compute_sparsity(tensor: torch.Tensor, threshold: float = 1e-6) -> float:
    """Fraction of activations near zero.

    High value means many neurons are inactive.
    For batch>1, returns mean across batch items.
    """
    return sparsity_tensor(tensor, threshold).item()


def compute_entropy(logits: torch.Tensor) -> float:
    """Prediction entropy from final-layer logits.

    Low value means the model is confident about the next token.
    For batch>1, returns mean across batch items.
    """
    return entropy_tensor(logits).item()
