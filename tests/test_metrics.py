"""Unit tests for metrics computation."""

import torch
import pytest

from univis.metrics import (
    compute_cosine_sim,
    compute_entropy,
    compute_relative_delta,
    compute_sparsity,
)


class TestRelativeDelta:
    def test_identical_tensors(self) -> None:
        t = torch.tensor([[1.0, 2.0, 3.0]])
        assert compute_relative_delta(t, t) == 0.0

    def test_different_tensors(self) -> None:
        inp = torch.tensor([[1.0, 0.0, 0.0]])
        out = torch.tensor([[0.0, 1.0, 0.0]])
        result = compute_relative_delta(inp, out)
        assert result > 0.0

    def test_3d_takes_last_position(self) -> None:
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        out = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        assert compute_relative_delta(inp, out) == 0.0

    def test_zero_input(self) -> None:
        inp = torch.zeros(1, 3)
        out = torch.ones(1, 3)
        assert compute_relative_delta(inp, out) == 0.0


class TestCosineSim:
    def test_identical(self) -> None:
        t = torch.tensor([[1.0, 2.0, 3.0]])
        assert abs(compute_cosine_sim(t, t) - 1.0) < 1e-5

    def test_opposite(self) -> None:
        t = torch.tensor([[1.0, 0.0, 0.0]])
        neg = torch.tensor([[-1.0, 0.0, 0.0]])
        assert abs(compute_cosine_sim(t, neg) - (-1.0)) < 1e-5

    def test_orthogonal(self) -> None:
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        assert abs(compute_cosine_sim(a, b)) < 1e-5


class TestSparsity:
    def test_all_zero(self) -> None:
        t = torch.zeros(1, 10)
        assert compute_sparsity(t) == 1.0

    def test_no_zeros(self) -> None:
        t = torch.ones(1, 10)
        assert compute_sparsity(t) == 0.0

    def test_half_zero(self) -> None:
        t = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        assert compute_sparsity(t) == 0.5


class TestEntropy:
    def test_uniform_distribution(self) -> None:
        logits = torch.zeros(1, 10)
        entropy = compute_entropy(logits)
        assert entropy > 0

    def test_peak_distribution(self) -> None:
        logits = torch.tensor([[100.0] + [-100.0] * 9])
        entropy = compute_entropy(logits)
        assert entropy < 0.01

    def test_3d_takes_last_position(self) -> None:
        logits = torch.zeros(1, 5, 10)
        result = compute_entropy(logits)
        assert result > 0
