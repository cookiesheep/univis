"""Unit tests for probe and detection."""

import torch
import torch.nn as nn
import pytest

from univis.detection import detect_block_prefixes, get_layer_count
from univis.probe import ModelProbe
from univis.transport import FileTransport

import tempfile
from pathlib import Path


# ---- Helper: minimal GPT-2-like model ----

class FakeBlock(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x + self.linear(x),)


class FakeGPT2(nn.Module):
    def __init__(self, n_layers: int = 4, dim: int = 32) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([FakeBlock(dim) for _ in range(n_layers)])
        self.lm_head = nn.Linear(dim, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer.h:
            x = block(x)[0]
        return self.lm_head(x)


class TestDetection:
    def test_detects_gpt2_prefix(self) -> None:
        model = FakeGPT2(n_layers=4)
        prefixes = detect_block_prefixes(model)
        assert prefixes == ['transformer.h.']

    def test_layer_count(self) -> None:
        model = FakeGPT2(n_layers=6)
        assert get_layer_count(model, 'transformer.h.') == 6


class TestProbe:
    def test_registers_correct_hooks(self) -> None:
        model = FakeGPT2(n_layers=3)
        probe = ModelProbe(model, ['transformer.h.'])
        assert probe.num_hooks == 3
        probe.remove_hooks()

    def test_collects_metrics_on_forward(self) -> None:
        model = FakeGPT2(n_layers=3, dim=16)
        probe = ModelProbe(model, ['transformer.h.'])

        x = torch.randn(1, 5, 16)
        with torch.no_grad():
            model(x)

        data = probe.flush_step(0)
        assert len(data) == 3
        for entry in data:
            assert 'relative_delta' in entry
            assert 'cosine_sim' in entry
            assert 'sparsity' in entry
            assert isinstance(entry['relative_delta'], float)
        probe.remove_hooks()

    def test_hooks_removed_cleanly(self) -> None:
        model = FakeGPT2(n_layers=2)
        probe = ModelProbe(model, ['transformer.h.'])
        assert probe.num_hooks == 2
        probe.remove_hooks()
        assert probe.num_hooks == 0


class TestFileTransport:
    def test_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.jsonl'
            transport = FileTransport(path)
            transport.send({'type': 'step', 'token_idx': 0})
            transport.send({'type': 'step', 'token_idx': 1})
            transport.close()

            lines = path.read_text().strip().split('\n')
            assert len(lines) == 2
            import json
            msg = json.loads(lines[0])
            assert msg['token_idx'] == 0
