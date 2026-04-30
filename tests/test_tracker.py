"""Tests for Tracker: logits_processor, report integration, finish."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from univis.tracker import Tracker, create_tracker
from univis.probe import ModelProbe
from univis.transport import FileTransport


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
        self.config = type('Cfg', (), {'_name_or_path': 'fake-gpt2'})()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer.h:
            x = block(x)[0]
        return self.lm_head(x)


def _make_tracker(tmpdir):
    model = FakeGPT2(n_layers=2, dim=16)
    return create_tracker(model, transport_mode='file', output_dir=tmpdir)


class TestLogitsProcessor:
    def test_processor_calls_on_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = _make_tracker(tmpdir)
            processor = tracker.logits_processor()

            logits = torch.randn(1, 100)
            result = processor(torch.tensor([[1, 2, 3]]), logits)

            assert torch.equal(result, logits)  # passthrough
            assert tracker._step_count == 1

            logits2 = torch.randn(1, 100)
            processor(torch.tensor([[1, 2, 3, 4]]), logits2)
            assert tracker._step_count == 2

            tracker.finish()

    def test_processor_noop_after_finish(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = _make_tracker(tmpdir)
            processor = tracker.logits_processor()

            logits = torch.randn(1, 100)
            processor(torch.tensor([[1]]), logits)
            assert tracker._step_count == 1

            tracker.finish()

            # After finish, processor should be a no-op
            processor(torch.tensor([[1, 2]]), torch.randn(1, 100))
            assert tracker._step_count == 1  # unchanged

    def test_processor_with_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = _make_tracker(tmpdir)

            # Minimal tokenizer mock
            class MockTokenizer:
                def decode(self, token_id):
                    return f'tok_{token_id.item()}'

            processor = tracker.logits_processor(tokenizer=MockTokenizer())

            logits = torch.zeros(1, 100)
            logits[0, 42] = 10.0  # argmax → token 42
            processor(torch.tensor([[1]]), logits)

            assert tracker._all_steps[-1]['generated_token'] == 'tok_42'
            tracker.finish()


class TestReportIntegration:
    def test_finish_generates_echarts_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = _make_tracker(tmpdir)
            model = tracker._model

            x = torch.randn(1, 5, 16)
            with torch.no_grad():
                for i in range(3):
                    out = model(x)
                    tracker.on_step(
                        token_index=i,
                        generated_token=f'tok{i}',
                        logits=out[:, -1, :],
                    )

            report_path = tracker.finish()
            assert Path(report_path).exists()
            html = Path(report_path).read_text(encoding='utf-8')
            assert 'echarts' in html
            assert 'heatmap' in html
            assert 'tok0' in html

    def test_finish_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = _make_tracker(tmpdir)
            x = torch.randn(1, 3, 16)
            with torch.no_grad():
                out = tracker._model(x)
                tracker.on_step(0, 'a', out[:, -1, :])

            path1 = tracker.finish()
            path2 = tracker.finish()
            assert path1 != ''
            assert path2 == ''

    def test_output_dir_from_create_tracker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = _make_tracker(tmpdir)
            x = torch.randn(1, 3, 16)
            with torch.no_grad():
                out = tracker._model(x)
                tracker.on_step(0, 'a', out[:, -1, :])

            report_path = tracker.finish()
            assert str(tmpdir) in report_path
