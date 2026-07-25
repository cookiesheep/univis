"""End-to-end integration tests: SDK -> JSONL -> CLI report pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

import univis
from univis.report import generate_report
from univis.tracker import create_tracker


class FakeBlock(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x + self.linear(x),)


class FakeGPT2(nn.Module):
    def __init__(self, n_layers: int = 3, dim: int = 16, vocab: int = 50) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([FakeBlock(dim) for _ in range(n_layers)])
        self.lm_head = nn.Linear(dim, vocab)
        self.config = type('Cfg', (), {'_name_or_path': 'fake-gpt2'})()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer.h:
            x = block(x)[0]
        return self.lm_head(x)


N_LAYERS = 3
DIM = 16
VOCAB = 50


def _make_model() -> FakeGPT2:
    return FakeGPT2(n_layers=N_LAYERS, dim=DIM, vocab=VOCAB)


def _read_jsonl_lines(path: Path) -> list[dict]:
    messages = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


class TestFullSdkPipeline:
    """SDK attach() -> inference loop -> finish() -> verify HTML report."""

    def test_manual_inference_loop_generates_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            tracker = univis.attach(model, transport='file', output_dir=tmpdir)

            x = torch.randn(1, 5, DIM)
            num_steps = 4
            with torch.no_grad():
                for i in range(num_steps):
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
            for i in range(num_steps):
                assert f'tok{i}' in html


class TestCliReportFromJsonl:
    """Read generated JSONL, call generate_report() directly (simulates CLI)."""

    def test_generate_report_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            tracker = create_tracker(model, transport_mode='file', output_dir=tmpdir)

            x = torch.randn(1, 4, DIM)
            with torch.no_grad():
                for i in range(3):
                    out = model(x)
                    tracker.on_step(token_index=i, generated_token=f'word{i}', logits=out[:, -1, :])

            tracker.finish()

            # Find the JSONL file
            jsonl_files = list(Path(tmpdir).glob('univis_data_*.jsonl'))
            assert len(jsonl_files) == 1
            jsonl_path = jsonl_files[0]

            messages = _read_jsonl_lines(jsonl_path)
            steps = [m for m in messages if m.get('type') == 'step']
            meta: dict = {}
            for m in messages:
                if m.get('type') == 'session_start':
                    meta = m
                    break

            assert len(steps) == 3

            # Generate a second report from JSONL (simulates CLI)
            cli_report = Path(tmpdir) / 'cli_report.html'
            result = generate_report(steps, meta, cli_report)
            assert Path(result).exists()
            assert result == str(cli_report)

            html = Path(result).read_text(encoding='utf-8')
            assert 'echarts' in html
            assert 'heatmap' in html
            for i in range(3):
                assert f'word{i}' in html


class TestLogitsProcessorPipeline:
    """Create tracker, get logits_processor, simulate generate steps."""

    def test_logits_processor_records_all_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            tracker = create_tracker(model, transport_mode='file', output_dir=tmpdir)
            processor = tracker.logits_processor()

            num_steps = 5
            for i in range(num_steps):
                logits = torch.randn(1, VOCAB)
                input_ids = torch.randint(0, VOCAB, (1, 3 + i))
                result = processor(input_ids, logits)
                # logits_processor must pass through unchanged
                assert torch.equal(result, logits)

            assert tracker._step_count == num_steps

            report_path = tracker.finish()
            assert Path(report_path).exists()

            html = Path(report_path).read_text(encoding='utf-8')
            assert 'echarts' in html
            assert 'heatmap' in html

            # Verify JSONL has exactly num_steps step messages
            jsonl_files = list(Path(tmpdir).glob('univis_data_*.jsonl'))
            assert len(jsonl_files) == 1
            messages = _read_jsonl_lines(jsonl_files[0])
            steps = [m for m in messages if m.get('type') == 'step']
            assert len(steps) == num_steps


class TestDataConsistency:
    """Verify JSONL step count matches on_step() calls and layer count."""

    def test_step_count_matches_on_step_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            tracker = create_tracker(model, transport_mode='file', output_dir=tmpdir)

            x = torch.randn(1, 3, DIM)
            num_steps = 6
            with torch.no_grad():
                for i in range(num_steps):
                    out = model(x)
                    tracker.on_step(token_index=i, generated_token=f's{i}', logits=out[:, -1, :])

            tracker.finish()

            jsonl_files = list(Path(tmpdir).glob('univis_data_*.jsonl'))
            messages = _read_jsonl_lines(jsonl_files[0])
            steps = [m for m in messages if m.get('type') == 'step']
            assert len(steps) == num_steps

    def test_layer_count_matches_model_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            tracker = create_tracker(model, transport_mode='file', output_dir=tmpdir)

            x = torch.randn(1, 3, DIM)
            with torch.no_grad():
                out = model(x)
                tracker.on_step(token_index=0, generated_token='a', logits=out[:, -1, :])

            tracker.finish()

            jsonl_files = list(Path(tmpdir).glob('univis_data_*.jsonl'))
            messages = _read_jsonl_lines(jsonl_files[0])
            session_start = next(m for m in messages if m.get('type') == 'session_start')
            assert session_start['num_layers'] == N_LAYERS

            steps = [m for m in messages if m.get('type') == 'step']
            assert len(steps) == 1
            assert len(steps[0]['layers']) == N_LAYERS


class TestReportContent:
    """Verify HTML contains all expected sections and data."""

    def test_report_contains_all_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            tracker = univis.attach(model, transport='file', output_dir=tmpdir)

            x = torch.randn(1, 4, DIM)
            tokens = ['hello', 'world', 'foo', 'bar']
            with torch.no_grad():
                for i, tok in enumerate(tokens):
                    out = model(x)
                    tracker.on_step(token_index=i, generated_token=tok, logits=out[:, -1, :])

            report_path = tracker.finish()
            html = Path(report_path).read_text(encoding='utf-8')

            # Model name appears in stats section
            assert 'fake-gpt2' in html

            # Token count
            assert '>4<' in html

            # Heatmap data embedded as JS variable
            assert 'heatmapData' in html

            # Entropy data
            assert 'entropyData' in html

            # Layer summary (treemap + layerSummary data)
            assert 'treemap' in html
            assert 'layerSummary' in html
            assert 'Avg RelDelta' in html

            # Token list
            assert 'token-display' in html
            for tok in tokens:
                assert tok in html
