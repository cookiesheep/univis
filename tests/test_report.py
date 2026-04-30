"""Tests for HTML report generation."""

import tempfile
from pathlib import Path

from univis.report import (
    generate_report,
    _build_entropy_data,
    _build_heatmap_data,
    _build_layer_summary,
)


def _sample_steps():
    return [
        {
            'token_idx': 0,
            'generated_token': 'Hello',
            'global': {'prediction_entropy': 2.5},
            'layers': [
                {'idx': 0, 'name': 'transformer.h.0', 'relative_delta': 0.1, 'cosine_sim': 0.99, 'sparsity': 0.3},
                {'idx': 1, 'name': 'transformer.h.1', 'relative_delta': 0.2, 'cosine_sim': 0.95, 'sparsity': 0.4},
            ],
        },
        {
            'token_idx': 1,
            'generated_token': ' world',
            'global': {'prediction_entropy': 1.8},
            'layers': [
                {'idx': 0, 'name': 'transformer.h.0', 'relative_delta': 0.15, 'cosine_sim': 0.98, 'sparsity': 0.35},
                {'idx': 1, 'name': 'transformer.h.1', 'relative_delta': 0.25, 'cosine_sim': 0.93, 'sparsity': 0.45},
            ],
        },
    ]


class TestBuildHeatmapData:
    def test_extracts_points(self):
        points, tokens, layers, max_val = _build_heatmap_data(_sample_steps())
        assert len(points) == 4  # 2 tokens x 2 layers
        assert points[0] == [0, 0, 0.1]
        assert max_val == 0.25

    def test_empty_steps(self):
        points, tokens, layers, max_val = _build_heatmap_data([])
        assert points == []
        assert max_val == 0.0


class TestBuildEntropyData:
    def test_extracts_entropy(self):
        data = _build_entropy_data(_sample_steps())
        assert len(data) == 2
        assert data[0] == [0, 2.5]
        assert data[1] == [1, 1.8]


class TestBuildLayerSummary:
    def test_averages_across_steps(self):
        summary = _build_layer_summary(_sample_steps())
        assert len(summary) == 2
        assert abs(summary[0]['avg_delta'] - 0.125) < 1e-6
        assert abs(summary[1]['avg_cosim'] - 0.94) < 1e-6

    def test_empty(self):
        assert _build_layer_summary([]) == []


class TestGenerateReport:
    def test_creates_html_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'report.html'
            result = generate_report(
                _sample_steps(), {'session_id': 'abc123', 'model_name': 'test'}, path,
            )
            assert Path(result).exists()
            html = Path(result).read_text(encoding='utf-8')
            assert 'UniVis' in html
            assert 'echarts' in html
            assert 'heatmap' in html
            assert 'test' in html

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'sub' / 'dir' / 'report.html'
            generate_report(_sample_steps(), {'session_id': 'test'}, path)
            assert path.exists()

    def test_embeds_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'report.html'
            generate_report(
                _sample_steps(), {'session_id': 'xyz', 'model_name': 'GPT'}, path,
            )
            html = path.read_text(encoding='utf-8')
            assert 'Hello' in html
            assert '0.1' in html

    def test_escapes_script_tag_in_tokens(self):
        steps = [{
            'token_idx': 0, 'generated_token': '</script><script>alert(1)</script>',
            'global': {'prediction_entropy': 1.0}, 'layers': [
                {'idx': 0, 'name': 'h.0', 'relative_delta': 0.1, 'cosine_sim': 0.9, 'sparsity': 0.1},
            ],
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'report.html'
            generate_report(steps, {'session_id': 'xss', 'model_name': 'test'}, path)
            html = path.read_text(encoding='utf-8')
            assert '</script>' not in html.split('</script>')[-1]
            assert '<\\/script' in html
