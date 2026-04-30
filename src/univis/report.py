"""Generate self-contained HTML report with embedded ECharts heatmap."""

from __future__ import annotations

import html as html_mod
import json
from pathlib import Path
from typing import Any

_ECHARTS_BUNDLED = Path(__file__).parent / 'data' / 'echarts.min.js'
ECHARTS_CDN = 'https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js'


def _get_echarts_source(offline: bool = False) -> str:
    """Return inline JS content (offline) or CDN script tag (online)."""
    if offline and _ECHARTS_BUNDLED.is_file():
        return f'<script>\n{_ECHARTS_BUNDLED.read_text(encoding="utf-8")}\n</script>'
    return f'<script src="{ECHARTS_CDN}"></script>'


def _build_heatmap_data(steps: list[dict]) -> tuple[list[list], list[str], list[str], float]:
    """Extract heatmap data from step messages."""
    points: list[list[float]] = []
    token_labels: list[str] = []
    layer_names: list[str] = []
    max_val = 0.0

    for step in steps:
        token_labels.append(str(step.get('token_idx', len(token_labels))))
        for layer in step.get('layers', []):
            val = layer.get('relative_delta', 0)
            points.append([step.get('token_idx', len(token_labels) - 1), layer.get('idx', 0), val])
            if val > max_val:
                max_val = val

    # Build layer names from first step
    if steps:
        first_layers = sorted(steps[0].get('layers', []), key=lambda l: l.get('idx', 0))
        layer_names = [l.get('name', f'Layer {i}') for i, l in enumerate(first_layers)]

    return points, token_labels, layer_names, max_val


def _build_entropy_data(steps: list[dict]) -> list[list]:
    """Extract entropy curve data: [[token_idx, entropy], ...]."""
    result = []
    for step in steps:
        entropy = step.get('global', {}).get('prediction_entropy', -1)
        if entropy >= 0:
            result.append([step.get('token_idx', len(result)), entropy])
    return result


def _build_layer_summary(steps: list[dict]) -> list[dict]:
    """Per-layer average metrics across all steps."""
    if not steps:
        return []
    layer_data: dict[int, dict[str, list[float]]] = {}
    for step in steps:
        for layer in step.get('layers', []):
            idx = layer.get('idx', 0)
            layer_data.setdefault(idx, {'deltas': [], 'cosims': [], 'sparsities': []})
            if 'relative_delta' in layer:
                layer_data[idx]['deltas'].append(layer['relative_delta'])
            if 'cosine_sim' in layer:
                layer_data[idx]['cosims'].append(layer['cosine_sim'])
            if 'sparsity' in layer:
                layer_data[idx]['sparsities'].append(layer['sparsity'])

    summary = []
    for idx in sorted(layer_data):
        d = layer_data[idx]
        summary.append({
            'idx': idx,
            'avg_delta': sum(d['deltas']) / max(len(d['deltas']), 1),
            'avg_cosim': sum(d['cosims']) / max(len(d['cosims']), 1),
            'avg_sparsity': sum(d['sparsities']) / max(len(d['sparsities']), 1),
        })
    return summary


def _build_token_list(steps: list[dict]) -> list[str]:
    return [s.get('generated_token', '') for s in steps]


def generate_report(
    steps: list[dict],
    meta: dict[str, Any],
    output_path: str | Path,
    offline: bool = False,
) -> str:
    """Generate a self-contained HTML report with ECharts visualizations.

    Args:
        steps: List of step message dicts from Tracker._all_steps.
        meta: Dict with session metadata (session_id, model_name, etc).
        output_path: Where to write the HTML file.
        offline: If True, embed ECharts JS inline for offline use.

    Returns:
        Path to the generated report.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    echarts_tag = _get_echarts_source(offline)

    session_id = html_mod.escape(meta.get('session_id', 'unknown')[:8])
    model_name = html_mod.escape(meta.get('model_name', 'unknown'))
    num_layers = meta.get('num_layers', 0)

    heatmap_points, token_labels, layer_labels, max_delta = _build_heatmap_data(steps)
    entropy_data = _build_entropy_data(steps)
    layer_summary = _build_layer_summary(steps)
    token_list = _build_token_list(steps)

    all_deltas = [p[2] for p in heatmap_points]
    avg_delta = sum(all_deltas) / max(len(all_deltas), 1)

    # Build JSON for embedding
    def _safe_json(obj):
        return json.dumps(obj).replace('</script', '<\\/script')

    heatmap_json = _safe_json(heatmap_points)
    token_labels_json = _safe_json(token_labels)
    layer_labels_json = _safe_json(layer_labels)
    entropy_json = _safe_json(entropy_data)
    token_list_json = _safe_json(token_list)
    layer_summary_json = _safe_json(layer_summary)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>UniVis Report — {session_id}</title>
{echarts_tag}
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 1em auto; padding: 0 1em;
         background: #fafafa; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 2px solid #64ffda; padding-bottom: 8px; }}
  h2 {{ color: #0f3460; margin-top: 2em; }}
  .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 1em 0; }}
  .stat {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px 16px; min-width: 120px; }}
  .stat-label {{ color: #888; font-size: 11px; text-transform: uppercase; }}
  .stat-value {{ color: #1a1a2e; font-size: 20px; font-weight: 600; }}
  .chart {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 16px; margin: 1em 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #e0e0e0; padding: 8px 12px; text-align: left; }}
  th {{ background: #f5f5f5; }}
  .tokens {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px;
             font-family: monospace; line-height: 2; }}
  .token {{ display: inline-block; background: #e3f2fd; color: #1565c0; padding: 2px 6px;
            border-radius: 3px; margin: 2px; font-size: 13px; }}
</style>
</head>
<body>

<h1>UniVis Analysis Report</h1>
<div class="stats">
  <div class="stat"><div class="stat-label">Model</div><div class="stat-value">{model_name}</div></div>
  <div class="stat"><div class="stat-label">Layers</div><div class="stat-value">{num_layers}</div></div>
  <div class="stat"><div class="stat-label">Tokens</div><div class="stat-value">{len(steps)}</div></div>
  <div class="stat"><div class="stat-label">Avg RelDelta</div><div class="stat-value">{avg_delta:.4f}</div></div>
  <div class="stat"><div class="stat-label">Session</div><div class="stat-value">{session_id}</div></div>
</div>

<h2>Layer Redundancy Heatmap</h2>
<p style="color:#666; font-size:13px;">White = redundant (low delta), Red = active (high delta). Hover for details.</p>
<div id="heatmap" class="chart" style="height:450px;"></div>

<h2>Prediction Entropy</h2>
<p style="color:#666; font-size:13px;">Lower = model is more confident about the next token.</p>
<div id="entropy" class="chart" style="height:250px;"></div>

<h2>Per-Layer Summary</h2>
<table id="summary-table">
<tr><th>Layer</th><th>Avg RelDelta</th><th>Avg CosSim</th><th>Avg Sparsity</th></tr>
</table>

<h2>Generated Tokens</h2>
<div class="tokens" id="token-display"></div>

<script>
(function() {{
  var heatmapData = {heatmap_json};
  var tokenLabels = {token_labels_json};
  var layerLabels = {layer_labels_json};
  var entropyData = {entropy_json};
  var layerSummary = {layer_summary_json};
  var tokenList = {token_list_json};

  // Heatmap
  var hm = echarts.init(document.getElementById('heatmap'));
  hm.setOption({{
    tooltip: {{ position: 'top' }},
    grid: {{ height: '72%', top: '8%', left: '14%' }},
    xAxis: {{ type: 'category', data: tokenLabels, name: 'Token',
              splitArea: {{ show: true }}, axisLabel: {{ color: '#666' }} }},
    yAxis: {{ type: 'category', data: layerLabels, name: 'Layer',
              axisLabel: {{ color: '#666', fontSize: 11 }} }},
    visualMap: {{
      min: 0, max: {max(max_delta * 1.1, 0.001):.4f}, calculable: true,
      orient: 'horizontal', left: 'center', bottom: '2%',
      inRange: {{ color: ['#e8e8e8','#ffe0b2','#ffb74d','#ff7043','#e53935','#b71c1c'] }},
      textStyle: {{ color: '#666' }}
    }},
    series: [{{ type: 'heatmap', data: heatmapData, progressive: 500, animation: false,
                itemStyle: {{ borderColor: '#fff', borderWidth: 1 }} }}]
  }});

  // Entropy curve
  var ec = echarts.init(document.getElementById('entropy'));
  ec.setOption({{
    tooltip: {{ trigger: 'axis' }},
    grid: {{ height: '65%', top: '12%', left: '10%' }},
    xAxis: {{ type: 'category', name: 'Token', axisLabel: {{ color: '#666' }} }},
    yAxis: {{ type: 'value', name: 'Entropy', axisLabel: {{ color: '#666' }} }},
    series: [{{ type: 'line', data: entropyData.map(function(d){{ return [d[0]+'', d[1]]; }}),
                smooth: true, symbol: 'none', lineStyle: {{ color: '#1565c0', width: 2 }},
                areaStyle: {{ color: 'rgba(21,101,192,0.1)' }} }}]
  }});

  // Summary table
  var tbody = document.querySelector('#summary-table');
  layerSummary.forEach(function(l) {{
    var tr = document.createElement('tr');
    var td1 = document.createElement('td'); td1.textContent = layerLabels[l.idx] || 'Layer ' + l.idx;
    var td2 = document.createElement('td'); td2.textContent = l.avg_delta.toFixed(4);
    var td3 = document.createElement('td'); td3.textContent = l.avg_cosim.toFixed(4);
    var td4 = document.createElement('td'); td4.textContent = l.avg_sparsity.toFixed(4);
    tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3); tr.appendChild(td4);
    tbody.appendChild(tr);
  }});

  // Tokens
  var tokenDiv = document.getElementById('token-display');
  tokenList.forEach(function(t) {{
    if (!t) return;
    var span = document.createElement('span');
    span.className = 'token';
    span.textContent = t;
    tokenDiv.appendChild(span);
  }});

  window.addEventListener('resize', function() {{ hm.resize(); ec.resize(); }});
}})();
</script>
</body>
</html>"""

    out.write_text(html, encoding='utf-8')
    return str(out)
