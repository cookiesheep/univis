"""Generate self-contained HTML report with embedded ECharts visualizations."""

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
    """Per-layer average metrics across all steps, with ranking and trend."""
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
        deltas = d['deltas']
        avg_d = sum(deltas) / max(len(deltas), 1)
        mid = len(deltas) // 2
        first_half = sum(deltas[:max(mid, 1)]) / max(mid, 1)
        second_half = sum(deltas[mid:]) / max(len(deltas) - mid, 1)
        trend = 'stable'
        if first_half > 0 and abs(second_half - first_half) / first_half > 0.15:
            trend = 'declining' if second_half < first_half else 'rising'
        summary.append({
            'idx': idx,
            'avg_delta': avg_d,
            'avg_cosim': sum(d['cosims']) / max(len(d['cosims']), 1),
            'avg_sparsity': sum(d['sparsities']) / max(len(d['sparsities']), 1),
            'min_delta': min(deltas) if deltas else 0,
            'max_delta': max(deltas) if deltas else 0,
            'trend': trend,
        })

    ranked = sorted(summary, key=lambda x: x['avg_delta'], reverse=True)
    for rank, entry in enumerate(ranked, 1):
        entry['rank'] = rank
        entry['label'] = 'most_active' if rank == 1 else (
            'most_redundant' if rank == len(ranked) else '')

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

    # Compute per-layer delta per step for sparklines and theme river
    layer_deltas_per_step: dict[int, list[float]] = {}
    for step in steps:
        for layer in step.get('layers', []):
            idx = layer.get('idx', 0)
            layer_deltas_per_step.setdefault(idx, []).append(layer.get('relative_delta', 0))

    def _safe_json(obj):
        return json.dumps(obj).replace('</script', '<\\/script')

    heatmap_json = _safe_json(heatmap_points)
    token_labels_json = _safe_json(token_labels)
    layer_labels_json = _safe_json(layer_labels)
    entropy_json = _safe_json(entropy_data)
    token_list_json = _safe_json(token_list)
    layer_summary_json = _safe_json(layer_summary)
    layer_deltas_json = _safe_json(layer_deltas_per_step)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UniVis 层冗余分析报告 — {session_id}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Playfair+Display:wght@400;600;700&family=Noto+Sans+SC:wght@400;500&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
{echarts_tag}
<style>
  /* ═══ Scientific Manuscript — 学术手稿风格 ═══ */
  :root {{
    --bg-deep: #faf8f5;
    --bg-surface: #ffffff;
    --bg-elevated: #f5f2ed;
    --border: #e0dbd5;
    --text-primary: #1a1a2e;
    --text-secondary: #6b6b7b;
    --accent: #c45c26;
    --accent-green: #2d6a4f;
    --accent-muted: #d4a373;
  }}
  *,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    font-family:'Noto Sans SC','Georgia',serif;
    background:var(--bg-deep); color:var(--text-primary);
    min-height:100vh;
  }}
  body::before {{
    content:''; position:fixed; inset:0;
    background-image:
      radial-gradient(circle at 20% 50%, rgba(212,163,115,0.04) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(196,92,38,0.03) 0%, transparent 40%);
    pointer-events:none; z-index:0;
  }}
  .container {{ max-width:1100px; margin:0 auto; padding:48px 32px; position:relative; z-index:1; }}

  /* ─── Hero ─── */
  .hero {{
    margin-bottom:40px; padding-bottom:32px;
    border-bottom:2px solid var(--border);
  }}
  .hero-title {{
    font-family:'Noto Serif SC','Playfair Display',serif;
    font-size:36px; font-weight:700; color:var(--text-primary);
    line-height:1.3; letter-spacing:1px;
  }}
  .hero-subtitle {{
    font-family:'Playfair Display','Noto Serif SC',serif;
    font-size:16px; color:var(--accent);
    margin-top:4px; font-style:italic; letter-spacing:0.5px;
  }}
  .hero-intro {{
    font-size:14px; line-height:1.8; color:var(--text-secondary);
    margin-top:16px; max-width:640px; text-align:justify;
  }}
  .hero-intro code {{
    font-family:'JetBrains Mono',monospace;
    background:var(--bg-elevated); padding:1px 5px;
    border-radius:3px; font-size:12px; color:var(--accent);
  }}
  .hero-meta {{
    margin-top:20px; display:flex; align-items:center;
    gap:12px; flex-wrap:wrap;
  }}
  .hero-badge {{
    display:inline-block; padding:5px 14px;
    border:1px solid var(--border); border-radius:20px;
    font-family:'JetBrains Mono',monospace;
    font-size:12px; color:var(--text-secondary);
    background:var(--bg-surface);
  }}
  .hero-model {{
    font-family:'JetBrains Mono',monospace;
    font-size:14px; font-weight:600; color:var(--accent);
  }}

  /* Layer strip — 小巧思: mini layer overview in hero */
  .layer-strip {{ display:flex; gap:3px; margin-top:16px; align-items:flex-end; }}
  .strip-block {{
    flex:1; max-width:40px; height:6px; border-radius:2px;
    transition:height 0.3s,transform 0.2s; cursor:pointer;
  }}
  .strip-block:hover {{ transform:scaleY(3); filter:brightness(1.15); }}
  .strip-label {{
    font-size:10px; color:var(--text-secondary);
    font-family:'JetBrains Mono',monospace; margin-top:4px;
  }}

  /* ─── Stats (Abstract style) ─── */
  .stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:40px; }}
  .stat {{
    text-align:center; padding:20px 12px;
    background:var(--bg-surface); border:1px solid var(--border);
    border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,0.04);
  }}
  .stat-number {{
    font-family:'Playfair Display','Noto Serif SC',serif;
    font-size:32px; font-weight:700; color:var(--text-primary); line-height:1.2;
  }}
  .stat-label {{
    font-size:12px; color:var(--text-secondary);
    margin-top:4px; letter-spacing:0.5px;
  }}

  /* ─── Section Headers ─── */
  .sep {{
    display:flex; align-items:center; gap:16px; margin:48px 0 16px;
  }}
  .sep::before,.sep::after {{
    content:''; flex:1; height:1px;
    background:linear-gradient(to right,transparent,var(--border),transparent);
  }}
  .sep-icon {{ color:var(--accent-muted); font-size:12px; letter-spacing:4px; }}
  .section-title {{
    font-family:'Noto Serif SC','Playfair Display',serif;
    font-size:20px; font-weight:600; color:var(--text-primary); margin-bottom:6px;
  }}
  .section-tag {{
    font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:400;
    color:var(--accent-muted); margin-left:8px; padding:2px 8px;
    border:1px solid var(--border); border-radius:12px; vertical-align:middle;
  }}
  .section-hint {{
    font-size:13px; color:var(--text-secondary); margin-bottom:14px; line-height:1.6;
  }}

  /* ─── Chart Containers ─── */
  .cb {{
    background:var(--bg-surface); border:1px solid var(--border);
    border-radius:8px; padding:24px; margin-bottom:24px;
    box-shadow:0 1px 4px rgba(0,0,0,0.04);
  }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }}

  /* ─── MRI ─── */
  .mri-wrap {{ position:relative; overflow:hidden; border-radius:8px; }}
  .mri-wrap::after {{
    content:''; position:absolute; top:50%; left:50%;
    width:45%; height:1px; pointer-events:none;
    background:linear-gradient(to right,rgba(196,92,38,0.5),transparent);
    transform-origin:left center; animation:scan 5s linear infinite;
  }}
  @keyframes scan {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}

  /* ─── Pulse Grid ─── */
  .pulse-grid {{
    display:grid; grid-template-columns:repeat(auto-fill,minmax(130px,1fr));
    gap:12px; margin-bottom:24px;
  }}
  .pulse-card {{
    background:var(--bg-surface); border:1px solid var(--border);
    border-radius:6px; padding:10px;
    transition:border-color 0.2s,box-shadow 0.2s;
  }}
  .pulse-card:hover {{
    border-color:var(--accent);
    box-shadow:0 2px 12px rgba(196,92,38,0.12);
  }}
  .pulse-card.active {{ border-color:var(--accent); background:rgba(196,92,38,0.03); }}
  .pulse-card.dormant {{ opacity:0.6; }}
  .pulse-title {{
    font-family:'JetBrains Mono',monospace;
    font-size:11px; color:var(--text-secondary);
  }}
  .pulse-delta {{
    font-family:'JetBrains Mono',monospace;
    font-size:12px; font-weight:600; color:var(--text-primary); margin-top:4px;
  }}
  .pulse-trend {{ font-size:11px; margin-left:2px; }}
  .pulse-bar {{
    height:3px; border-radius:2px; margin-top:6px;
    background:var(--bg-elevated); overflow:hidden;
  }}
  .pulse-bar-fill {{ height:100%; border-radius:2px; transition:width 0.5s; }}

  /* ─── Token Spectrum ─── */
  .token-spectrum {{
    background:var(--bg-surface); border:1px solid var(--border);
    border-radius:8px; padding:20px; margin-bottom:24px;
    display:flex; flex-wrap:wrap; gap:4px; align-items:flex-end; min-height:80px;
  }}
  .token-block {{
    display:inline-flex; align-items:flex-end; justify-content:center;
    font-family:'JetBrains Mono',monospace; font-size:11px;
    border-radius:3px; padding:2px 6px;
    transition:transform 0.15s,box-shadow 0.15s;
    color:var(--text-primary); min-width:24px; text-align:center;
  }}
  .token-block:hover {{ transform:translateY(-3px); box-shadow:0 4px 12px rgba(196,92,38,0.2); }}

  /* ─── Diagnosis ─── */
  .diag-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }}
  .diag-card {{
    background:var(--bg-surface); border:1px solid var(--border);
    border-radius:8px; padding:24px;
  }}
  .diag-card h3 {{
    font-family:'Noto Serif SC',serif; font-size:16px; font-weight:600;
    margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid var(--border);
  }}
  .diag-metric {{
    display:flex; justify-content:space-between; padding:5px 0; font-size:14px;
  }}
  .diag-metric .label {{ color:var(--text-secondary); }}
  .diag-metric .val {{ font-family:'JetBrains Mono',monospace; font-weight:600; }}

  /* ─── Footer ─── */
  .footer {{ text-align:center; padding:32px 0; margin-top:48px; border-top:2px solid var(--border); }}
  .footer-name {{
    font-family:'Noto Serif SC',serif; font-size:14px;
    font-weight:600; color:var(--text-primary);
  }}
  .footer-detail {{
    font-size:12px; color:var(--text-secondary);
    font-family:'JetBrains Mono',monospace; margin-top:6px;
  }}

  /* ─── Animations ─── */
  @keyframes fadeUp {{ from{{opacity:0;transform:translateY(12px)}} to{{opacity:1;transform:translateY(0)}} }}
  .fade-in {{ animation:fadeUp 0.5s ease-out backwards; }}

  /* ─── Responsive ─── */
  @media (max-width:768px) {{
    .stats {{ grid-template-columns:1fr 1fr; }}
    .grid2,.diag-grid {{ grid-template-columns:1fr; }}
    .hero-title {{ font-size:28px; }}
    .pulse-grid {{ grid-template-columns:repeat(auto-fill,minmax(100px,1fr)); }}
    .container {{ padding:24px 16px; }}
  }}
</style>
</head>
<body>
<div class="container">

  <!-- 1. HERO — 项目介绍 -->
  <div class="hero">
    <div class="hero-title">UniVis 层冗余分析报告</div>
    <div class="hero-subtitle">Transformer Inference Redundancy Diagnostics</div>
    <p class="hero-intro">
      UniVis 是面向生成式大模型的时空冗余可视化与智能分析平台。通过
      <code>register_forward_hook</code> 在推理过程中逐层捕获关键指标
      （Relative Delta、Cosine Similarity、Activation Sparsity），
      帮助研究者快速定位冗余层，为模型压缩与高效推理提供量化依据。
    </p>
    <div class="hero-meta">
      <span class="hero-model">{model_name}</span>
      <span class="hero-badge">{num_layers} 层</span>
      <span class="hero-badge">{len(steps)} tokens</span>
      <span class="hero-badge">session {session_id}</span>
    </div>
    <div class="layer-strip" id="layer-strip"></div>
  </div>

  <!-- 2. STATS — 摘要统计 (preserve 'Avg RelDelta' and '>4<') -->
  <div class="stats">
    <div class="stat">
      <div class="stat-number">{len(steps)}</div>
      <div class="stat-label">Token 数</div>
    </div>
    <div class="stat">
      <div class="stat-number" style="color:{'#c45c26' if avg_delta < 0.3 else '#c0392b'}">{avg_delta:.4f}</div>
      <div class="stat-label">Avg RelDelta</div>
    </div>
    <div class="stat">
      <div class="stat-number">{num_layers}</div>
      <div class="stat-label">模型层数</div>
    </div>
    <div class="stat">
      <div class="stat-number" style="font-size:16px;color:var(--text-secondary)">{session_id}</div>
      <div class="stat-label">会话编号</div>
    </div>
  </div>

  <!-- 3. MODEL MRI -->
  <div class="sep"><span class="sep-icon">&#9670; &#9671; &#9670;</span></div>
  <div class="section-title">模型 MRI <span class="section-tag">同心环扫描</span></div>
  <div class="section-hint">同心环扫描 — 内环 = 浅层，外环 = 深层。赤陶色 = 活跃，灰绿色 = 冗余。悬停查看详情。</div>
  <div class="cb mri-wrap fade-in" style="animation-delay:0.1s">
    <div id="mri-chart" style="height:400px;"></div>
  </div>

  <!-- 4. LAYER PULSE -->
  <div class="sep"><span class="sep-icon">&#9670; &#9671; &#9670;</span></div>
  <div class="section-title">层级脉冲 <span class="section-tag">ICU 波形网格</span></div>
  <div class="section-hint">每张卡片 = 一层的活动波形。平线 = 冗余，尖峰 = 活跃。</div>
  <div class="pulse-grid" id="pulse-grid"></div>

  <!-- 5. DATA RIVER -->
  <div class="sep"><span class="sep-icon">&#9670; &#9671; &#9670;</span></div>
  <div class="section-title">数据河流 <span class="section-tag">ThemeRiver</span></div>
  <div class="section-hint">层级活跃度随 Token 推进的流动。宽 = 活跃，窄 = 冗余。</div>
  <div class="cb fade-in" style="animation-delay:0.2s">
    <div id="river-chart" style="height:320px;"></div>
  </div>

  <!-- 6. HEATMAP — preserve id="heatmap" -->
  <div class="sep"><span class="sep-icon">&#9670; &#9671; &#9670;</span></div>
  <div class="section-title">冗余热力图</div>
  <div class="section-hint">Plasma 配色 — 暗色 = 冗余，亮色 = 活跃。滚轮缩放。</div>
  <div class="cb fade-in" style="animation-delay:0.25s">
    <div id="heatmap" style="height:420px;"></div>
  </div>

  <!-- 7. ENTROPY + TREEMAP — preserve ids -->
  <div class="grid2">
    <div>
      <div class="sep"><span class="sep-icon">&#9671;</span></div>
      <div class="section-title">熵波形</div>
      <div class="section-hint">模型置信度随时间变化。低 = 更确定。</div>
      <div class="cb fade-in" style="animation-delay:0.3s">
        <div id="entropy" style="height:280px;"></div>
      </div>
    </div>
    <div>
      <div class="sep"><span class="sep-icon">&#9671;</span></div>
      <div class="section-title">活跃度树图</div>
      <div class="section-hint">面积 = 活跃度贡献。点击可探索。</div>
      <div class="cb fade-in" style="animation-delay:0.35s">
        <div id="treemap" style="height:280px;"></div>
      </div>
    </div>
  </div>

  <!-- 8. TOKEN SPECTRUM — preserve id="token-display" -->
  <div class="sep"><span class="sep-icon">&#9670; &#9671; &#9670;</span></div>
  <div class="section-title">Token 频谱</div>
  <div class="section-hint">颜色 = 预测熵（灰绿 = 确定，赤陶 = 不确定）。高度 = 活跃度。</div>
  <div class="token-spectrum" id="token-display"></div>

  <!-- 9. DIAGNOSIS -->
  <div class="sep"><span class="sep-icon">&#9670; &#9671; &#9670;</span></div>
  <div class="section-title">诊断结论</div>
  <div class="section-hint">最活跃层 vs 最冗余层对比分析。</div>
  <div class="diag-grid" id="diag-grid"></div>

  <!-- 10. FOOTER -->
  <div class="footer">
    <div class="footer-name">UniVis — 面向生成式大模型的时空冗余可视化与智能分析平台</div>
    <div class="footer-detail">中山大学 · v0.5 · Powered by ECharts · session {session_id}</div>
  </div>
</div>

<script>
(function(){{
  var heatmapData = {heatmap_json};
  var tokenLabels = {token_labels_json};
  var layerLabels = {layer_labels_json};
  var entropyData = {entropy_json};
  var layerSummary = {layer_summary_json};
  var tokenList = {token_list_json};
  var layerDeltas = {layer_deltas_json};
  var totalLayers = layerSummary.length;
  var maxDelta = {max(max_delta * 1.1, 0.001):.4f};
  var allCharts = [];

  // Color helpers — Scientific Manuscript palette
  // Redundant (low) → sage green, Active (high) → terracotta
  var minDelta = layerSummary.length ? layerSummary.reduce(function(m,l){{ return Math.min(m,l.avg_delta); }}, Infinity) : 0;
  var maxAvgDelta = layerSummary.length ? layerSummary.reduce(function(m,l){{ return Math.max(m,l.avg_delta); }}, 0) : maxDelta;
  var rangeDelta = Math.max(maxAvgDelta - minDelta, 0.0001);

  function divergingColor(v) {{
    var t = Math.max(0, Math.min(1, (v - minDelta) / rangeDelta));
    // t=0 → sage (143,170,126)  t=0.5 → gold (212,163,115)  t=1 → terracotta (196,92,38)
    if (t < 0.5) {{
      var s = t / 0.5;
      return 'rgba(' + Math.round(143+69*s) + ',' + Math.round(170-7*s) + ',' + Math.round(126-11*s) + ',1)';
    }} else {{
      var s = (t - 0.5) / 0.5;
      return 'rgba(' + Math.round(212-16*s) + ',' + Math.round(163-71*s) + ',' + Math.round(115-77*s) + ',1)';
    }}
  }}

  // ===== Hero: Layer activity strip =====
  var stripEl = document.getElementById('layer-strip');
  if (stripEl) {{
    layerSummary.forEach(function(l) {{
      var block = document.createElement('div');
      block.className = 'strip-block';
      block.style.background = divergingColor(l.avg_delta);
      var h = Math.round(4 + (l.avg_delta / maxDelta) * 14);
      block.style.height = h + 'px';
      block.title = 'L' + l.idx + ': Delta=' + l.avg_delta.toFixed(4);
      stripEl.appendChild(block);
    }});
  }}

  // ===== 3. MODEL MRI — concentric rings =====
  var mriEl = document.getElementById('mri-chart');
  var mri = echarts.init(mriEl);
  var mriW = mriEl.clientWidth || 400;
  var mriH = mriEl.clientHeight || 380;
  var mriCx = mriW / 2, mriCy = mriH / 2;
  var mriMaxR = Math.min(mriCx, mriCy) - 15;
  var mriRW = mriMaxR / (totalLayers + 0.5);

  mri.setOption({{
    tooltip: {{
      backgroundColor:'rgba(255,252,248,0.98)', borderColor:'#e0dbd5', borderWidth:1,
      textStyle:{{ color:'#1a1a2e', fontFamily:"'Noto Sans SC',serif" }},
      formatter:function(p){{
        var l=layerSummary[p.dataIndex]; if(!l)return'';
        return '<b>'+(layerLabels[l.idx]||'Layer '+l.idx)+'</b><br/>'+
          '<span style="color:#6b6b7b">Avg Delta:</span> <span style="font-family:JetBrains Mono,monospace">'+l.avg_delta.toFixed(4)+'</span><br/>'+
          '<span style="color:#6b6b7b">CosSim:</span> <span style="font-family:JetBrains Mono,monospace">'+l.avg_cosim.toFixed(4)+'</span><br/>'+
          '<span style="color:#6b6b7b">Sparsity:</span> <span style="font-family:JetBrains Mono,monospace">'+l.avg_sparsity.toFixed(4)+'</span><br/>'+
          '<span style="color:#6b6b7b">Trend:</span> '+(l.trend==='declining'?'↓':l.trend==='rising'?'↑':'→')+' '+l.trend;
      }}
    }},
    graphic: [{{
      type:'text', left:'center', top:'middle',
      style: {{
        text: '{model_name}',
        fill: '#6b6b7b',
        fontSize: 13,
        fontFamily: "'JetBrains Mono', monospace",
        fontWeight: 600
      }}
    }}],
    series:[{{
      type:'custom', coordinateSystem:'none',
      renderItem:function(params,api){{
        var idx=params.dataIndex, val=api.value(1);
        var t=Math.max(0, Math.min(1, (val - minDelta) / rangeDelta));
        var innerR=Math.max(10, 10+idx*mriRW);
        var outerR=innerR+mriRW*0.82;
        var c = divergingColor(val);
        var isActive = t > 0.7;
        return {{
          type:'ring',
          shape:{{ cx:mriCx, cy:mriCy, r:outerR, r0:innerR }},
          style:{{
            fill: c,
            shadowBlur: isActive?12:0,
            shadowColor: isActive?'rgba(196,92,38,0.3)':'transparent'
          }},
          emphasis:{{ style:{{ shadowBlur:20, shadowColor:'rgba(196,92,38,0.4)' }} }}
        }};
      }},
      data:layerSummary.map(function(l){{ return [l.idx, l.avg_delta]; }}),
      animationDuration:1500, animationEasing:'cubicOut'
    }}]
  }});
  allCharts.push(mri);

  // ===== 4. LAYER PULSE — ICU sparkline grid =====
  var pulseGrid = document.getElementById('pulse-grid');
  layerSummary.forEach(function(l, i) {{
    var card = document.createElement('div');
    card.className = 'pulse-card' + (l.label==='most_active'?' active':'') + (l.label==='most_redundant'?' dormant':'');
    var title = document.createElement('div');
    title.className = 'pulse-title';
    title.textContent = 'L'+l.idx;
    card.appendChild(title);
    var chartEl = document.createElement('div');
    chartEl.style.height = '50px';
    card.appendChild(chartEl);
    var deltaEl = document.createElement('div');
    deltaEl.className = 'pulse-delta';
    deltaEl.innerHTML = l.avg_delta.toFixed(4) + ' <span class="pulse-trend">' +
      (l.trend==='declining'?'↓':l.trend==='rising'?'↑':'→') + '</span>';
    card.appendChild(deltaEl);
    var bar = document.createElement('div');
    bar.className = 'pulse-bar';
    var fill = document.createElement('div');
    fill.className = 'pulse-bar-fill';
    var pct = Math.min(l.avg_delta / maxDelta * 100, 100);
    fill.style.width = pct + '%';
    fill.style.background = pct > 60 ? '#c45c26' : pct > 30 ? '#d4a373' : '#2d6a4f';
    bar.appendChild(fill);
    card.appendChild(bar);
    pulseGrid.appendChild(card);

    var mini = echarts.init(chartEl);
    var deltas = layerDeltas[l.idx] || [];
    mini.setOption({{
      grid:{{ top:3, right:2, bottom:3, left:2 }},
      xAxis:{{ show:false, type:'category', data:deltas.map(function(_,j){{ return j; }}) }},
      yAxis:{{ show:false, type:'value' }},
      series:[{{
        type:'line', data:deltas, smooth:true, symbol:'none',
        lineStyle:{{
          width:1.5,
          color: l.avg_delta > maxDelta*0.3 ? '#c45c26' : '#2d6a4f'
        }},
        areaStyle:{{
          color: l.avg_delta > maxDelta*0.3 ? 'rgba(196,92,38,0.08)' : 'rgba(45,106,79,0.06)'
        }}
      }}]
    }});
    allCharts.push(mini);
  }});

  // ===== 5. DATA RIVER — themeRiver =====
  var riverData = [];
  Object.keys(layerDeltas).forEach(function(idx) {{
    var deltas = layerDeltas[idx];
    deltas.forEach(function(delta, stepIdx) {{
      riverData.push([stepIdx, delta, 'L'+idx]);
    }});
  }});
  var riverEl = document.getElementById('river-chart');
  var riverChart = echarts.init(riverEl);
  riverChart.setOption({{
    tooltip:{{
      backgroundColor:'rgba(255,252,248,0.98)', borderColor:'#e0dbd5', borderWidth:1,
      textStyle:{{ color:'#1a1a2e', fontFamily:"'Noto Sans SC',serif" }}
    }},
    singleAxis:{{
      type:'category', data:tokenLabels,
      axisLine:{{ lineStyle:{{ color:'#e0dbd5' }} }},
      axisLabel:{{ color:'#6b6b7b' }}
    }},
    series:[{{
      type:'themeRiver', data:riverData,
      label:{{ show:false }},
      emphasis:{{ focus:'series' }},
      itemStyle:{{
        color:function(params){{
          var d = layerSummary.find(function(l){{ return 'L'+l.idx===params.data[2]; }});
          return d ? divergingColor(d.avg_delta) : '#d4a373';
        }}
      }}
    }}]
  }});
  allCharts.push(riverChart);

  // ===== 6. HEATMAP =====
  var hm = echarts.init(document.getElementById('heatmap'));
  hm.setOption({{
    tooltip:{{
      position:'top',
      backgroundColor:'rgba(255,252,248,0.98)', borderColor:'#e0dbd5', borderWidth:1,
      textStyle:{{ color:'#1a1a2e', fontFamily:"'Noto Sans SC',serif", fontSize:12 }},
      formatter:function(p){{
        var d=p.data;
        return '<b>'+(layerLabels[d[1]]||'Layer '+d[1])+'</b> — Token '+d[0]+'<br/>'+
          '<span style="font-family:JetBrains Mono,monospace">RelDelta: '+d[2].toFixed(4)+'</span>';
      }}
    }},
    grid:{{ height:'62%', top:'10%', left:'14%', bottom:'14%' }},
    xAxis:{{ type:'category', data:tokenLabels, name:'Token',
      splitArea:{{ show:false }},
      axisLine:{{ lineStyle:{{ color:'#e0dbd5' }} }},
      axisLabel:{{ color:'#6b6b7b' }}, nameTextStyle:{{ color:'#6b6b7b' }}
    }},
    yAxis:{{ type:'category', data:layerLabels, name:'Layer',
      axisLine:{{ lineStyle:{{ color:'#e0dbd5' }} }},
      axisLabel:{{ color:'#6b6b7b', fontSize:11 }}, nameTextStyle:{{ color:'#6b6b7b' }}
    }},
    visualMap:{{
      min:0, max:maxDelta, calculable:true,
      orient:'horizontal', left:'center', bottom:'3%',
      inRange:{{ color:['#0d0887','#4903a0','#7d03a8','#b93289','#db5c68','#f48849','#febd2a','#f0f921'] }},
      textStyle:{{ color:'#6b6b7b' }}
    }},
    dataZoom:[
      {{ type:'inside', xAxisIndex:0, filterMode:'none' }},
      {{ type:'slider', xAxisIndex:0, bottom:'0%', height:20,
        borderColor:'transparent', backgroundColor:'#f5f2ed',
        fillerColor:'rgba(196,92,38,0.1)', handleStyle:{{ color:'#c45c26' }},
        textStyle:{{ color:'#6b6b7b' }} }}
    ],
    series:[{{
      type:'heatmap', data:heatmapData, progressive:500,
      animation:true, animationDuration:800,
      itemStyle:{{ borderColor:'#ffffff', borderWidth:1 }},
      emphasis:{{ itemStyle:{{ borderColor:'#c45c26', borderWidth:2 }} }}
    }}]
  }});
  allCharts.push(hm);

  // ===== 7a. ENTROPY WAVEFORM =====
  var ec = echarts.init(document.getElementById('entropy'));
  ec.setOption({{
    tooltip:{{
      trigger:'axis',
      backgroundColor:'rgba(255,252,248,0.98)', borderColor:'#e0dbd5', borderWidth:1,
      textStyle:{{ color:'#1a1a2e', fontFamily:"'JetBrains Mono',monospace", fontSize:12 }}
    }},
    grid:{{ height:'65%', top:'10%', left:'10%', right:'5%' }},
    xAxis:{{ type:'category', name:'Token',
      axisLine:{{ lineStyle:{{ color:'#e0dbd5' }} }}, axisLabel:{{ color:'#6b6b7b' }},
      nameTextStyle:{{ color:'#6b6b7b' }}
    }},
    yAxis:{{ type:'value', name:'Entropy',
      axisLine:{{ lineStyle:{{ color:'#e0dbd5' }} }}, axisLabel:{{ color:'#6b6b7b' }},
      nameTextStyle:{{ color:'#6b6b7b' }},
      splitLine:{{ lineStyle:{{ color:'#f0ece6', width:0.5, type:'dashed' }} }}
    }},
    series:[{{
      type:'line',
      data:entropyData.map(function(d){{ return [d[0]+'', d[1]]; }}),
      smooth:true, symbol:'none',
      lineStyle:{{ color:'#2d6a4f', width:2 }},
      areaStyle:{{
        color:{{
          type:'linear', x:0, y:0, x2:0, y2:1,
          colorStops:[
            {{ offset:0, color:'rgba(45,106,79,0.15)' }},
            {{ offset:1, color:'rgba(45,106,79,0.01)' }}
          ]
        }}
      }}
    }}]
  }});
  allCharts.push(ec);

  // ===== 7b. TREEMAP =====
  var tm = echarts.init(document.getElementById('treemap'));
  tm.setOption({{
    tooltip:{{
      backgroundColor:'rgba(255,252,248,0.98)', borderColor:'#e0dbd5', borderWidth:1,
      textStyle:{{ color:'#1a1a2e', fontFamily:"'Noto Sans SC',serif" }}
    }},
    series:[{{
      type:'treemap',
      data:layerSummary.map(function(l){{
        var short='L'+l.idx;
        return {{
          name:short, value:l.avg_delta,
          label: l.label==='most_active'?{{ formatter:short+' ★', color:'#c45c26' }} :
                 l.label==='most_redundant'?{{ formatter:short+' ○', color:'#2d6a4f' }} :
                 {{ formatter:short, color:'#6b6b7b' }},
          itemStyle:{{
            color:divergingColor(l.avg_delta),
            borderColor:'#ffffff', borderWidth:2, gapWidth:2
          }}
        }};
      }}),
      leafDepth:1, roam:false, breadcrumb:{{ show:false }},
      width:'100%', height:'100%', animationDuration:1000
    }}]
  }});
  allCharts.push(tm);

  // ===== 8. TOKEN SPECTRUM =====
  var tokenDiv = document.getElementById('token-display');
  var maxEnt = Math.max.apply(null, entropyData.map(function(d){{ return d[1]; }}).concat([1]));
  tokenList.forEach(function(t, i) {{
    if (!t) return;
    var span = document.createElement('span');
    span.className = 'token-block';
    span.textContent = t;
    var ent = entropyData[i] ? entropyData[i][1] / maxEnt : 0;
    var h = Math.round(24 + ent * 40);
    span.style.height = h + 'px';
    // Sage green → terracotta based on entropy
    var r = Math.round(143 + 53*ent);
    var g = Math.round(170 - 78*ent);
    var b = Math.round(126 - 88*ent);
    span.style.background = 'rgba('+r+','+g+','+b+',0.15)';
    span.style.color = 'rgb('+r+','+g+','+b+')';
    span.style.borderBottom = '2px solid rgba('+r+','+g+','+b+',0.4)';
    tokenDiv.appendChild(span);
  }});

  // ===== 9. DIAGNOSIS CARDS =====
  var diagGrid = document.getElementById('diag-grid');
  var mostActive = null, mostRedundant = null;
  layerSummary.forEach(function(l) {{
    if (l.label === 'most_active') mostActive = l;
    if (l.label === 'most_redundant') mostRedundant = l;
  }});

  function makeDiagCard(title, layer, borderColor, icon) {{
    var card = document.createElement('div');
    card.className = 'diag-card';
    card.style.borderLeft = '4px solid ' + borderColor;
    if (!layer) {{
      card.innerHTML = '<h3 style="color:'+borderColor+'">' + icon + ' ' + title + '</h3><div style="color:#6b6b7b">暂无数据</div>';
      return card;
    }}
    card.innerHTML =
      '<h3 style="color:'+borderColor+'">' + icon + ' ' + title + '</h3>' +
      '<div class="diag-metric"><span class="label">层级</span><span class="val">' + layer.idx + '</span></div>' +
      '<div class="diag-metric"><span class="label">Avg Delta</span><span class="val" style="color:'+borderColor+'">' + layer.avg_delta.toFixed(4) + '</span></div>' +
      '<div class="diag-metric"><span class="label">余弦相似度</span><span class="val">' + layer.avg_cosim.toFixed(4) + '</span></div>' +
      '<div class="diag-metric"><span class="label">稀疏度</span><span class="val">' + layer.avg_sparsity.toFixed(4) + '</span></div>' +
      '<div class="diag-metric"><span class="label">Min / Max</span><span class="val">' + layer.min_delta.toFixed(4) + ' / ' + layer.max_delta.toFixed(4) + '</span></div>' +
      '<div class="diag-metric"><span class="label">趋势</span><span class="val">' +
        (layer.trend==='declining'?'↓ 下降':layer.trend==='rising'?'↑ 上升':'→ 稳定') + '</span></div>';
    return card;
  }}

  diagGrid.appendChild(makeDiagCard('最活跃层', mostActive, '#c45c26', '✓'));
  diagGrid.appendChild(makeDiagCard('最冗余层', mostRedundant, '#2d6a4f', '✗'));

  // ===== RESIZE =====
  window.addEventListener('resize', function() {{
    mriW = mriEl.clientWidth || 400;
    mriH = mriEl.clientHeight || 380;
    mriCx = mriW / 2; mriCy = mriH / 2;
    mriMaxR = Math.min(mriCx, mriCy) - 15;
    mriRW = mriMaxR / (totalLayers + 0.5);
    allCharts.forEach(function(c) {{ c.resize(); }});
  }});
}})();
</script>
</body>
</html>"""

    out.write_text(html, encoding='utf-8')
    return str(out)


def _load_jsonl(path: str | Path) -> tuple[list[dict], dict]:
    """Load steps and metadata from a JSONL file."""
    messages: list[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    steps = [m for m in messages if m.get('type') == 'step']
    meta = next((m for m in messages if m.get('type') == 'session_start'), {})
    return steps, meta


def generate_comparison_report(
    jsonl_paths: list[str | Path],
    output_path: str | Path,
    offline: bool = False,
) -> str:
    """Generate HTML report comparing redundancy across multiple models.

    Args:
        jsonl_paths: List of JSONL file paths from different models.
        output_path: Where to write the HTML file.
        offline: If True, embed ECharts JS inline.

    Returns:
        Path to the generated report.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    echarts_tag = _get_echarts_source(offline)

    models: list[dict] = []
    for p in jsonl_paths:
        steps, meta = _load_jsonl(p)
        summary = _build_layer_summary(steps)
        name = meta.get('model_name', Path(p).stem)
        models.append({
            'name': name,
            'num_layers': meta.get('num_layers', 0),
            'num_steps': len(steps),
            'summary': summary,
            'avg_delta': sum(s['avg_delta'] for s in summary) / max(len(summary), 1),
            'most_active': max(summary, key=lambda s: s['avg_delta']) if summary else None,
            'most_redundant': min(summary, key=lambda s: s['avg_delta']) if summary else None,
        })

    def _safe_json(obj):
        return json.dumps(obj).replace('</script', '<\\/script')

    models_json = _safe_json(models)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UniVis 模型对比报告</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Playfair+Display:wght@400;600;700&family=Noto+Sans+SC:wght@400;500&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
{echarts_tag}
<style>
  :root {{
    --bg-deep:#faf8f5; --bg-surface:#ffffff; --bg-elevated:#f5f2ed;
    --border:#e0dbd5; --text-primary:#1a1a2e; --text-secondary:#6b6b7b;
    --accent:#c45c26; --accent-green:#2d6a4f; --accent-muted:#d4a373;
  }}
  *,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    font-family:'Noto Sans SC','Georgia',serif;
    background:var(--bg-deep); color:var(--text-primary); min-height:100vh;
  }}
  body::before {{
    content:''; position:fixed; inset:0;
    background-image:
      radial-gradient(circle at 20% 50%, rgba(212,163,115,0.04) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(196,92,38,0.03) 0%, transparent 40%);
    pointer-events:none; z-index:0;
  }}
  .container {{ max-width:1100px; margin:0 auto; padding:48px 32px; position:relative; z-index:1; }}
  .hero {{ margin-bottom:40px; padding-bottom:32px; border-bottom:2px solid var(--border); }}
  .hero-title {{ font-family:'Noto Serif SC','Playfair Display',serif; font-size:36px; font-weight:700; }}
  .hero-accent {{ color:var(--accent); }}
  .hero-sub {{
    font-size:14px; color:var(--text-secondary); margin-top:8px;
    font-family:'JetBrains Mono',monospace;
  }}
  .sep {{ display:flex; align-items:center; gap:16px; margin:40px 0 16px; }}
  .sep::before,.sep::after {{ content:''; flex:1; height:1px; background:linear-gradient(to right,transparent,var(--border),transparent); }}
  .sep-icon {{ color:var(--accent-muted); font-size:12px; letter-spacing:4px; }}
  .section-title {{ font-family:'Noto Serif SC',serif; font-size:20px; font-weight:600; margin-bottom:6px; }}
  .section-hint {{ font-size:13px; color:var(--text-secondary); margin-bottom:14px; }}
  .cb {{
    background:var(--bg-surface); border:1px solid var(--border);
    border-radius:8px; padding:24px; margin-bottom:24px;
    box-shadow:0 1px 4px rgba(0,0,0,0.04);
  }}
  .model-cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; margin-bottom:24px; }}
  .model-card {{
    background:var(--bg-surface); border:1px solid var(--border);
    border-radius:8px; padding:20px;
    box-shadow:0 1px 4px rgba(0,0,0,0.04);
  }}
  .model-card h3 {{ font-family:'Noto Serif SC',serif; font-size:16px; margin-bottom:12px; }}
  .metric {{ display:flex; justify-content:space-between; padding:4px 0; font-size:14px; }}
  .metric-label {{ color:var(--text-secondary); }}
  .metric-value {{ font-family:'JetBrains Mono',monospace; font-weight:600; }}
  .tag {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600; }}
  .tag-active {{ background:rgba(196,92,38,0.12); color:#c45c26; }}
  .tag-redundant {{ background:rgba(45,106,79,0.12); color:#2d6a4f; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }}
  .footer {{
    text-align:center; padding:32px 0; margin-top:48px;
    border-top:2px solid var(--border);
  }}
  .footer-name {{ font-family:'Noto Serif SC',serif; font-size:14px; font-weight:600; }}
  .footer-detail {{ font-size:12px; color:var(--text-secondary); font-family:'JetBrains Mono',monospace; margin-top:6px; }}
  @keyframes fadeUp {{ from{{opacity:0;transform:translateY(12px)}} to{{opacity:1;transform:translateY(0)}} }}
  .fade-in {{ animation:fadeUp 0.5s ease-out backwards; }}
  @media (max-width:768px) {{ .grid2{{grid-template-columns:1fr}} }}
</style>
</head>
<body>
<div class="container">
  <div class="hero">
    <div class="hero-title">UniVis <span class="hero-accent">模型对比</span></div>
    <div class="hero-sub">{len(models)} 个模型 · 跨架构冗余分析</div>
  </div>

  <div class="model-cards" id="overview-cards"></div>

  <div class="sep"><span class="sep-icon">&#9670; &#9671; &#9670;</span></div>
  <div class="section-title">层级活跃度对比</div>
  <div class="section-hint">分组柱状图 — 每组 = 一层，每种颜色 = 一个模型。</div>
  <div class="cb"><div id="comparison-chart" style="height:380px;"></div></div>

  <div class="grid2">
    <div>
      <div class="sep"><span class="sep-icon">&#9671;</span></div>
      <div class="section-title">模型雷达图</div>
      <div class="section-hint">多维度画像 — 平均 Delta、余弦相似度、稀疏度。</div>
      <div class="cb"><div id="radar-chart" style="height:320px;"></div></div>
    </div>
    <div>
      <div class="sep"><span class="sep-icon">&#9671;</span></div>
      <div class="section-title">详细对比表</div>
      <div class="section-hint">单元格背景强度 = 活跃度。</div>
      <div class="cb" style="overflow-x:auto">
        <table id="comparison-table" style="width:100%;border-collapse:collapse;font-size:13px;">
          <tr id="table-header"></tr>
        </table>
      </div>
    </div>
  </div>

  <div class="footer">
    <div class="footer-name">UniVis — 面向生成式大模型的时空冗余可视化与智能分析平台</div>
    <div class="footer-detail">中山大学 · v0.5 · 模型对比报告</div>
  </div>
</div>
<script>
(function(){{
  var models = {models_json};
  var palette = ['#c45c26','#2d6a4f','#d4a373','#7c6f64','#8b5e3c'];

  // Overview cards
  var cardsDiv = document.getElementById('overview-cards');
  models.forEach(function(m, i) {{
    var card = document.createElement('div');
    card.className = 'model-card';
    card.style.borderTop = '3px solid ' + palette[i % palette.length];
    card.innerHTML = '<h3>' + m.name + '</h3>' +
      '<div class="metric"><span class="metric-label">层数</span><span class="metric-value">' + m.num_layers + '</span></div>' +
      '<div class="metric"><span class="metric-label">Token 数</span><span class="metric-value">' + m.num_steps + '</span></div>' +
      '<div class="metric"><span class="metric-label">Avg RelDelta</span><span class="metric-value" style="color:'+palette[i%palette.length]+'">' + m.avg_delta.toFixed(4) + '</span></div>' +
      (m.most_active ? '<div class="metric"><span class="metric-label">最活跃层</span><span class="tag tag-active">Layer '+m.most_active.idx+'</span></div>' : '') +
      (m.most_redundant ? '<div class="metric"><span class="metric-label">最冗余层</span><span class="tag tag-redundant">Layer '+m.most_redundant.idx+'</span></div>' : '');
    cardsDiv.appendChild(card);
  }});

  // Grouped bar chart
  var maxLayers = Math.max.apply(null, models.map(function(m){{ return m.num_layers; }}));
  var layerCats = [];
  for (var i = 0; i < maxLayers; i++) layerCats.push('L' + i);
  var series = models.map(function(m, i) {{
    return {{
      name: m.name, type: 'bar',
      data: m.summary.map(function(s){{ return s.avg_delta; }}),
      itemStyle: {{ color: palette[i % palette.length], borderRadius: [3,3,0,0] }},
      barGap: '10%'
    }};
  }});
  var compChart = echarts.init(document.getElementById('comparison-chart'));
  compChart.setOption({{
    tooltip: {{ trigger: 'axis',
      backgroundColor: 'rgba(255,252,248,0.98)', borderColor: '#e0dbd5', borderWidth: 1,
      textStyle: {{ color: '#1a1a2e', fontFamily: "'Noto Sans SC', serif" }}
    }},
    legend: {{ top: 4, textStyle: {{ color: '#6b6b7b' }} }},
    grid: {{ top: 48, left: '8%', right: '4%', bottom: '10%' }},
    xAxis: {{ type: 'category', data: layerCats, name: 'Layer',
      axisLine: {{ lineStyle: {{ color: '#e0dbd5' }} }}, axisLabel: {{ color: '#6b6b7b' }},
      nameTextStyle: {{ color: '#6b6b7b' }}
    }},
    yAxis: {{ type: 'value', name: 'Avg RelDelta',
      axisLine: {{ lineStyle: {{ color: '#e0dbd5' }} }}, axisLabel: {{ color: '#6b6b7b' }},
      nameTextStyle: {{ color: '#6b6b7b' }},
      splitLine: {{ lineStyle: {{ color: '#f0ece6', width: 0.5, type: 'dashed' }} }}
    }},
    series: series
  }});

  // Radar chart
  var radarChart = echarts.init(document.getElementById('radar-chart'));
  var radarIndicators = [
    {{ name: 'Avg Delta', max: 0.3 }},
    {{ name: 'Avg CosSim', max: 1.0 }},
    {{ name: 'Avg Sparsity', max: 1.0 }}
  ];
  var radarSeries = models.map(function(m, i) {{
    return {{
      value: [m.avg_delta, m.summary.length ? m.summary.reduce(function(s,l){{return s+l.avg_cosim;}},0)/m.summary.length : 0,
              m.summary.length ? m.summary.reduce(function(s,l){{return s+l.avg_sparsity;}},0)/m.summary.length : 0],
      name: m.name,
      lineStyle: {{ color: palette[i % palette.length], width: 2 }},
      areaStyle: {{ color: palette[i % palette.length].replace(')', ',0.1)').replace('#','rgba(').replace(/([0-9a-f]{{2}})/gi, function(m){{ return parseInt(m,16)+','; }}) }},
      itemStyle: {{ color: palette[i % palette.length] }}
    }};
  }});
  radarChart.setOption({{
    tooltip: {{
      backgroundColor: 'rgba(255,252,248,0.98)', borderColor: '#e0dbd5', borderWidth: 1,
      textStyle: {{ color: '#1a1a2e' }}
    }},
    legend: {{ bottom: 0, textStyle: {{ color: '#6b6b7b' }} }},
    radar: {{
      indicator: radarIndicators,
      axisName: {{ color: '#6b6b7b' }},
      splitArea: {{ areaStyle: {{ color: ['rgba(240,236,230,0.3)','rgba(240,236,230,0.1)'] }} }},
      splitLine: {{ lineStyle: {{ color: '#e0dbd5' }} }},
      axisLine: {{ lineStyle: {{ color: '#e0dbd5' }} }}
    }},
    series: [{{ type: 'radar', data: radarSeries, animationDuration: 1000 }}]
  }});

  // Comparison table
  var header = document.getElementById('table-header');
  var thIdx = document.createElement('th');
  thIdx.textContent = 'Layer';
  thIdx.style.cssText = 'padding:8px 12px;text-align:left;color:#6b6b7b;border-bottom:1px solid #e0dbd5;';
  header.appendChild(thIdx);
  models.forEach(function(m, i) {{
    var th = document.createElement('th');
    th.textContent = m.name;
    th.style.cssText = 'padding:8px 12px;text-align:center;color:'+palette[i%palette.length]+';border-bottom:1px solid #e0dbd5;font-family:JetBrains Mono,monospace;font-size:12px;';
    header.appendChild(th);
  }});
  var tbody = document.querySelector('#comparison-table');
  for (var li = 0; li < maxLayers; li++) {{
    var tr = document.createElement('tr');
    var tdIdx = document.createElement('td');
    tdIdx.textContent = 'Layer ' + li;
    tdIdx.style.cssText = 'padding:8px 12px;font-weight:600;color:#6b6b7b;font-family:JetBrains Mono,monospace;font-size:12px;border-bottom:1px solid #e0dbd5;';
    tr.appendChild(tdIdx);
    models.forEach(function(m, mi) {{
      var td = document.createElement('td');
      td.style.cssText = 'padding:8px 12px;text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;border-bottom:1px solid #e0dbd5;';
      var layer = m.summary.find(function(s){{ return s.idx === li; }});
      if (layer) {{
        td.textContent = layer.avg_delta.toFixed(4);
        var it = Math.min(layer.avg_delta / 0.3, 1);
        td.style.background = 'rgba(196,92,38,' + (0.03+it*0.15) + ')';
        td.style.color = it > 0.5 ? '#c45c26' : '#2d6a4f';
      }} else {{
        td.textContent = '—'; td.style.color = '#d4a373';
      }}
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }}

  window.addEventListener('resize', function() {{ compChart.resize(); radarChart.resize(); }});
}})();
</script>
</body>
</html>"""

    out.write_text(html, encoding='utf-8')
    return str(out)
