"""Layer-wise numeric diff: find where two runs of the same model start to diverge.

record() runs a fixed token set through the model in one teacher-forced prefill
per prompt and keeps, for every block output, a small random-projection sketch
computed on the device (dim floats per position instead of the hidden state).
Because the projection is linear, sketches from two machines are enough to
estimate per-layer relative error: P(a) - P(b) = P(a - b).

compare() turns two fingerprints (plus an optional known-good baseline) into a
three-state verdict: PASS / DIVERGE (with onset layer) / UNKNOWN.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .detection import detect_block_prefixes

FORMAT = 'univis-fingerprint/1'
NOISE_RATIO = 3.0   # candidate error / baseline error that counts as out of tolerance
JUMP_RATIO = 3.0    # error growth over the previous layer that counts as a divergence onset
MIN_ERROR = 1e-3    # relative errors below this are never flagged

DEFAULT_PROMPTS = [
    'The capital of France is Paris. The capital of Japan is',
    '请用一句话解释为什么天空是蓝色的。',
    'def fibonacci(n):\n    if n < 2:\n        return n\n    return',
    '1, 1, 2, 3, 5, 8, 13, 21, 34,',
    'Translate to English: 开源是 AI 基础设施发展的基石。',
    'Q: If a train travels 120 km in 1.5 hours, what is its average speed?\nA:',
    '注意力机制让模型在生成每个词时',
    'import torch\nx = torch.randn(4, 8)\ny = x @ x.T\nprint(y.shape)  #',
]


def projection(hidden: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Rademacher projection [hidden, dim] derived from SHA-3, identical on every machine."""
    n = hidden * dim
    raw = hashlib.shake_256(f'univis-diff/{seed}/{hidden}/{dim}'.encode()).digest((n + 7) // 8)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n].astype(np.float32)
    return torch.from_numpy((bits * 2 - 1).reshape(hidden, dim) / dim ** 0.5)


def _blocks(model: nn.Module) -> list[nn.Module]:
    prefixes = detect_block_prefixes(model)
    found = {}
    for name, module in model.named_modules():
        for p in prefixes:
            if name.startswith(p) and name[len(p):].isdigit():
                found[int(name[len(p):])] = module
    if not found:
        raise ValueError('No Transformer blocks detected; is this a decoder model?')
    return [found[i] for i in sorted(found)]


_ATTN_NAMES = ('self_attn', 'attn', 'attention', 'self_attention')
_MLP_NAMES = ('mlp', 'feed_forward', 'ffn')


def _sublayers(blocks: list[nn.Module]) -> list[tuple[nn.Module, nn.Module]] | None:
    """(attention, mlp) child modules of every block, or None if any block lacks them."""
    pairs = []
    for b in blocks:
        attn = next((getattr(b, n) for n in _ATTN_NAMES if isinstance(getattr(b, n, None), nn.Module)), None)
        mlp = next((getattr(b, n) for n in _MLP_NAMES if isinstance(getattr(b, n, None), nn.Module)), None)
        if attn is None or mlp is None:
            return None
        pairs.append((attn, mlp))
    return pairs


def _encode(t: torch.Tensor) -> str:
    return base64.b64encode(t.contiguous().numpy().astype('<f4').tobytes()).decode()


def _decode(s: str, dim: int) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype='<f4').reshape(-1, dim)


def _env(model: nn.Module, device: torch.device) -> dict[str, Any]:
    env = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'device': str(device),
        'dtype': str(next(model.parameters()).dtype).replace('torch.', ''),
    }
    try:
        import transformers
        env['transformers'] = transformers.__version__
    except ImportError:
        pass
    if device.type == 'cuda':
        env['device_name'] = torch.cuda.get_device_name(device)
    return env


@torch.no_grad()
def record(
    model: nn.Module,
    tokenizer: Any = None,
    prompts: list[str] | None = None,
    token_ids: list[list[int]] | None = None,
    dim: int = 64,
    seed: int = 0,
    label: str = '',
    sublayers: bool = True,
) -> dict[str, Any]:
    """Teacher-forced pass over a fixed token set; returns a fingerprint dict.

    With sublayers=True, attention and MLP outputs are sketched too, so a divergence
    can be attributed to one of them.
    """
    if token_ids is None:
        if tokenizer is None:
            raise ValueError('Pass a tokenizer or explicit token_ids.')
        token_ids = [tokenizer(p).input_ids for p in (prompts or DEFAULT_PROMPTS)]
    blocks = _blocks(model)
    device = next(model.parameters()).device
    proj: dict[torch.device, torch.Tensor] = {}
    step: dict[Any, torch.Tensor] = {}

    def sketch(h: torch.Tensor) -> torch.Tensor:
        if h.device not in proj:
            proj[h.device] = projection(h.shape[-1], dim, seed).to(h.device)
        return h[0].float() @ proj[h.device]

    def make_hook(i: int):
        def hook(module, args, kwargs, output):
            if i == 0:
                h_in = args[0] if args else kwargs.get('hidden_states')
                step[-1] = sketch(h_in)
            step[i] = sketch(output[0] if isinstance(output, tuple) else output)
        return hook

    def make_sub_hook(key: tuple[str, int]):
        def hook(module, args, output):
            step[key] = sketch(output[0] if isinstance(output, tuple) else output)
        return hook

    handles = [b.register_forward_hook(make_hook(i), with_kwargs=True) for i, b in enumerate(blocks)]
    subs = _sublayers(blocks) if sublayers else None
    for i, (attn, mlp) in enumerate(subs or []):
        handles.append(attn.register_forward_hook(make_sub_hook(('attn', i))))
        handles.append(mlp.register_forward_hook(make_sub_hook(('mlp', i))))
    layers: list[list[torch.Tensor]] = [[] for _ in range(len(blocks) + 1)]
    sub_chunks: dict[str, list[list[torch.Tensor]]] = {k: [[] for _ in blocks] for k in ('attn', 'mlp')}
    top1: list[list[int]] = []
    try:
        for ids in token_ids:
            step.clear()
            logits = model(torch.tensor([ids], device=device)).logits
            top1.append(logits[0].argmax(-1).tolist())
            batch = torch.stack([step[i] for i in range(-1, len(blocks))]).cpu()
            for i in range(len(blocks) + 1):
                layers[i].append(batch[i])
            if subs:
                for k in sub_chunks:
                    sub = torch.stack([step[(k, i)] for i in range(len(blocks))]).cpu()
                    for i in range(len(blocks)):
                        sub_chunks[k][i].append(sub[i])
    finally:
        for h in handles:
            h.remove()
    name = getattr(getattr(model, 'config', None), '_name_or_path', '') or type(model).__name__
    fp = {
        'format': FORMAT,
        'label': label,
        'model': name,
        'num_layers': len(blocks),
        'dim': dim,
        'seed': seed,
        'env': _env(model, device),
        'token_ids': token_ids,
        'top1': top1,
        'layers': [_encode(torch.cat(chunks)) for chunks in layers],
    }
    if subs:
        fp['sublayers'] = {k: [_encode(torch.cat(c)) for c in v] for k, v in sub_chunks.items()}
    return fp


def save(fp: dict[str, Any], path: str | Path) -> Path:
    """Write a fingerprint as JSON."""
    path = Path(path)
    path.write_text(json.dumps(fp, ensure_ascii=False), encoding='utf-8')
    return path


def load(path: str | Path) -> dict[str, Any]:
    """Read a fingerprint and check its format tag."""
    fp = json.loads(Path(path).read_text(encoding='utf-8'))
    if fp.get('format') != FORMAT:
        raise ValueError(f'{path}: not a {FORMAT} file')
    return fp


def _check_pair(a: dict, b: dict) -> None:
    for key in ('num_layers', 'dim', 'seed', 'token_ids'):
        if a[key] != b[key]:
            raise ValueError(f'Fingerprints differ in {key!r}; record both with the same token set '
                             '(use --tokens-from on the second machine).')


def _rel_cos(a_s: str, b_s: str, dim: int) -> tuple[float, float]:
    a, b = _decode(a_s, dim), _decode(b_s, dim)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    safe = na > 1e-12
    rel = float(np.mean(np.linalg.norm(b - a, axis=1)[safe] / na[safe]))
    cos = float(np.mean((a * b).sum(1)[safe] / (na[safe] * np.maximum(nb[safe], 1e-12))))
    return rel, cos


def layer_errors(ref: dict, cand: dict) -> tuple[list[float], list[float]]:
    """Per-layer mean relative L2 error and cosine (index 0 = embedding output)."""
    _check_pair(ref, cand)
    pairs = [_rel_cos(a, b, ref['dim']) for a, b in zip(ref['layers'], cand['layers'])]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _attribute(ref: dict, cand: dict, baseline: dict | None, block: int) -> dict[str, float] | None:
    """Error ratio of the attention and MLP outputs at the onset block.

    Against the baseline when given, otherwise against the same sublayer one block earlier.
    """
    sides = [ref, cand] + ([baseline] if baseline is not None else [])
    if not all('sublayers' in fp for fp in sides) or (baseline is None and block < 1):
        return None
    out = {}
    for k in ('attn', 'mlp'):
        err = _rel_cos(ref['sublayers'][k][block], cand['sublayers'][k][block], ref['dim'])[0]
        if baseline is not None:
            tol = _rel_cos(ref['sublayers'][k][block], baseline['sublayers'][k][block], ref['dim'])[0]
        else:
            tol = _rel_cos(ref['sublayers'][k][block - 1], cand['sublayers'][k][block - 1], ref['dim'])[0]
        out[k] = err / max(tol, MIN_ERROR / NOISE_RATIO) if err > MIN_ERROR else 1.0
    return out


def _top1_agree(ref: dict, cand: dict) -> float:
    pairs = [(x, y) for ra, ca in zip(ref['top1'], cand['top1']) for x, y in zip(ra, ca)]
    return sum(x == y for x, y in pairs) / max(len(pairs), 1)


def compare(ref: dict, cand: dict, baseline: dict | None = None) -> dict[str, Any]:
    """Three-state verdict for cand against ref, using baseline as the tolerance if given."""
    rel, cos = layer_errors(ref, cand)
    result: dict[str, Any] = {
        'ref': {'label': ref['label'], 'model': ref['model'], 'env': ref['env']},
        'cand': {'label': cand['label'], 'model': cand['model'], 'env': cand['env']},
        'rel_error': rel,
        'cosine': cos,
        'top1_agreement': _top1_agree(ref, cand),
        'onset': None,
    }
    if baseline is not None:
        noise, _ = layer_errors(ref, baseline)
        ratio = [r / max(n, 1e-12) if r > MIN_ERROR else 1.0 for r, n in zip(rel, noise)]
        onset = next((i for i, x in enumerate(ratio) if x > NOISE_RATIO), None)
        result.update(baseline={'label': baseline['label'], 'env': baseline['env']},
                      baseline_error=noise, ratio=ratio, method='baseline',
                      baseline_top1_agreement=_top1_agree(ref, baseline))
        result['verdict'] = 'DIVERGE' if onset is not None else 'PASS'
    else:
        # embedding -> block 0 always jumps (rounding -> compute error), so start at block 1
        # and compare against the largest error seen so far
        growth = [1.0, 1.0]
        for i in range(2, len(rel)):
            prev = max(max(rel[1:i]), MIN_ERROR / JUMP_RATIO)
            growth.append(rel[i] / prev if rel[i] > MIN_ERROR else 1.0)
        onset = next((i for i, g in enumerate(growth) if g > JUMP_RATIO), None)
        result.update(ratio=growth, method='jump')
        result['verdict'] = 'DIVERGE' if onset is not None else 'UNKNOWN'
    result['onset'] = onset
    result['onset_sublayer'] = None
    if onset is not None and onset >= 1:
        ratios = _attribute(ref, cand, baseline, onset - 1)
        if ratios:
            result['sublayer_ratio'] = ratios
            # attention runs first inside a block, so it wins when both are out of tolerance
            hit = [k for k in ('attn', 'mlp') if ratios[k] > NOISE_RATIO]
            result['onset_sublayer'] = hit[0] if hit else None
    result['summary'] = _summary(result)
    return result


def _layer_name(i: int) -> str:
    return 'embedding' if i == 0 else f'block {i - 1}'


def _summary(r: dict) -> str:
    if r['verdict'] == 'PASS':
        return (f"PASS: every layer stays within {NOISE_RATIO:g}x of the baseline error "
                f"(max {max(r['ratio']):.2f}x).")
    if r['verdict'] == 'DIVERGE':
        i = r['onset']
        kind = 'the baseline error' if r['method'] == 'baseline' else 'the previous layer'
        where = {'attn': ', attention output', 'mlp': ', MLP output'}.get(r.get('onset_sublayer'), '')
        return (f"DIVERGE: error first exceeds {kind} by {r['ratio'][i]:.1f}x at {_layer_name(i)}{where} "
                f"(relative error {r['rel_error'][i]:.2e}).")
    return ('UNKNOWN: no abrupt jump from block 1 on, but without a known-good baseline '
            'the error level (and the embedding / first block) cannot be judged.')


_VERDICT_STYLE = {
    'PASS': ('#1f8a5b', '#e8f5ee', '一致 · within tolerance'),
    'DIVERGE': ('#c2410c', '#fdf0e8', '分叉 · divergence found'),
    'UNKNOWN': ('#6b7280', '#f1f2f4', '无法判定 · no baseline'),
}


def render_html(result: dict[str, Any], path: str | Path) -> Path:
    """Self-contained HTML report for a compare() result."""
    from .report import _get_echarts_source

    color, bg, tag = _VERDICT_STYLE[result['verdict']]
    names = [_layer_name(i) for i in range(len(result['rel_error']))]

    def env_row(side: dict) -> str:
        e = side['env']
        dev = e.get('device_name', e.get('device', ''))
        return (f"<td>{side['label'] or '—'}</td><td>{dev}</td><td>{e.get('dtype', '')}</td>"
                f"<td>torch {e.get('torch', '')}</td>")

    rows = [('ref', result['ref']), ('cand', result['cand'])]
    if 'baseline' in result:
        rows.append(('baseline', result['baseline']))
    env_html = ''.join(f'<tr><th>{k}</th>{env_row(v)}</tr>' for k, v in rows)
    positive = [v for v in result['rel_error'] + (result.get('baseline_error') or []) if v > 0] or [1e-6]
    payload = json.dumps({
        'names': names, 'rel': [v if v > 0 else None for v in result['rel_error']],
        'ratio': result['ratio'], 'onset': result['onset'],
        'base': [v if v > 0 else None for v in result['baseline_error']] if 'baseline_error' in result else None,
        'ymin': 10 ** math.floor(math.log10(min(positive))), 'ymax': 10 ** math.ceil(math.log10(max(positive))),
        'threshold': NOISE_RATIO if result['method'] == 'baseline' else JUMP_RATIO,
        'method': result['method'], 'color': color,
    })
    agree = f"{result['top1_agreement']:.1%}"
    if 'baseline_top1_agreement' in result:
        agree += f" <span class='muted'>(baseline {result['baseline_top1_agreement']:.1%})</span>"
    html = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UniVis diff · {result['cand']['model']}</title>
{_get_echarts_source(offline=True)}
<style>
body{{margin:0;background:#faf9f9;color:#1f2328;font:15px/1.6 "PingFang SC","Microsoft YaHei","Noto Sans CJK SC",-apple-system,sans-serif}}
.wrap{{max-width:1040px;margin:0 auto;padding:32px 20px 48px}}
.brand{{font-size:13px;letter-spacing:.12em;color:#6b7280;text-transform:uppercase}}
h1{{font-size:26px;margin:6px 0 18px;font-weight:650}}
.card{{background:#fff;border:1px solid #e3e5e8;border-radius:14px;padding:20px 22px;margin-bottom:18px}}
.verdict{{background:{bg};border-color:{color}33}}
.pill{{display:inline-block;padding:3px 12px;border-radius:999px;background:{color};color:#fff;font-weight:600;font-size:13px}}
.big{{font-size:18px;margin-top:10px}}
.kpis{{display:flex;gap:28px;flex-wrap:wrap;margin-top:14px}}
.kpi b{{display:block;font-size:22px}} .muted{{color:#6b7280;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #eef0f2}}
th{{color:#6b7280;font-weight:500;width:80px}}
.chart{{height:360px}} .chart.small{{height:240px}}
footer{{color:#6b7280;font-size:12px;margin-top:24px}}
</style></head><body><div class="wrap">
<div class="brand">UniVis · layer-wise diff</div>
<h1>{result['cand']['model']}</h1>
<div class="card verdict"><span class="pill">{result['verdict']}</span> <span class="muted">{tag}</span>
<div class="big">{result['summary']}</div>
<div class="kpis"><div class="kpi"><span class="muted">分叉起点 onset</span><b>{names[result['onset']] if result['onset'] is not None else '—'}</b></div>
<div class="kpi"><span class="muted">子层 sublayer</span><b>{ {'attn': 'attention', 'mlp': 'MLP'}.get(result.get('onset_sublayer'), '—') }</b></div>
<div class="kpi"><span class="muted">末层相对误差 last-layer error</span><b>{result['rel_error'][-1]:.2e}</b></div>
<div class="kpi"><span class="muted">top-1 一致率 agreement</span><b>{agree}</b></div></div></div>
<div class="card"><div class="muted">逐层相对误差（对数轴）· per-layer relative error vs reference</div><div id="err" class="chart"></div></div>
<div class="card"><div class="muted">{'相对基线的误差倍数' if result['method'] == 'baseline' else '相对上一层的误差增长'} · ratio (threshold dashed)</div><div id="ratio" class="chart small"></div></div>
<div class="card"><table>{env_html}</table></div>
<footer>Generated by UniVis diff — sketches are {len(names)} layers × random projections; relative error is estimated from linear sketches. 结论只覆盖本次记录的模型、提示词集与软件栈，不外推。</footer>
</div><script>
var D={payload};
var mark=D.onset===null?[]:[{{xAxis:D.names[D.onset],lineStyle:{{color:D.color,type:'solid',width:2}},label:{{formatter:'onset',color:D.color}}}}];
var series=[{{name:'candidate',type:'line',data:D.rel,symbolSize:5,lineStyle:{{width:2.5,color:D.color}},itemStyle:{{color:D.color}},markLine:{{symbol:'none',data:mark}}}}];
if(D.base)series.push({{name:'baseline (tolerance)',type:'line',data:D.base,symbolSize:4,lineStyle:{{width:2,color:'#9ca3af',type:'dashed'}},itemStyle:{{color:'#9ca3af'}}}});
var axis={{type:'category',data:D.names,axisLabel:{{color:'#6b7280',fontSize:11}},axisLine:{{lineStyle:{{color:'#e3e5e8'}}}}}};
echarts.init(document.getElementById('err')).setOption({{grid:{{left:60,right:20,top:30,bottom:50}},tooltip:{{trigger:'axis',valueFormatter:function(v){{return v==null?'':v.toExponential(2)}}}},legend:{{top:0,right:0,textStyle:{{color:'#6b7280'}}}},xAxis:axis,yAxis:{{type:'log',min:D.ymin,max:D.ymax,axisLabel:{{color:'#6b7280'}},splitLine:{{lineStyle:{{color:'#f0f1f3'}}}}}},series:series}});
echarts.init(document.getElementById('ratio')).setOption({{grid:{{left:60,right:20,top:20,bottom:50}},tooltip:{{trigger:'axis'}},xAxis:axis,yAxis:{{type:'value',axisLabel:{{color:'#6b7280'}},splitLine:{{lineStyle:{{color:'#f0f1f3'}}}}}},series:[{{type:'bar',data:D.ratio.map(function(v,i){{return{{value:v,itemStyle:{{color:v>D.threshold?D.color:'#cfd4da',borderRadius:[3,3,0,0]}}}}}}),markLine:{{symbol:'none',lineStyle:{{color:'#6b7280',type:'dashed'}},data:[{{yAxis:D.threshold}}],label:{{formatter:D.threshold+'x'}}}}}}]}});
window.addEventListener('resize',function(){{['err','ratio'].forEach(function(id){{echarts.getInstanceByDom(document.getElementById(id)).resize()}})}});
</script></body></html>'''
    path = Path(path)
    path.write_text(html, encoding='utf-8')
    return path
