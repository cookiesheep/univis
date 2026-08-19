<div align="center">

# UniVis

**Find out which layers of your LLM actually compute at inference — and which are idle.**

*找出大模型推理时哪些层在空转浪费算力 —— Transformer 推理冗余诊断与可视化工具*

`AI Infra · LLM inference optimization & deployment`

[![CI](https://github.com/cookiesheep/univis/actions/workflows/ci.yml/badge.svg)](https://github.com/cookiesheep/univis/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-80-brightgreen.svg)](tests)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

English · [简体中文](README.md)

</div>

<img src="docs/images/dashboard.png" alt="UniVis Dashboard" width="100%">

UniVis attaches to any PyTorch / HuggingFace Transformer via zero-intrusion `forward` hooks, reduces every layer's activations to scalar metrics **at the edge (inside the hook)** — only ~1–2 KB per step — and streams them as JSONL / WebSocket into a real-time dashboard or a standalone offline HTML report. It answers one question: **at inference time, which layers actually compute, and which are redundant.**

UniVis is a **measurement & visualization** tool. It does not modify, prune, or accelerate your model — it gives you the evidence to decide what to optimize, plus an experimental path to act on it (Pilot).

---

## ✨ Showcase

<table>
<tr>
<td width="50%" align="center"><b>Real-time Dashboard</b></td>
<td width="50%" align="center"><b>Model MRI · concentric rings</b></td>
</tr>
<tr>
<td width="50%" align="center"><img src="docs/images/dashboard.png" alt="dashboard"></td>
<td width="50%" align="center"><img src="docs/images/report-mri.png" alt="model MRI"></td>
</tr>
<tr>
<td width="50%" align="center"><sub>Heatmap grows left-to-right as tokens are generated (plasma: dark = redundant, bright = active)</sub></td>
<td width="50%" align="center"><sub>Each layer as a ring — orange = active, gray-green = redundant. Spot the idle layers at a glance.</sub></td>
</tr>
<tr>
<td width="50%" align="center"><b>Layer pulse + data river</b></td>
<td width="50%" align="center"><b>Multi-model comparison</b></td>
</tr>
<tr>
<td width="50%" align="center"><img src="docs/images/report-pulse.png" alt="layer pulse"></td>
<td width="50%" align="center"><img src="docs/images/report-comparison.png" alt="comparison"></td>
</tr>
<tr>
<td width="50%" align="center"><sub>A sparkline per layer: flat = redundant, spiking = active; ThemeRiver shows activity over tokens</sub></td>
<td width="50%" align="center"><sub><code>univis compare</code> cross-model redundancy distribution</sub></td>
</tr>
</table>

## Why UniVis

Transformer inference cost and memory grow explosively with parameter count, yet not every layer or generation step contributes equally — saving compute means saving money, which matters all the more when accelerator capacity is scarce. Before optimizing, you need to know where the redundancy is. Existing tools look elsewhere:

| Tool | Phase | Granularity | Intrusiveness | Barrier |
|---|---|---|---|---|
| TensorBoard / W&B | training | macro scalars (loss, …) | instrumentation | low |
| BertViz | inference | attention semantics | custom scripts | medium |
| Nsight & profilers | inference | GPU operators / kernels | environment-level | high |
| **UniVis** | **inference** | **per-layer × per-token redundancy** | **zero-intrusion hooks** | **low** |

UniVis fills the missing semantic layer in between: **per-layer, per-token redundancy during inference**, at a few KB per step.

## Domestic accelerators & MXMACA adaptation

UniVis's implementation is inherently portable across hardware — every claim below is independently checkable:

- **Pure PyTorch `forward` hooks** — no CUDA-specific API dependency whatsoever;
- metrics computed **at the edge** (inside the hook), ~1–2 KB per step, never a new bottleneck;
- the offline path (JSONL → HTML report) runs on **CPU only**;
- no dependency on any GPU vendor's profiler toolchain.

### Environment matrix

| Environment | Hardware | Status | Verified content |
|---|---|---|---|
| NVIDIA L20 (48GB) | NVIDIA GPU | ✅ **verified** | full diagnostics on Qwen2.5-0.5B / 3B / 7B + 27B-class hybrid (~54× parameter span) |
| CPU (no GPU) | — | ✅ **verified** | full 80-test suite, offline report rendering |
| MetaX Xiyun C500 (64GB HBM2e) | MetaX GPU + MXMACA stack | 🔄 **planned** (phase A → B) | see staged roadmap below |

### Staged roadmap

| Phase | Goal | Completion criterion | Status |
|---|---|---|---|
| A | metric-collection pipeline on Xiyun C500 | one full Qwen2.5-0.5B generation producing an HTML report + archived JSONL + environment fingerprint (exact MXMACA / mcPyTorch versions + `mx-smi` output) | planned |
| B | cross-scale redundancy baseline on C500 | baseline reports for 0.5B / 3B / 7B + a same-model, same-prompt comparison table vs. NVIDIA L20 | planned |
| C | feed diagnostics back into the MetaX ecosystem | use redundancy findings as input to FlagGems / mcTriton operator optimization | exploratory |

### Exact support contract & scope statement

| Item | Current contract |
|---|---|
| Models tested | Qwen2.5-0.5B / 3B / 7B-Instruct (bf16), Qwen3.6-27B (hybrid) |
| Auto-detected architectures | GPT-2 family; LLaMA family (LLaMA 1/2/3, Mistral, Mixtral); Qwen family (1.5 / 2 / 2.5 / 3); BERT family (BERT, RoBERTa, ALBERT, DeBERTa) |
| Software stack | Python ≥ 3.10, PyTorch ≥ 2.0, HuggingFace transformers |
| Hardware | NVIDIA GPU (CUDA) and CPU verified; MetaX MXMACA adaptation in progress (see above) |

**Scope statement:** verification covers only the models and architectures listed above; redundancy findings do not extrapolate to untested models. That is precisely UniVis's value — run one diagnostic on a new model and get *its own* redundancy profile before optimizing anything.

### Auditable without a GPU

Everything UniVis produces — raw JSONL, self-contained HTML reports, environment fingerprints — can be reviewed on a machine with **no GPU at all**. The diagnostic tool is itself the evidence chain: C500 validation results will be published together with their data and fingerprints, auditable by anyone without renting a card.

We look forward to collaborating with MetaX and the MXMACA community on test resources, technical guidance, and ecosystem integration. Adaptation details and fingerprint conventions: [docs/mxmaca-adaptation.md](docs/mxmaca-adaptation.md).

## Core features

- **Low-intrusion SDK** — one line, `univis.attach(model)`: detection auto-identifies the architecture from the model config, probes collect per-layer data via `forward` hooks, zero changes to the target model. 5 core metrics: relative delta, inter-layer cosine similarity, activation sparsity, prediction entropy, VRAM delta.
- **Real-time dashboard** — React 18 + TypeScript + ECharts; the heatmap grows left-to-right as tokens are generated, with statistics panels, filters, and click interactions.
- **Offline HTML report** — single file, zero dependencies, opens anywhere: model-MRI rings, layer-pulse sparklines, ThemeRiver, redundancy ranking with trend annotations.
- **Multi-model comparison CLI** — `univis compare` for cross-model redundancy distributions, with radar charts and tables.
- **Pilot intervention (experimental)** — turning measurement into action: confidence-based early-exit during generation; the negative result of the layer-skip route is fully public (below).
- **Engineering quality** — 80 unit tests green, GitHub Actions CI, 3 CLI entry points, full type hints on Python 3.10+.

## Empirical findings

### 1. Middle-layer redundancy is most pronounced — and holds across scales

<div align="center">

<img src="docs/images/cross-scale.png" alt="cross-scale redundancy" width="92%">

<sub>Qwen2.5-0.5B / 3B / 7B + Qwen3.6-27B (hybrid), ~54× parameter span. With layer depth normalized to [0,1], the middle segment shows markedly higher cosine similarity than shallow/deep layers in all four models.</sub>

</div>

### 2. A public failure: layer-skip does not hold

Pilot v1 followed the intuition "high cosine similarity = skippable" and skipped middle redundant layers. Measured on Qwen2.5-7B (NVIDIA L20):

| Configuration | perplexity | Change |
|---|---|---|
| Baseline (all layers) | 13.81 | — |
| Skip middle redundant layers | 105.67 | **+665%** |

Conclusion: saturated cosine similarity ≠ useless layer; "apparently redundant" layers in residual architectures still perform necessary refinement (full record in commit `1938d60` and `examples/pilot_perplexity.py`). Pilot has therefore pivoted to **confidence-based early-exit**: terminate generation when prediction entropy falls below a threshold — the entropy-threshold × early-stop quality/speedup tradeoff curve is being quantified now.

*All experiments above ran on NVIDIA L20 (48GB); domestic-accelerator validation is tracked in the environment matrix.*

## Quick start

```bash
git clone https://github.com/cookiesheep/univis && cd univis
pip install -e ".[dev]"
```

**A · Offline report (minimal)**

```python
import univis

tracker = univis.attach(model, transport="file")
for token_id in generate_loop():
    tracker.on_step(token_id)
report_path = tracker.finish()   # → standalone HTML report
```

**B · `model.generate()` integration**

```python
tracker = univis.attach(model)
lp = tracker.logits_processor(tokenizer)
output = model.generate(input_ids, logits_processor=[lp], max_new_tokens=50)
tracker.finish()
```

**C · Real-time dashboard**

```bash
python -m univis serve              # terminal 1: WebSocket server (:8765)
cd dashboard && npm run dev         # terminal 2: dashboard (:5173)
python your_script.py               # terminal 3: inference with transport="websocket"
```

Three CLI entry points: `univis serve` (live server), `univis report` (JSONL → HTML), `univis compare` (multi-model comparison).

## Architecture

```
your model
    │  univis.attach(model)
    ▼
┌──────────────────────── univis SDK ───────────────────────┐
│  detection → probe (forward_hook) → metrics → transport    │
│   (auto-detect)  (per-layer)      (scalar)   (JSONL/HTTP)  │
│                                                            │
│   metrics computed at the edge ── only KB of JSON emitted  │
└───────────────────────────┬────────────────────────────────┘
                            │ HTTP POST → WebSocket
                            ▼
                  React + ECharts dashboard
                  (live heatmap · stats · filters)
                            │
                            ▼  tracker.finish()
                  standalone HTML report
```

Three layers: the collection SDK (detection / probe / metrics / transport / tracker / pilot), a FastAPI WebSocket server, and the React frontend. The core design is **edge computing**: high-dimensional activations are reduced to scalars right inside the hook, so monitoring never blocks inference. Full module design and APIs: [ARCHITECTURE.md](ARCHITECTURE.md); scope and motivation: [PRD.md](PRD.md).

## Engineering quality

- **80 unit tests** (8 test files) covering the SDK, server, report generation, the three-terminal pipeline, and Pilot — `pytest tests/` fully green;
- **GitHub Actions CI**: every push runs the full suite on Python 3.10 / 3.12 with CPU torch;
- full type hints on Python ≥ 3.10; 10 SDK modules; public API exported from `__init__.py`.

```
src/univis/      Python SDK — attach() → on_step() → finish()
dashboard/       React 18 + TypeScript + ECharts frontend
docs/images/     sample reports and charts
examples/        runnable examples
tests/           80 unit tests
```

## Roadmap

- **Near term** — Pilot early-exit tradeoff curves (entropy threshold × early stop → quantified quality/speedup); MetaX Xiyun C500 adaptation (phases A / B).
- **Mid term** — deeper MXMACA validation (phase C, integrating with the FlagGems / mcTriton ecosystem); DiT / video-generation models (temporal redundancy across denoising steps); hooks drilled down to Attention / FFN sub-layers.
- **Long term** — 70B+ and MoE models; integration with mainstream inference frameworks.

## Community & collaboration

UniVis is used for day-to-day diagnostics and teaching demos in our lab. A GitLink mirror is in preparation. Contributions are welcome — new metric plugins, model adapters, dashboard views, and benchmarks are all useful. See [CONTRIBUTING.md](CONTRIBUTING.md); the metric functions in `metrics.py` are pure and independently testable, which makes adding a new metric low-risk.

## License

[MIT](LICENSE)

## Acknowledgements

Computing resources and technical discussions from the AI Systems research group at Sun Yat-sen University. Built on PyTorch, FastAPI, React, and ECharts.
