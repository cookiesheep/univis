# UniVis

**Open-source observability for Transformer inference — see which layers compute, and which are redundant.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-80-brightgreen.svg)](tests)

> Read this in: [简体中文](README.zh-CN.md)

UniVis is a lightweight diagnostic toolkit that attaches to any PyTorch / HuggingFace Transformer and visualizes per-layer behavior during inference. By computing compact metrics on the fly through `forward` hooks, it produces dynamic heatmaps and standalone reports that make *"where is compute actually happening — and where is it being wasted"* immediately legible.

It is a **measurement & visualization** tool. It does not modify, prune, or accelerate your model — it gives you the evidence to decide what to optimize, and offers an experimental path to act on it.

## Why

Transformer inference is expensive, and not every layer or generation step contributes equally. Existing tools tend to look elsewhere:

- **TensorBoard / W&B** track training-time scalars (loss, learning rate) and treat the network as a black box at inference time.
- **BertViz** explains attention semantics, not runtime compute cost.
- **NVIDIA Nsight / profilers** operate at the GPU-operator level — powerful, but hard to map back to *"which Transformer layer."*

UniVis fills the missing semantic layer: **per-layer, per-token redundancy during inference**, transmitted as just a few KB per step.

## Features

- **Zero-intrusion hooks** — no changes to the target model; attach and run.
- **Edge-computed metrics** — tensors are reduced to scalars inside the hook, so each step's payload is ~1–2 KB with negligible inference overhead.
- **5 core metrics** — relative delta, cosine similarity, activation sparsity, prediction entropy, VRAM delta.
- **Architecture auto-detection** — GPT-2, LLaMA / Mistral / Mixtral, Qwen (1.5 / 2 / 2.5 / 3), BERT family.
- **Three output modes** — JSONL log, standalone offline HTML report, real-time WebSocket dashboard.
- **`model.generate()` integration** — plugs in via a HuggingFace `LogitsProcessor`, no manual loop changes.
- **Rich reports** — model-MRI rings, layer-pulse sparklines, ThemeRiver data-river, plasma heatmap, redundancy ranking with trend annotations.
- **Multi-model comparison** — `univis compare` cross-model redundancy CLI with radar charts and tables.
- **Pilot intervention (experimental)** — threshold-based early-exit during generation, with quantitative perplexity impact.

## How it works

```
your model
    │  univis.attach(model)
    ▼
┌──────────────────────── univis SDK ───────────────────────┐
│  detection → probe (forward_hook) → metrics → transport    │
│           (auto-detect)  (per-layer)   (scalar)  (JSONL/HTTP)│
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

The design principle is **edge computing**: high-dimensional activations are reduced to scalar metrics right inside the hook, so monitoring never blocks inference.

## Quick start

```bash
pip install -e ".[dev]"
```

### Pattern A — Minimal (SDK only, offline report)

```python
import univis

tracker = univis.attach(model, transport="file")
for token_id in generate_loop():
    tracker.on_step(token_id)
report_path = tracker.finish()   # → standalone HTML report
```

### Pattern B — `model.generate()` integration

```python
tracker = univis.attach(model)
lp = tracker.logits_processor(tokenizer)
output = model.generate(input_ids, logits_processor=[lp], max_new_tokens=50)
tracker.finish()
```

### Pattern C — Real-time dashboard

```bash
python -m univis serve              # terminal 1: WebSocket server
cd dashboard && npm run dev         # terminal 2: dashboard (:5173)
python your_script.py               # terminal 3: inference with transport="websocket"
```

## CLI

```bash
python -m univis serve --port 8765               # start the WebSocket server
python -m univis report data.jsonl -o out.html    # render an HTML report from JSONL
python -m univis compare a.jsonl b.jsonl -o c.html  # cross-model comparison
```

## Metrics

| Metric | Formula | Meaning |
|---|---|---|
| Relative Delta | `‖output − input‖₂ / ‖input‖₂` | How much the representation changed at this layer |
| Cosine Similarity | `cos(input, output)` | Directional alignment between input and output |
| Activation Sparsity | `count(|x| < ε) / total` | Fraction of near-zero activations |
| Prediction Entropy | `H(softmax(logits))` | Model uncertainty at this step |
| VRAM Delta | `Δ torch.cuda.memory_allocated()` | Memory pressure introduced by this step |

Relative Delta is the primary heatmap metric. Because Transformers use residual connections, cosine similarity saturates above 0.9 and blurs inter-layer differences; Relative Delta directly measures *"how much this layer changed"* and is far more sensitive to redundancy in residual architectures.

## Supported models

| Architecture | Variants |
|---|---|
| GPT-2 | gpt2, gpt2-medium, gpt2-large, gpt2-xl |
| LLaMA | LLaMA 1/2/3, Mistral, Mixtral |
| Qwen | Qwen 1.5, Qwen 2, Qwen 2.5, Qwen 3 |
| BERT | BERT, RoBERTa, ALBERT, DeBERTa |

Architectures are auto-detected from the model config — no manual registration. Verified on Qwen2.5-0.5B / 3B / 7B and Qwen3-27B-class hybrid models.

## Roadmap

- **Pilot early-exit** — turn measurement into action: exit generation when the model is confident enough, with explicit *quality (perplexity) ↔ speedup* tradeoff curves. (Naive layer-skipping was found to harm quality, so the focus is confidence-based early-exit.)
- **Domestic-accelerator adaptation** — validate hooks and metrics on domestic AI compute (e.g. the MXMACA software stack), so redundancy diagnostics work across NVIDIA and domestic GPUs.
- **DiT / video generation** — monitor temporal redundancy across denoising steps and enable time-step-level early stop.
- **Sub-layer granularity** — drill from Transformer block down to Attention and FFN.
- **Redundancy–quality study** — empirical correlation between UniVis metrics and downstream quality loss after pruning / layer-skip.

## Architecture & testing

Full module design, data model, and API in [ARCHITECTURE.md](ARCHITECTURE.md); scope and motivation in [PRD.md](PRD.md). The SDK is covered by **80 unit tests** across 9 files (metrics, probe, report, server, tracker, pilot, integration, three-end).

```
src/univis/      Python SDK — attach() → on_step() → finish()
dashboard/       React 18 + TypeScript + ECharts frontend
examples/        runnable examples
tests/           80 unit tests
```

## Contributing

Contributions are welcome — new metric plugins, model adapters, dashboard views, and benchmarks are all useful. See [CONTRIBUTING.md](CONTRIBUTING.md). The metric functions in `metrics.py` are pure and independently testable, which makes adding a new metric low-risk.

## License

[MIT](LICENSE).

## Acknowledgements

Computing resources and technical discussions from the AI Systems research group at Sun Yat-sen University. Built on PyTorch, FastAPI, React, and ECharts.
