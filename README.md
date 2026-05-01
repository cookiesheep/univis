# UniVis

**Transformer Inference Redundancy Diagnostic & Visualization Toolkit**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

UniVis is a diagnostic visualization toolkit for Transformer model inference. It attaches to any PyTorch-based model via lightweight hooks and collects per-layer metrics at every generation step, revealing where computation is active and where it is redundant. The result is a dynamic heatmap that makes layer-level behavior immediately visible.

UniVis is a **diagnostic tool**, not an optimization tool. It does not modify, prune, or accelerate models -- it measures and visualizes so that researchers can make informed optimization decisions.

## Features

- **Hook-based metrics collection** -- zero code change to the target model; attach and go
- **4 core metrics** -- relative delta, cosine similarity, activation sparsity, prediction entropy
- **Architecture auto-detection** -- GPT-2, LLaMA, Qwen (1.5/2/2.5), BERT family
- **3 output modes** -- JSONL log, standalone HTML report, real-time WebSocket dashboard
- **`model.generate()` support** -- integrates via HuggingFace `LogitsProcessor` with no loop changes
- **Batch aggregation** -- supports multi-input inference with per-sample tracking

## Quick Start

### Pattern A -- Minimal (SDK only, HTML report)

```python
import univis

tracker = univis.attach(model, transport='file')
for token_id in generate_loop():
    tracker.on_step(token_id)
report_path = tracker.finish()
```

### Pattern B -- `model.generate()` integration

```python
tracker = univis.attach(model)
lp = tracker.logits_processor(tokenizer)
output = model.generate(input_ids, logits_processor=[lp], max_new_tokens=50)
tracker.finish()
```

### Pattern C -- Real-time Dashboard

```bash
# Terminal 1: start server
python -m univis serve

# Terminal 2: start dashboard
cd dashboard && npm run dev

# Terminal 3: run inference
python your_script.py  # with transport='websocket'
```

## Installation

```bash
pip install -e .              # basic SDK
pip install -e ".[dev]"       # with dev tools (pytest, transformers)
pip install -e ".[server]"    # with FastAPI server dependencies
```

## CLI Usage

```bash
python -m univis serve --port 8765              # start WebSocket server
python -m univis report data.jsonl -o out.html   # generate HTML report from JSONL
```

## Supported Models

| Architecture | Variants |
|---|---|
| GPT-2 | gpt2, gpt2-medium, gpt2-large, gpt2-xl |
| LLaMA | LLaMA 1/2/3, Mistral, Mixtral |
| Qwen | Qwen 1.5, Qwen 2, Qwen 2.5 |
| BERT | BERT, RoBERTa, ALBERT, DeBERTa |

Architectures are auto-detected from the model config. No manual registration required.

## Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| Relative Delta | `||output - input||_2 / ||input||_2` | How much representation changed at this layer |
| Cosine Similarity | `cos(input, output)` | Directional alignment between input and output |
| Activation Sparsity | `count(|x| < eps) / total` | Fraction of near-zero activations |
| Prediction Entropy | `H(softmax(logits))` | Model uncertainty at this step |

Relative Delta is the primary heatmap metric. Because Transformers use residual connections, cosine similarity tends to saturate above 0.9, making inter-layer differences hard to distinguish. Relative Delta directly measures "how much this layer changed" and is more sensitive to redundancy in residual architectures.

## Project Structure

```
src/univis/       # Python SDK (pip install -e .)
dashboard/        # React + ECharts frontend
examples/         # Runnable examples
tests/            # Unit tests (67 tests)
```

## License

MIT

## Acknowledgments

Sun Yat-sen University (中山大学) -- Undergraduate Innovation Training Project (大学生创新训练项目)
