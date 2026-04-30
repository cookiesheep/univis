"""UniVis Qwen2.5-0.5B demo: test on a real model with GPU.

Usage (on L20 server or any GPU machine):
    python examples/qwen_basic.py

This script tests:
1. Hook overhead on a real model (should be < 15%)
2. Metric value ranges for a real model (determines heatmap color scale)
3. Full pipeline with WebSocket push to server

Requires: GPU with ~2GB VRAM, Qwen2.5-0.5B-Instruct model downloaded.
"""

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import univis


MODEL_NAME = os.environ.get('UNIVIS_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')
PROMPT = 'Explain the concept of computational redundancy in neural networks.'
MAX_TOKENS = 50


def benchmark(model, tokenizer, input_ids, num_tokens: int, label: str) -> float:
    """Run inference and return average time per token in ms."""
    ids = input_ids.clone()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_tokens):
            outputs = model(ids)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=-1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    per_token = elapsed / num_tokens * 1000
    print(f'  {label}: {elapsed*1000:.0f} ms total, {per_token:.1f} ms/token')
    return elapsed


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Model:  {MODEL_NAME}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device,
    )
    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(device)
    print(f'Prompt: "{PROMPT[:60]}..." ({input_ids.shape[1]} tokens)')
    print(f'Generating {MAX_TOKENS} tokens\n')

    # ── Baseline (no hooks) ──
    print('=== Baseline ===')
    baseline = benchmark(model, tokenizer, input_ids, MAX_TOKENS, 'baseline')

    # ── With UniVis hooks ──
    print('\n=== With UniVis ===')
    tracker = univis.attach(model, project='qwen_overhead_test', transport='file')

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    ids = input_ids.clone()
    with torch.no_grad():
        for i in range(MAX_TOKENS):
            outputs = model(ids)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_text = tokenizer.decode(next_token[0])
            ids = torch.cat([ids, next_token], dim=-1)
            tracker.on_step(
                token_index=i,
                generated_token=token_text,
                logits=outputs.logits[:, -1, :],
            )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    hooked = time.perf_counter() - t0

    report_path = tracker.finish()

    # ── Overhead Analysis ──
    overhead_pct = (hooked - baseline) / baseline * 100
    print(f'\n=== Overhead ===')
    print(f'  Baseline:  {baseline*1000:.0f} ms')
    print(f'  Hooked:    {hooked*1000:.0f} ms')
    print(f'  Overhead:  {overhead_pct:+.1f}%')
    if overhead_pct < 15:
        print('  PASS: overhead < 15%')
    elif overhead_pct < 30:
        print('  WARNING: overhead 15-30%')
    else:
        print('  FAIL: overhead > 30%')

    # ── Generated Text ──
    generated = tokenizer.decode(ids[0], skip_special_tokens=True)
    print(f'\nGenerated: {generated[:200]}...')
    print(f'Report:    {report_path}')


if __name__ == '__main__':
    main()
