"""Sweep entropy thresholds for Pilot early-exit: token savings vs quality.

For each threshold, generate greedily with Pilot's early-exit logits
processor and compare against unconstrained baseline generation: tokens
generated, early-stop rate, and how much of the natural output is kept.
Greedy decoding keeps early-exited prefixes identical to baseline, so
quality loss is truncation — quantified here as kept_ratio.

Usage (L20):
    PYTHONPATH=src python examples/pilot_early_exit.py
Smoke test (CPU):
    UNIVIS_MODEL=sshleifer/tiny-gpt2 UNIVIS_THRESHOLDS=1.0,3.0 UNIVIS_MAX_NEW=24 \
        PYTHONPATH=src python examples/pilot_early_exit.py
"""

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from univis import Pilot, PilotPolicy

MODEL = os.environ.get('UNIVIS_MODEL', 'Qwen/Qwen2.5-7B-Instruct')
THRESHOLDS = [float(t) for t in os.environ.get('UNIVIS_THRESHOLDS', '0.05,0.1,0.2,0.5,1.0,2.0').split(',')]
WINDOW = int(os.environ.get('UNIVIS_WINDOW', '1'))
MAX_NEW_TOKENS = int(os.environ.get('UNIVIS_MAX_NEW', '128'))
OUT_PATH = os.environ.get('UNIVIS_OUT', 'early_exit_results.json')

PROMPTS = [
    '用三句话介绍一下 Transformer 架构的核心思想。',
    '解释什么是大语言模型的推理冗余。',
    '把下面的话翻译成英文：开源是 AI 基础设施发展的基石。',
    '列出三种常见的模型压缩方法，各配一句说明。',
    '为什么残差连接会让层间余弦相似度饱和？',
    '写一个 Python 函数计算列表的中位数。',
    '对比 greedy search 和 beam search 的优缺点。',
    '用一句话总结注意力机制的作用。',
]


def build_inputs(tok, prompt):
    """Apply chat template when available, else raw continuation."""
    if getattr(tok, 'chat_template', None):
        text = tok.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    return tok(text, return_tensors='pt')


def generate(model, inputs, processor=None):
    """Greedy generation; returns (new_token_ids, wall seconds)."""
    kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    if processor is not None:
        kwargs['logits_processor'] = [processor]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    elapsed = time.perf_counter() - t0
    return out[0, inputs['input_ids'].shape[-1]:], elapsed


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    print(f'Loading {MODEL} on {device} ({dtype})...')
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device)
    model.eval()
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise SystemExit('tokenizer has no eos_token_id; early-exit cannot force stop')

    inputs_list = [build_inputs(tok, p) for p in PROMPTS]

    # 1. baseline: natural greedy length (EOS the model chooses itself)
    base_new, base_wall = [], 0.0
    for inp in inputs_list:
        new, dt = generate(model, inp)
        base_new.append(new)
        base_wall += dt
    base_lens = [int(t.numel()) for t in base_new]
    print(f'\nbaseline: avg {sum(base_lens) / len(base_lens):.1f} tokens, '
          f'{base_wall:.1f}s total ({len(PROMPTS)} prompts, max {MAX_NEW_TOKENS})')

    # 2. sweep thresholds
    rows = []
    for th in THRESHOLDS:
        policy = PilotPolicy(skip_layers=set(), entropy_threshold=th,
                             early_exit_enabled=True, entropy_window=WINDOW)
        pilot = Pilot(model, policy)
        lens, kept, stops, prefix_ok = [], [], 0, 0
        wall = 0.0
        for inp, base in zip(inputs_list, base_new):
            before = pilot.early_exit_count
            new, dt = generate(model, inp, pilot.logits_processor(eos_id))
            wall += dt
            lens.append(int(new.numel()))
            if pilot.early_exit_count > before:
                stops += 1
            common = min(new.numel(), base.numel())
            if torch.equal(new[:common], base[:common]):
                prefix_ok += 1
            base_len = max(base.numel(), 1)
            kept.append(min(new.numel() / base_len, 1.0))
        rows.append({
            'threshold': th,
            'avg_new_tokens': sum(lens) / len(lens),
            'avg_kept_ratio': sum(kept) / len(kept),
            'early_stop_rate': stops / len(PROMPTS),
            'prefix_intact_rate': prefix_ok / len(PROMPTS),
            'wall_s': wall,
            'token_lens': lens,
        })
        print(f'  th={th:<5} avg_tokens={rows[-1]["avg_new_tokens"]:6.1f}  '
              f'kept={rows[-1]["avg_kept_ratio"]:.2f}  early_stop={stops}/{len(PROMPTS)}  '
              f'prefix_ok={prefix_ok}/{len(PROMPTS)}  {wall:.1f}s')

    # 3. report
    result = {
        'model': MODEL,
        'max_new_tokens': MAX_NEW_TOKENS,
        'entropy_window': WINDOW,
        'num_prompts': len(PROMPTS),
        'baseline': {'avg_new_tokens': sum(base_lens) / len(base_lens), 'wall_s': base_wall, 'token_lens': base_lens},
        'sweep': rows,
    }
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f'\n| threshold | avg tokens | kept vs baseline | early stop | wall (s) |')
    print('|---|---|---|---|---|')
    print(f'| baseline (no exit) | {result["baseline"]["avg_new_tokens"]:.1f} | 1.00 | 0/{len(PROMPTS)} | {base_wall:.1f} |')
    for r in rows:
        print(f'| {r["threshold"]} | {r["avg_new_tokens"]:.1f} | {r["avg_kept_ratio"]:.2f} | '
              f'{r["early_stop_rate"]:.0%} | {r["wall_s"]:.1f} |')
    print(f'\nsaved: {OUT_PATH}')
    print('prefix_intact_rate = 1.0 confirms early-exit only truncates, never rewrites, greedy output.')


if __name__ == '__main__':
    main()
