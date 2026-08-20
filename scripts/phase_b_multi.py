"""Multi-prompt cross-hardware protocol: 10 diverse prompts per model.

Same decoding protocol as scripts/phase_b_run.py (raw prompt, greedy,
50 new tokens, bf16) but loops over a fixed diverse prompt set, so the
cross-hardware profile-agreement claim rests on more than one prompt.

Env: UNIVIS_SRC, UNIVIS_OUT, MODELS (comma-separated HF ids)
"""

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.environ.get('UNIVIS_SRC', '/data/univis/src'))
import univis

PROMPTS = [
    '请简述 Transformer 模型中注意力机制的作用，并讨论为什么深层网络可能出现计算冗余。',
    '请解释什么是梯度下降，并说明学习率的作用。',
    'Write a short paragraph about the history of the internet.',
    '用 Python 实现二分查找算法。',
    '简述量子计算与经典计算的区别。',
    'Translate to Chinese: The quick brown fox jumps over the lazy dog.',
    '为什么开源软件对科学研究很重要？',
    'Explain the difference between TCP and UDP briefly.',
    '写一首关于秋天的四行短诗。',
    '列举三种常见的排序算法及其时间复杂度。',
]
NTOK = 50


def run(mid: str, outdir: str) -> None:
    label = mid.split('/')[-1]
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if os.environ.get('UNIVIS_DEVICE_MAP') == 'auto':
        model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.eval()
    tracker = univis.attach(model, project=f'multi-{label}', transport='file', output_dir=outdir)
    lp = tracker.logits_processor(tok)
    for i, p in enumerate(PROMPTS):
        ids = tok(p, return_tensors='pt').input_ids.to(model.device)
        with torch.no_grad():
            model.generate(ids, logits_processor=[lp], max_new_tokens=NTOK, do_sample=False)
        print(f'[{label}] prompt {i + 1}/{len(PROMPTS)} done', flush=True)
    print(f'[{label}] report:', tracker.finish(), flush=True)
    del model
    torch.cuda.empty_cache()


def main() -> None:
    outdir = os.environ.get('UNIVIS_OUT', '/data/phase-b-multi')
    Path(outdir).mkdir(parents=True, exist_ok=True)
    models = os.environ.get('MODELS', 'Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct').split(',')
    for mid in models:
        t0 = time.time()
        try:
            run(mid.strip(), outdir)
            print(f'[{mid}] OK {time.time() - t0:.0f}s', flush=True)
        except RuntimeError as e:
            print(f'[{mid}] FAIL: {str(e)[:200]}', flush=True)


if __name__ == '__main__':
    main()
