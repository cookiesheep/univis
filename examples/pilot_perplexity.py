"""Measure perplexity impact of Pilot layer-skip on a real model.

Computes cross-entropy/perplexity over a set of sentences for baseline vs
skip (Pilot applied), quantifying the quality cost of the 1.29x speedup.

Usage (on L20):
    PYTHONPATH=src python examples/pilot_perplexity.py
"""

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import univis
from univis import Pilot, PilotPolicy

MODEL = 'Qwen/Qwen2.5-7B-Instruct'

# Diverse Chinese sentences for perplexity probing
SENTENCES = [
    '人工智能是计算机科学的一个重要分支。',
    'Transformer 架构推动了自然语言处理的快速发展。',
    '大语言模型在推理时存在大量的计算冗余。',
    '中山大学位于广东省广州市大学城。',
    '梯度下降是训练神经网络最常用的优化算法。',
    '注意力机制允许模型关注输入序列的不同位置。',
    '开源软件对学术研究和工业应用都至关重要。',
    '气候变暖是全球面临的重大挑战之一。',
    '量子计算机利用量子叠加原理进行计算。',
    '良好的代码规范能够显著提升团队协作效率。',
]


def compute_ce(model, tok, sentences, device):
    """Mean per-token cross-entropy over sentences."""
    total_loss, total_tokens = 0.0, 0
    for s in sentences:
        ids = tok(s, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            out = model(ids, use_cache=False)
            logits = out.logits[:, :-1, :]
            target = ids[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                reduction='sum',
            )
        total_loss += loss.item()
        total_tokens += target.numel()
    return total_loss / max(total_tokens, 1)


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading {MODEL} on {device}...')
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='auto')
    model.eval()

    # 1. monitor for layer summary
    tracker = univis.attach(model, transport='file', output_dir='/tmp/pilot_ppl')
    lp = tracker.logits_processor(tok)
    inp = tok('测试句', return_tensors='pt').input_ids.to(model.device)
    with torch.no_grad():
        model.generate(inp, logits_processor=[lp], max_new_tokens=20, use_cache=False)
    tracker.finish()
    summary = tracker.layer_summary

    policy = PilotPolicy.from_layer_summary(summary, cos_threshold=0.95)
    print(f'model layers: {len(summary)}  skip policy: {len(policy.skip_layers)} layers')

    # 2. baseline CE
    base_ce = compute_ce(model, tok, SENTENCES, model.device)

    # 3. skip CE (pilot applied)
    pilot = Pilot(model, policy)
    with pilot:
        skip_ce = compute_ce(model, tok, SENTENCES, model.device)

    base_ppl = math.exp(base_ce)
    skip_ppl = math.exp(skip_ce)
    delta_pct = (skip_ppl / base_ppl - 1) * 100

    print('\n=== Perplexity result ===')
    print(f'baseline CE: {base_ce:.4f}  ppl: {base_ppl:.2f}')
    print(f'skip CE:     {skip_ce:.4f}  ppl: {skip_ppl:.2f}')
    print(f'ppl increase: {skip_ppl - base_ppl:+.2f}  ({delta_pct:+.1f}%)')
    print(f'(lower is better; <+10% typically acceptable for inference acceleration)')


if __name__ == '__main__':
    main()
