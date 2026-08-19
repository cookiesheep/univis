"""UniVis Phase B: same-prompt cross-hardware baseline.

Mirrors the original L20 cross-scale protocol exactly: raw prompt (no chat
template), greedy decoding, 50 new tokens, bf16 — so C500 results are
directly comparable with L20 results.

Usage: python phase_b_run.py <LABEL> <MODEL_PATH_OR_HF_ID>
Env: UNIVIS_SRC (default <repo>/src), UNIVIS_OUT (default ./phase-b-out)
"""

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.environ.get('UNIVIS_SRC', str(Path(__file__).resolve().parent.parent / 'src')))
import univis

PROMPT = '请简述 Transformer 模型中注意力机制的作用，并讨论为什么深层网络可能出现计算冗余。'
NTOK = 50


def main() -> None:
    label, model_path = sys.argv[1], sys.argv[2]
    outdir = os.environ.get('UNIVIS_OUT', './phase-b-out')
    Path(outdir).mkdir(parents=True, exist_ok=True)
    print(f'=== {label} ({model_path}) ===', flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f'loaded {time.time() - t0:.0f}s layers={model.config.num_hidden_layers} mem={mem:.2f}GB', flush=True)

    tracker = univis.attach(model, project=label, transport='file', output_dir=outdir)
    lp = tracker.logits_processor(tok)
    ids = tok(PROMPT, return_tensors='pt').input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, logits_processor=[lp], max_new_tokens=NTOK, do_sample=False)
    report = tracker.finish()
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f'OK report={report}', flush=True)
    print(f'gen={gen[:120]}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        kind = 'OOM' if 'out of memory' in str(e).lower() else 'FAIL'
        print(f'{kind} {sys.argv[1]}: {str(e)[:300]}', flush=True)
        if kind == 'FAIL':
            traceback.print_exc()
