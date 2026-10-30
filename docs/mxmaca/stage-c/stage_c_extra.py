"""Stage C extras on C500: vendor flash-attn vs sdpa vs eager, long sequences, run-to-run determinism."""
import inspect
import json
import json.decoder
import textwrap
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from univis import diff

OUT = '/data/univis_stage_c_extra'
ZH = ('大语言模型在推理时逐层计算隐藏状态，每一层都包含注意力与前馈两个子层。'
      '把模型从一种硬件迁移到另一种硬件时，算子实现、累加顺序与精度设置都会改变数值结果。') * 20
LONG_TEXTS = [inspect.getsource(json.decoder), inspect.getsource(textwrap), ZH]
log = {}


def load(name, impl):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation=impl)
    return m.to('cuda').eval()


def timed_forward(model, ids, reps=3):
    x = torch.tensor([ids], device='cuda')
    with torch.no_grad():
        model(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


for size in ('0.5B', '3B'):
    name = f'Qwen/Qwen2.5-{size}-Instruct'
    tok = AutoTokenizer.from_pretrained(name)
    short_ids = diff.load(f'/data/univis_stage_c/c500_bf16_{size}.json')['token_ids']
    long_ids = [tok(t).input_ids[:1024] for t in LONG_TEXTS]
    for impl in ('sdpa', 'flash_attention_2', 'eager'):
        key = f'{size}/{impl}'
        try:
            m = load(name, impl)
        except Exception as exc:
            log[key] = f'load failed: {type(exc).__name__}: {exc}'[:300]
            print(key, log[key], flush=True)
            continue
        fp = diff.record(m, token_ids=short_ids, label=f'MetaX C500 bf16 {impl}')
        diff.save(fp, f'{OUT}/c500_bf16_{impl}_{size}.json')
        fpl = diff.record(m, token_ids=long_ids, label=f'MetaX C500 bf16 {impl} long')
        diff.save(fpl, f'{OUT}/c500_bf16long_{impl}_{size}.json')
        entry = {'long_forward_ms': round(timed_forward(m, long_ids[0]), 2)}
        if impl == 'sdpa':
            again = diff.record(m, token_ids=long_ids, label='rerun')
            entry['rerun_bitwise_identical'] = again['layers'] == fpl['layers']
            entry['rerun_max_rel'] = max(diff.layer_errors(fpl, again)[0])
        log[key] = entry
        print(key, entry, flush=True)
        del m
        torch.cuda.empty_cache()
json.dump(log, open(f'{OUT}/extra_log.json', 'w'), indent=1)
print('== extra done', flush=True)
