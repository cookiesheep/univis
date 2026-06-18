"""Pilot demo: verify layer-skip on a real HF model (tiny-gpt2) — must not crash,
output must stay sane. Run locally on CPU in seconds.

Usage:
    python examples/pilot_demo.py
"""

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import univis
from univis import Pilot, PilotPolicy

MODEL = 'sshleifer/tiny-gpt2'
PROMPT = 'The meaning of life is'
N = 20


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()

    # 1. Monitor: collect per-layer summary
    tracker = univis.attach(model, transport='file', output_dir='/tmp/pilot_demo')
    lp = tracker.logits_processor(tok)
    inp = tok(PROMPT, return_tensors='pt').input_ids
    with torch.no_grad():
        model.generate(inp, logits_processor=[lp], max_new_tokens=N)
    tracker.finish()
    summary = tracker.layer_summary
    cos_vals = [l['avg_cosim'] for l in summary]
    print(f'model layers: {len(summary)}  cos range: {min(cos_vals):.3f}-{max(cos_vals):.3f}')

    # 2. Pilot: build policy (low threshold to force some skips for the test)
    policy = PilotPolicy.from_layer_summary(summary, cos_threshold=0.5, max_skip_ratio=0.5)
    pilot = Pilot(model, policy)
    n = pilot.apply()
    print(f'pilot: skipping {n}/{len(summary)} layers: {sorted(policy.skip_layers)}')

    # 3. generate WITH pilot layer-skip — use_cache=False (KV cache incompatible
    #    with skipped layers, which produce no present key/value)
    inp2 = tok(PROMPT, return_tensors='pt').input_ids
    with torch.no_grad():
        out_skip = model.generate(inp2, max_new_tokens=N, use_cache=False)
    skip_text = tok.decode(out_skip[0][inp2.shape[1]:], skip_special_tokens=True)
    print(f'skip gen:     {skip_text!r}')

    pilot.restore()

    # 4. baseline (no skip, same use_cache=False for fair comparison)
    with torch.no_grad():
        out_base = model.generate(inp2, max_new_tokens=N, use_cache=False)
    base_text = tok.decode(out_base[0][inp2.shape[1]:], skip_special_tokens=True)
    print(f'baseline gen: {base_text!r}')

    print(f'\nearly_exit triggered: {pilot.early_exit_count}')
    print('OK: pilot layer-skip ran on real model without crash')


if __name__ == '__main__':
    main()
