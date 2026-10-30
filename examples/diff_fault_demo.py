"""CPU demo for `univis diff`: can a single wrong kernel be told apart from precision noise?

Records three fingerprints of the same model on the same tokens:
  ref      fp32                      (reference)
  bf16     bf16, correct             (a correct lower-precision stack, used as tolerance)
  faulty   bf16, one block's MLP activation swapped SiLU -> tanh-GELU (a simulated wrong kernel)
then compares them and writes two HTML reports.

Usage:
    python examples/diff_fault_demo.py [model] [block]
    python examples/diff_fault_demo.py HuggingFaceTB/SmolLM2-135M 17
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from univis import diff

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'HuggingFaceTB/SmolLM2-135M'
BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 17


def load(dtype: torch.dtype) -> torch.nn.Module:
    try:
        return AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).eval()
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).eval()


tok = AutoTokenizer.from_pretrained(MODEL)
ref = diff.record(load(torch.float32), tok, label='CPU fp32 reference')
ids = ref['token_ids']
bf16 = diff.record(load(torch.bfloat16), token_ids=ids, label='CPU bf16 (correct)')
broken = load(torch.bfloat16)
broken.model.layers[BLOCK].mlp.act_fn = torch.nn.GELU(approximate='tanh')
faulty = diff.record(broken, token_ids=ids, label=f'CPU bf16, wrong activation in block {BLOCK}')
for fp, name in ((ref, 'ref'), (bf16, 'bf16'), (faulty, 'faulty')):
    diff.save(fp, f'diff_{name}.json')

for title, cand, base in (('correct bf16, no baseline', bf16, None),
                          ('faulty, no baseline', faulty, None),
                          ('faulty vs bf16 baseline', faulty, bf16)):
    res = diff.compare(ref, cand, base)
    print(f"{title:28s} {res['verdict']:8s} top-1 {res['top1_agreement']:.1%}  {res['summary']}")
diff.render_html(diff.compare(ref, faulty, bf16), 'diff_faulty_report.html')
diff.render_html(diff.compare(ref, bf16), 'diff_bf16_report.html')
print('Reports: diff_faulty_report.html, diff_bf16_report.html')
