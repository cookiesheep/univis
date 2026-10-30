"""`univis diff` subcommands: record a fingerprint, compare fingerprints."""

from __future__ import annotations

import argparse
import json
import sys

EXIT = {'PASS': 0, 'DIVERGE': 1, 'UNKNOWN': 3}
DTYPES = {'fp32': 'float32', 'bf16': 'bfloat16', 'fp16': 'float16'}


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register `diff record` and `diff compare`."""
    p = sub.add_parser('diff', help='Layer-wise numeric diff between two runs of a model')
    dsub = p.add_subparsers(dest='diff_command', required=True)

    r = dsub.add_parser('record', help='Record a layer fingerprint (teacher-forced, fixed tokens)')
    r.add_argument('model', help='HF model name or path')
    r.add_argument('-o', '--output', required=True, help='Fingerprint JSON path')
    r.add_argument('--dtype', choices=sorted(DTYPES), default='bf16')
    r.add_argument('--device', default=None, help='cpu / cuda / cuda:1 (default: cuda if available)')
    r.add_argument('--tokens-from', default=None, help='Reuse the exact token ids of another fingerprint')
    r.add_argument('--dim', type=int, default=64, help='Sketch dimension (default: 64)')
    r.add_argument('--label', default='', help='Free-text label, e.g. "C500 MXMACA 3.3"')

    d = dsub.add_parser('demo', help='CPU demo: tell a simulated wrong kernel apart from bf16 noise')
    d.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M', help='Llama-style HF model')
    d.add_argument('--block', type=int, default=17, help='Block whose MLP activation gets swapped')
    d.add_argument('-o', '--output', default='univis_diff_demo.html', help='HTML report path')

    c = dsub.add_parser('compare', help='Compare a candidate fingerprint with a reference')
    c.add_argument('ref', help='Reference fingerprint (e.g. fp32 or a trusted GPU)')
    c.add_argument('cand', help='Candidate fingerprint')
    c.add_argument('--baseline', default=None, help='Known-good fingerprint in the candidate precision')
    c.add_argument('-o', '--output', default=None, help='HTML report path')
    c.add_argument('--json', default=None, help='Write the full result as JSON')


def _load(name: str, dtype_name: str, device: str = 'cpu'):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, DTYPES[dtype_name])
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return model.to(device).eval()


def _demo(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    from . import diff

    print(f'[1/3] fp32 reference: {args.model} on CPU')
    ref = diff.record(_load(args.model, 'fp32'), AutoTokenizer.from_pretrained(args.model),
                      label='CPU fp32 reference')
    print('[2/3] bf16, correct (the tolerance baseline)')
    good = diff.record(_load(args.model, 'bf16'), token_ids=ref['token_ids'], label='CPU bf16 (correct)')
    print(f'[3/3] bf16 with a wrong activation kernel in block {args.block}')
    broken = _load(args.model, 'bf16')
    broken.model.layers[args.block].mlp.act_fn = torch.nn.GELU(approximate='tanh')
    bad = diff.record(broken, token_ids=ref['token_ids'], label=f'CPU bf16, wrong kernel in block {args.block}')
    ok, res = diff.compare(ref, good), diff.compare(ref, bad, good)
    print(f"\n  correct bf16 : top-1 {ok['top1_agreement']:.1%}  (used as the tolerance baseline)")
    print(f"  broken  bf16 : top-1 {res['top1_agreement']:.1%}  {res['summary']}")
    print('\nSame output-level agreement, but the layer view pinpoints the block.')
    print('Report:', diff.render_html(res, args.output))


def run(args: argparse.Namespace) -> None:
    """Dispatch a parsed `univis diff` command."""
    from . import diff

    if args.diff_command == 'demo':
        _demo(args)
        return

    if args.diff_command == 'record':
        import torch
        from transformers import AutoTokenizer

        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model = _load(args.model, args.dtype, device)
        token_ids = diff.load(args.tokens_from)['token_ids'] if args.tokens_from else None
        tok = None if token_ids else AutoTokenizer.from_pretrained(args.model)
        fp = diff.record(model, tok, token_ids=token_ids, dim=args.dim, label=args.label)
        diff.save(fp, args.output)
        print(f"Recorded {fp['num_layers']} blocks x {sum(map(len, fp['token_ids']))} positions -> {args.output}")
        return

    ref, cand = diff.load(args.ref), diff.load(args.cand)
    baseline = diff.load(args.baseline) if args.baseline else None
    result = diff.compare(ref, cand, baseline)
    print(result['summary'])
    print(f"top-1 agreement {result['top1_agreement']:.1%}")
    if args.output:
        print('Report:', diff.render_html(result, args.output))
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
    sys.exit(EXIT[result['verdict']])
