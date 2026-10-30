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

    c = dsub.add_parser('compare', help='Compare a candidate fingerprint with a reference')
    c.add_argument('ref', help='Reference fingerprint (e.g. fp32 or a trusted GPU)')
    c.add_argument('cand', help='Candidate fingerprint')
    c.add_argument('--baseline', default=None, help='Known-good fingerprint in the candidate precision')
    c.add_argument('-o', '--output', default=None, help='HTML report path')
    c.add_argument('--json', default=None, help='Write the full result as JSON')


def run(args: argparse.Namespace) -> None:
    """Dispatch a parsed `univis diff` command."""
    from . import diff

    if args.diff_command == 'record':
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = getattr(torch, DTYPES[args.dtype])
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        model = model.to(device).eval()
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
