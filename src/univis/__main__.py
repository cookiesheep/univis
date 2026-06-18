"""CLI entry point: python -m univis serve / python -m univis report <jsonl_path>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog='univis', description='UniVis CLI')
    sub = parser.add_subparsers(dest='command')

    # serve
    serve_p = sub.add_parser('serve', help='Start the WebSocket server')
    serve_p.add_argument('--host', default='0.0.0.0', help='Bind host (default: 0.0.0.0)')
    serve_p.add_argument('--port', type=int, default=8765, help='Bind port (default: 8765)')

    # report
    report_p = sub.add_parser('report', help='Generate HTML report from JSONL data')
    report_p.add_argument('jsonl_path', help='Path to the JSONL data file')
    report_p.add_argument('-o', '--output', default=None, help='Output HTML path')

    # compare
    compare_p = sub.add_parser('compare', help='Compare models from multiple JSONL files')
    compare_p.add_argument('jsonl_paths', nargs='+', help='Two or more JSONL data files to compare')
    compare_p.add_argument('-o', '--output', default=None, help='Output HTML path')

    # pilot
    pilot_p = sub.add_parser('pilot', help='Pilot intervention: monitor + layer-skip + speedup compare')
    pilot_p.add_argument('model', help='HF model name or path')
    pilot_p.add_argument('--cos-threshold', type=float, default=0.95, help='Cosine similarity threshold for redundant layers (default: 0.95)')
    pilot_p.add_argument('--max-skip-ratio', type=float, default=0.3, help='Max fraction of layers to skip (default: 0.3)')
    pilot_p.add_argument('--max-tokens', type=int, default=50, help='Tokens to generate (default: 50)')
    pilot_p.add_argument('--prompt', default='The meaning of life is', help='Prompt text')
    pilot_p.add_argument('--output-dir', default='.', help='Dir for monitor JSONL/report')

    args = parser.parse_args()

    if args.command == 'serve':
        import uvicorn
        print(f'Server running at ws://{args.host}:{args.port}')
        uvicorn.run('univis.server:app', host=args.host, port=args.port, reload=False)

    elif args.command == 'report':
        from univis.report import generate_report

        jsonl = Path(args.jsonl_path)
        if not jsonl.exists():
            print(f'Error: file not found: {jsonl}', file=sys.stderr)
            sys.exit(1)

        messages: list[dict] = []
        with open(jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(json.loads(line))

        steps = [m for m in messages if m.get('type') == 'step']
        if not steps:
            print('Error: no step messages found in JSONL file', file=sys.stderr)
            sys.exit(1)

        meta: dict = {}
        for m in messages:
            if m.get('type') == 'session_start':
                meta = m
                break

        if not args.output:
            session_id = meta.get('session_id', 'unknown')[:8]
            args.output = str(jsonl.parent / f'univis_report_{session_id}.html')

        generate_report(steps, meta, args.output)
        print(f'Report saved to: {args.output}')

    elif args.command == 'compare':
        from univis.report import generate_comparison_report

        if len(args.jsonl_paths) < 2:
            print('Error: compare requires at least 2 JSONL files', file=sys.stderr)
            sys.exit(1)

        for p in args.jsonl_paths:
            if not Path(p).exists():
                print(f'Error: file not found: {p}', file=sys.stderr)
                sys.exit(1)

        if not args.output:
            args.output = 'univis_comparison_report.html'

        generate_comparison_report(args.jsonl_paths, args.output)
        print(f'Comparison report saved to: {args.output}')

    elif args.command == 'pilot':
        import os
        import time
        os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import univis
        from univis import Pilot, PilotPolicy

        print(f'Loading {args.model}...')
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.eval()

        # 1. monitor to get layer summary
        tracker = univis.attach(model, transport='file', output_dir=args.output_dir)
        lp = tracker.logits_processor(tok)
        inp = tok(args.prompt, return_tensors='pt').input_ids
        with torch.no_grad():
            model.generate(inp, logits_processor=[lp], max_new_tokens=args.max_tokens)
        tracker.finish()
        summary = tracker.layer_summary

        # 2. build policy + apply
        policy = PilotPolicy.from_layer_summary(
            summary, cos_threshold=args.cos_threshold, max_skip_ratio=args.max_skip_ratio)
        pilot = Pilot(model, policy)
        n = pilot.apply()

        # 3. baseline vs skip (use_cache=False), measure wall-clock with warmup
        # to remove cold-start bias (review: skip-first inflates speedup).
        inp2 = tok(args.prompt, return_tensors='pt').input_ids

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def _timed_gen():
            _sync()
            t0 = time.perf_counter()
            out = model.generate(inp2, max_new_tokens=args.max_tokens, use_cache=False)
            _sync()
            return out, time.perf_counter() - t0

        # warmup both paths (2 tokens) so neither is cold-start biased
        with torch.no_grad():
            model.generate(inp2, max_new_tokens=2, use_cache=False)
            with pilot:
                model.generate(inp2, max_new_tokens=2, use_cache=False)

        # baseline first (cold->warm balanced), then skip via context manager
        with torch.no_grad():
            out_base, t_base = _timed_gen()
            with pilot:
                out_skip, t_skip = _timed_gen()

        speedup = (t_base / t_skip) if t_skip > 0 else 0.0
        print('\n=== Pilot result ===')
        pct = n / max(len(summary), 1) * 100
        print(f'model layers: {len(summary)}  skipped: {n}  ({pct:.0f}%)')
        print(f'baseline: {t_base:.3f}s   skip: {t_skip:.3f}s   speedup: {speedup:.2f}x')
        print(f'baseline gen: {tok.decode(out_base[0][inp2.shape[1]:], skip_special_tokens=True)[:100]}')
        print(f'skip gen:     {tok.decode(out_skip[0][inp2.shape[1]:], skip_special_tokens=True)[:100]}')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
