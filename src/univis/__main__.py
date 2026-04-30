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

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
