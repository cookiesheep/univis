# Contributing to UniVis

Thanks for your interest in improving UniVis! The project is young and there are many good ways to contribute.

## Good first contributions

- **New metrics** — `metrics.py` functions are pure (`Tensor → float`), independently testable, and auto-aggregated for batched inputs. Adding a metric is low-risk.
- **Architecture adapters** — extend `detection.py` to auto-detect new Transformer families, or add per-architecture hook targets.
- **Dashboard views** — the React + ECharts frontend welcomes new visualizations (e.g. attention/FFN drill-downs).
- **Benchmarks / model coverage** — run UniVis on more models and share redundancy reports.
- **Domestic-accelerator testing** — reports of how hooks and metrics behave on domestic GPUs (MXMACA stack, etc.) are especially valuable.

## Development setup

```bash
pip install -e ".[dev]"          # SDK + dev tools
python -m pytest tests/ -v       # 80 unit tests
cd dashboard && npm install && npm run dev   # dashboard
```

## Workflow

1. Fork and branch from `main`.
2. Keep changes focused; add or update tests under `tests/`.
3. Make sure `pytest tests/` passes.
4. Open a pull request describing the motivation and what changed.

## Code style

- Python 3.10+, type hints, single quotes, 4-space indent.
- Keep hook code paths cheap — metrics must be edge-computable.
- Public API stays exported from `__init__.py`.

## Reporting issues

When reporting a bug, include: model + revision, PyTorch / OS, the UniVis version, and a minimal reproduction. Sample JSONL or the generated HTML report help a lot.

## License

By contributing you agree your contributions are licensed under the project's [MIT license](LICENSE).
