"""UniVis: Transformer inference redundancy diagnostic and visualization toolkit.

Usage:
    import univis
    tracker = univis.attach(model, project="my_experiment")
    for i in range(max_tokens):
        output = model(input_ids)
        tracker.on_step(token_index=i, generated_token="hello", logits=output.logits[:, -1, :])
    report_path = tracker.finish()
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .tracker import Tracker, create_tracker


def attach(
    model: nn.Module,
    project: str = 'default',
    hook_prefixes: list[str] | None = None,
    transport: str = 'file',
    output_dir: str = '.',
    port: int = 8765,
    session_id: str | None = None,
) -> Tracker:
    """Attach a UniVis tracker to a Transformer model.

    Args:
        model: A HuggingFace (or compatible) Transformer model.
        project: Project name for organizing sessions.
        hook_prefixes: Manual block name prefixes (auto-detected if None).
        transport: "file" for JSONL output, "websocket" for live dashboard.
        output_dir: Directory for output files (JSONL data + HTML report).
        port: WebSocket port (only used when transport="websocket").
        session_id: Optional session ID (auto-generated UUID if None).

    Returns:
        A Tracker instance. Call tracker.on_step() during inference,
        then tracker.finish() to generate the report.
    """
    return create_tracker(
        model=model,
        project=project,
        hook_prefixes=hook_prefixes,
        transport_mode=transport,
        output_dir=output_dir,
        port=port,
        session_id=session_id,
    )


__all__ = ['attach', 'Tracker']
