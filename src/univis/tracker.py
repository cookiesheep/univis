"""Tracker: user-facing API for recording inference steps."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .detection import detect_block_prefixes, get_layer_count
from .metrics import compute_entropy
from .probe import ModelProbe
from .report import generate_report
from .transport import FileTransport, HttpPushTransport, MultiTransport, Transport


class Tracker:
    """High-level interface for tracking model inference."""

    def __init__(
        self,
        model: nn.Module,
        probe: ModelProbe,
        transport: Transport,
        session_id: str,
        model_name: str,
        num_layers: int,
        layer_names: list[str],
        output_dir: str = '.',
    ) -> None:
        self._model = model
        self._probe = probe
        self._transport = transport
        self._session_id = session_id
        self._model_name = model_name
        self._num_layers = num_layers
        self._layer_names = layer_names
        self._output_dir = output_dir
        self._start_time = time.perf_counter()
        self._step_count = 0
        self._all_steps: list[dict] = []
        self._vram_start = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )
        self._finished = False

    def on_step(
        self,
        token_index: int,
        generated_token: str = '',
        logits: torch.Tensor | None = None,
    ) -> None:
        """Mark one token generation step complete. Flushes metrics to transport."""
        layers = self._probe.flush_step(token_index)

        entropy = compute_entropy(logits) if logits is not None else -1.0

        vram_total = (
            torch.cuda.memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )

        message: dict[str, Any] = {
            'type': 'step',
            'session_id': self._session_id,
            'token_idx': token_index,
            'timestamp_ms': int(time.time() * 1000),
            'generated_token': generated_token,
            'global': {
                'vram_total_mb': round(vram_total, 1),
                'prediction_entropy': round(entropy, 4),
            },
            'layers': layers,
        }

        self._transport.send(message)
        self._all_steps.append(message)
        self._step_count += 1

    def finish(self, output_dir: str | None = None) -> str:
        """End tracking, remove hooks, generate HTML report."""
        if self._finished:
            return ''
        self._finished = True

        elapsed = time.perf_counter() - self._start_time

        end_msg = {
            'type': 'session_end',
            'session_id': self._session_id,
            'total_tokens': self._step_count,
            'total_time_ms': round(elapsed * 1000, 1),
            'num_layers': self._num_layers,
        }
        self._transport.send(end_msg)
        self._transport.close()
        self._probe.remove_hooks()

        report_path = self._generate_report(output_dir or self._output_dir)
        return str(report_path)

    def remove(self) -> None:
        """Remove hooks without generating report (for cleanup on error)."""
        if not self._finished:
            self._probe.remove_hooks()
            self._transport.close()
            self._finished = True

    def logits_processor(self, tokenizer=None):
        """Return a logits processor for model.generate().

        Usage:
            tracker = univis.attach(model, transport='file')
            lp = tracker.logits_processor(tokenizer)
            output = model.generate(input_ids, logits_processor=[lp], max_new_tokens=50)
            tracker.finish()

        Args:
            tokenizer: Optional tokenizer for decoding generated tokens.

        Returns:
            A callable compatible with HuggingFace LogitsProcessorList.
        """
        tracker = self
        tok = tokenizer

        def processor(input_ids, scores):
            if not tracker._finished:
                next_token_id = scores.argmax(dim=-1)
                token_text = ''
                if tok is not None:
                    token_text = tok.decode(next_token_id[0])
                tracker.on_step(
                    token_index=tracker._step_count,
                    generated_token=token_text,
                    logits=scores,
                )
            return scores

        return processor

    def _generate_report(self, output_dir: str) -> Path:
        """Generate HTML report with ECharts visualizations."""
        meta = {
            'session_id': self._session_id,
            'model_name': self._model_name,
            'num_layers': self._num_layers,
        }
        out_path = Path(output_dir) / f'univis_report_{self._session_id[:8]}.html'
        generate_report(self._all_steps, meta, out_path, offline=True)
        return out_path


def create_tracker(
    model: nn.Module,
    project: str = 'default',
    hook_prefixes: list[str] | None = None,
    transport_mode: str = 'file',
    output_dir: str = '.',
    port: int = 8765,
) -> Tracker:
    """Factory: build a Tracker with auto-detected hooks and configured transport."""
    session_id = uuid.uuid4().hex[:16]

    # Detect model structure
    prefixes = hook_prefixes or detect_block_prefixes(model)
    if not prefixes:
        raise ValueError('No valid block prefixes found. Pass hook_prefixes manually.')
    num_layers = get_layer_count(model, prefixes[0])

    layer_names = [
        name for name, _ in model.named_modules()
        if any(name.startswith(p) and name[len(p):].isdigit() for p in prefixes)
    ]

    model_name = getattr(model, 'name_or_path', type(model).__name__)
    if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
        model_name = model.config._name_or_path

    # Create probe
    probe = ModelProbe(model, prefixes)

    # Create transport
    transports: list[Transport] = [
        FileTransport(Path(output_dir) / f'univis_data_{session_id[:8]}.jsonl'),
    ]
    if transport_mode == 'websocket':
        transports.append(HttpPushTransport(f'http://127.0.0.1:{port}', session_id))

    transport = MultiTransport(transports) if len(transports) > 1 else transports[0]

    # Send session_start
    start_msg: dict[str, Any] = {
        'type': 'session_start',
        'session_id': session_id,
        'model_name': model_name,
        'num_layers': num_layers,
        'layer_names': layer_names,
        'project': project,
    }
    transport.send(start_msg)

    return Tracker(
        model=model,
        probe=probe,
        transport=transport,
        session_id=session_id,
        model_name=model_name,
        num_layers=num_layers,
        layer_names=layer_names,
        output_dir=output_dir,
    )
