"""Three-end integration test: SDK → Server → WebSocket client.

Starts a real uvicorn server so HttpPushTransport can reach it.
"""

import asyncio
import json
import sys
import time
import threading
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Import server module directly (avoid torch dependency in __init__.py)
_server_path = str(Path(__file__).resolve().parent.parent / 'src' / 'univis')
if _server_path not in sys.path:
    sys.path.insert(0, _server_path)

import server as _server_mod

from univis import attach

PORT = 18765


class _FakeBlock(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(x)


class _FakeGPT2(nn.Module):
    def __init__(self, dim: int = 16, n_layers: int = 3, vocab: int = 50) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([_FakeBlock(dim) for _ in range(n_layers)])
        self.lm_head = nn.Linear(dim, vocab)
        self.config = type('Config', (), {'_name_or_path': 'test-three-end'})()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(x)


@pytest.fixture(autouse=True)
def _clear_state():
    _server_mod._subscribers.clear()
    _server_mod._history.clear()
    yield
    _server_mod._subscribers.clear()
    _server_mod._history.clear()


@pytest.fixture(scope='module')
def live_server():
    """Start a real uvicorn server in a background thread."""
    import uvicorn

    def _run():
        uvicorn.run(_server_mod.app, host='127.0.0.1', port=PORT, log_level='error')

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/health', timeout=1)
            break
        except Exception:
            time.sleep(0.2)

    yield PORT


def _run_inference(session_id: str, n_steps: int, port: int, use_finish: bool = False) -> None:
    """Run inference in a separate thread (sync HTTP push)."""
    model = _FakeGPT2(dim=16, n_layers=2)
    tracker = attach(model, session_id=session_id, transport='websocket', port=port)

    x = torch.randn(1, 2, 16)
    with torch.no_grad():
        for i in range(n_steps):
            out = model(x)
            tracker.on_step(i, f't{i}', out[:, -1, :])

    if use_finish:
        tracker.finish(output_dir='/tmp')
    else:
        tracker.remove()


class TestThreeEndHttp:
    """SDK → HTTP → Server tests."""

    def test_sdk_push_and_read_history(self, live_server) -> None:
        """SDK pushes data, server stores it, retrievable via /api/history."""
        import urllib.request

        session_id = 'http-history-test'

        model = _FakeGPT2(dim=16, n_layers=2)
        tracker = attach(model, session_id=session_id, transport='websocket', port=PORT)

        x = torch.randn(1, 2, 16)
        with torch.no_grad():
            for i in range(3):
                out = model(x)
                tracker.on_step(i, f't{i}', out[:, -1, :])
        tracker.remove()

        resp = urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/history/{session_id}')
        history = json.loads(resp.read())
        # session_start + 3 steps = 4 messages
        assert len(history) == 4
        step_msgs = [m for m in history if m.get('type') == 'step']
        assert len(step_msgs) == 3
        for msg in step_msgs:
            assert 'layers' in msg
            assert len(msg['layers']) == 2

    def test_sdk_push_with_websocket_subscriber(self, live_server) -> None:
        """SDK pushes data while WS subscriber is connected — verify WS receives it."""
        import websockets

        session_id = 'ws-live-test'

        async def _run():
            messages: list[dict] = []
            async with websockets.connect(f'ws://127.0.0.1:{PORT}/ws/{session_id}') as ws:
                # Run SDK in a separate thread to avoid blocking event loop
                sdk_thread = threading.Thread(
                    target=_run_inference,
                    args=(session_id, 3, PORT),
                )
                sdk_thread.start()
                sdk_thread.join(timeout=10)

                # Read all available messages
                for _ in range(10):
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=3)
                        messages.append(json.loads(raw))
                    except asyncio.TimeoutError:
                        break

            return messages

        received = asyncio.run(_run())
        step_msgs = [m for m in received if m.get('type') == 'step']
        assert len(step_msgs) == 3
        for msg in step_msgs:
            assert 'layers' in msg
            assert len(msg['layers']) == 2

    def test_late_join_websocket_gets_history(self, live_server) -> None:
        """WS subscriber connecting after push still gets full history."""
        import websockets

        session_id = 'late-join-test'

        # 1. Push data first (no WS connected)
        _run_inference(session_id, 3, PORT)

        # 2. Connect WS — should receive history replay
        async def _run():
            messages: list[dict] = []
            async with websockets.connect(f'ws://127.0.0.1:{PORT}/ws/{session_id}') as ws:
                # History: session_start + 3 steps = 4 messages
                for _ in range(10):
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=3)
                        messages.append(json.loads(raw))
                    except asyncio.TimeoutError:
                        break
            return messages

        received = asyncio.run(_run())
        step_msgs = [m for m in received if m.get('type') == 'step']
        assert len(step_msgs) == 3
        for msg in step_msgs:
            assert 'layers' in msg

    def test_health_endpoint(self, live_server) -> None:
        """Health endpoint reports server is running."""
        import urllib.request
        resp = urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/health')
        body = json.loads(resp.read())
        assert body['status'] == 'ok'
