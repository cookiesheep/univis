"""FastAPI WebSocket server for UniVis real-time metric streaming.

Architecture:
    SDK --HTTP POST--> Server --WebSocket--> Dashboard(s)

The SDK pushes data via POST (sync, reliable).
The server broadcasts to all subscribed Dashboard WebSocket clients.

Usage:
    python -m univis.server
    # Server starts at http://localhost:8765
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


# Session -> list of subscribed WebSocket connections
_subscribers: dict[str, list[WebSocket]] = {}

# Session -> history of messages (for late-joining dashboards)
_history: dict[str, list[dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _subscribers.clear()
    _history.clear()


app = FastAPI(title='UniVis Server', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── HTTP Push (SDK → Server) ───────────────────────────────


@app.post('/api/push/{session_id}')
async def push_data(session_id: str, data: dict[str, Any]) -> dict[str, str]:
    """Receive metric data from SDK and broadcast to subscribers."""
    _history.setdefault(session_id, []).append(data)

    payload = json.dumps(data, ensure_ascii=False)
    dead: list[WebSocket] = []

    for ws in _subscribers.get(session_id, []):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)

    for ws in dead:
        _subscribers.get(session_id, []).remove(ws)

    return {'status': 'ok'}


# ── WebSocket Subscribe (Server → Dashboard) ───────────────


@app.websocket('/ws/{session_id}')
async def ws_subscribe(websocket: WebSocket, session_id: str) -> None:
    """Dashboard connects here to receive real-time updates."""
    await websocket.accept()
    _subscribers.setdefault(session_id, []).append(websocket)

    # Send history to late-joining clients
    for msg in _history.get(session_id, []):
        try:
            await websocket.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            break

    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        pass
    finally:
        subs = _subscribers.get(session_id, [])
        if websocket in subs:
            subs.remove(websocket)


# ── Health / Info ───────────────────────────────────────────


@app.get('/api/sessions')
async def list_sessions() -> dict[str, Any]:
    """List active sessions with subscriber counts."""
    return {
        'sessions': {
            sid: {
                'subscribers': len(conns),
                'messages': len(_history.get(sid, [])),
            }
            for sid, conns in _subscribers.items()
        }
    }


@app.get('/api/history/{session_id}')
async def get_history(session_id: str) -> list[dict]:
    """Get full message history for a session."""
    return _history.get(session_id, [])


@app.get('/')
async def index() -> HTMLResponse:
    return HTMLResponse('<h1>UniVis Server</h1><p>Use /ws/{session_id} to subscribe.</p>')


# ── CLI entry point ─────────────────────────────────────────


def main() -> None:
    import uvicorn
    uvicorn.run('univis.server:app', host='0.0.0.0', port=8765, reload=False)


if __name__ == '__main__':
    main()
