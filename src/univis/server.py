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

import json
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

_MAX_HISTORY = 10000
_SESSION_RE = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')

# Session -> list of subscribed WebSocket connections
_subscribers: dict[str, list[WebSocket]] = {}

# Session -> history of messages (for late-joining dashboards)
_history: dict[str, list[dict]] = {}


def _validate_session(session_id: str) -> JSONResponse | None:
    if not _SESSION_RE.match(session_id):
        return JSONResponse(status_code=400, content={'error': 'invalid session_id'})
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    for ws_list in _subscribers.values():
        for ws in ws_list:
            try:
                await ws.close(code=1001, reason='server shutdown')
            except Exception:
                pass
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
async def push_data(session_id: str, request: Request) -> JSONResponse:
    """Receive metric data from SDK and broadcast to subscribers."""
    err = _validate_session(session_id)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={'error': 'invalid JSON body'},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={'error': 'request body must be a JSON object'},
        )

    _history.setdefault(session_id, []).append(body)
    if len(_history[session_id]) > _MAX_HISTORY:
        _history[session_id] = _history[session_id][-_MAX_HISTORY:]

    payload = json.dumps(body, ensure_ascii=False)
    dead: list[WebSocket] = []

    for ws in _subscribers.get(session_id, []):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)

    if dead:
        subs = _subscribers.get(session_id)
        if subs:
            _subscribers[session_id] = [ws for ws in subs if ws not in set(dead)]

    return JSONResponse(status_code=200, content={'status': 'ok'})


# ── WebSocket Subscribe (Server → Dashboard) ───────────────


@app.websocket('/ws/{session_id}')
async def ws_subscribe(websocket: WebSocket, session_id: str) -> None:
    """Dashboard connects here to receive real-time updates."""
    if not _SESSION_RE.match(session_id):
        await websocket.close(code=4000, reason='invalid session_id')
        return
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
    except Exception:
        logger.debug('WebSocket error for session %s', session_id)
    finally:
        subs = _subscribers.get(session_id, [])
        if websocket in subs:
            subs.remove(websocket)


# ── Health / Info ───────────────────────────────────────────


@app.get('/api/health')
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {'status': 'ok', 'sessions': len(_subscribers)}


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
    if not _SESSION_RE.match(session_id):
        return []
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
