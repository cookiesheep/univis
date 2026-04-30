"""Transport layer: send metrics to file or WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger('univis')


class Transport(ABC):
    """Abstract transport for metric messages."""

    @abstractmethod
    def send(self, message: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class FileTransport(Transport):
    """Append JSONL messages to a local file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, 'a', encoding='utf-8')  # noqa: SIM115

    def send(self, message: dict[str, Any]) -> None:
        self._file.write(json.dumps(message, ensure_ascii=False) + '\n')
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    @property
    def path(self) -> Path:
        return self._path


class WebSocketTransport(Transport):
    """Send messages via WebSocket.

    Note: This is a placeholder. The actual WebSocket connection is managed
    by the server module. Messages are silently dropped until connect() is called.
    """

    def __init__(self, uri: str = 'ws://127.0.0.1:8765') -> None:
        self._uri = uri
        self._ws: Any = None
        self._connected = False

    def connect(self, ws: Any) -> None:
        """Set the active WebSocket connection."""
        self._ws = ws
        self._connected = True

    def send(self, message: dict[str, Any]) -> None:
        if self._ws is None:
            return  # Not connected yet, message dropped silently
        payload = json.dumps(message)

        async def _send() -> None:
            await self._ws.send(payload)

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(_send())
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        except RuntimeError:
            asyncio.run(_send())

    def close(self) -> None:
        if self._ws is not None:

            async def _close() -> None:
                await self._ws.close()

            try:
                asyncio.run(_close())
            except Exception:
                pass
        self._connected = False


class HttpPushTransport(Transport):
    """Push data to UniVis server via HTTP POST (sync, zero extra deps)."""

    def __init__(self, base_url: str, session_id: str) -> None:
        self._url = f'{base_url}/api/push/{session_id}'

    def send(self, message: dict[str, Any]) -> None:
        import urllib.request
        data = json.dumps(message, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={'Content-Type': 'application/json'},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            logger.debug('Failed to push data to server', exc_info=True)

    def close(self) -> None:
        pass


class MultiTransport(Transport):
    """Fan-out to multiple transports."""

    def __init__(self, transports: list[Transport]) -> None:
        self._transports = transports

    def send(self, message: dict[str, Any]) -> None:
        for t in self._transports:
            t.send(message)

    def close(self) -> None:
        for t in self._transports:
            t.close()
