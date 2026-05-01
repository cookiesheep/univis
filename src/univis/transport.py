"""Transport layer: send metrics to file or push to server."""

from __future__ import annotations

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
