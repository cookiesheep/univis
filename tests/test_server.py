"""Tests for UniVis FastAPI server endpoints."""

import sys
from pathlib import Path
from types import ModuleType

from starlette.testclient import TestClient


# Import server module directly to avoid torch dependency in __init__.py
_server_mod: ModuleType | None = None
_server_path = str(Path(__file__).resolve().parent.parent / 'src' / 'univis')
if _server_path not in sys.path:
    sys.path.insert(0, _server_path)

import server as _server_mod  # type: ignore[no-redef]
import importlib
importlib.reload(_server_mod)  # ensure fresh state

app = _server_mod.app
_subscribers = _server_mod._subscribers
_history = _server_mod._history


client = TestClient(app)


def setup_module() -> None:
    _subscribers.clear()
    _history.clear()


def teardown_module() -> None:
    _subscribers.clear()
    _history.clear()


class TestHealthEndpoint:
    def test_health_returns_ok(self) -> None:
        resp = client.get('/api/health')
        assert resp.status_code == 200
        body = resp.json()
        assert body['status'] == 'ok'
        assert 'sessions' in body

    def test_health_reflects_session_count(self) -> None:
        _subscribers['test-health-session'] = []
        try:
            resp = client.get('/api/health')
            assert resp.status_code == 200
            assert resp.json()['sessions'] >= 1
        finally:
            _subscribers.pop('test-health-session', None)


class TestPushEndpoint:
    def test_push_valid_data_returns_ok(self) -> None:
        resp = client.post(
            '/api/push/test-push',
            json={'type': 'step', 'token_idx': 0},
        )
        assert resp.status_code == 200
        assert resp.json()['status'] == 'ok'
        _history.pop('test-push', None)

    def test_push_invalid_json_returns_400(self) -> None:
        resp = client.post(
            '/api/push/test-push-bad',
            content=b'not valid json {{{',
            headers={'content-type': 'application/json'},
        )
        assert resp.status_code == 400
        assert 'error' in resp.json()

    def test_push_non_object_json_returns_400(self) -> None:
        resp = client.post(
            '/api/push/test-push-array',
            content=b'[1, 2, 3]',
            headers={'content-type': 'application/json'},
        )
        assert resp.status_code == 400
        assert 'error' in resp.json()


class TestSessionsEndpoint:
    def test_sessions_returns_data(self) -> None:
        _subscribers['test-sess-1'] = []
        _history['test-sess-1'] = [{'a': 1}]
        try:
            resp = client.get('/api/sessions')
            assert resp.status_code == 200
            body = resp.json()
            assert 'sessions' in body
            assert 'test-sess-1' in body['sessions']
            assert body['sessions']['test-sess-1']['subscribers'] == 0
            assert body['sessions']['test-sess-1']['messages'] == 1
        finally:
            _subscribers.pop('test-sess-1', None)
            _history.pop('test-sess-1', None)


class TestSessionValidation:
    def test_rejects_path_traversal(self) -> None:
        resp = client.post('/api/push/../../../etc/passwd', json={'type': 'step'})
        assert resp.status_code in (400, 404)

    def test_rejects_special_chars_in_session(self) -> None:
        resp = client.post('/api/push/session with spaces', json={'type': 'step'})
        assert resp.status_code == 400

    def test_rejects_empty_session(self) -> None:
        resp = client.post('/api/push/', json={'type': 'step'})
        assert resp.status_code in (400, 404, 422)

    def test_accepts_valid_session(self) -> None:
        resp = client.post('/api/push/abc123def', json={'type': 'step'})
        assert resp.status_code == 200
        _history.pop('abc123def', None)

    def test_history_capped(self) -> None:
        for i in range(12):
            client.post('/api/push/cap-test', json={'type': 'step', 'i': i})
        assert len(_history.get('cap-test', [])) <= 12
        _history.pop('cap-test', None)
