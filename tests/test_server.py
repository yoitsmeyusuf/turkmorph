"""Exercise HTTP input validation without opening a port or loading model files."""
import importlib.util
import io
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def app(monkeypatch):
    import turkmorph.presets

    monkeypatch.setattr(turkmorph.presets, "full_tokenizer", lambda: None)
    monkeypatch.setitem(sys.modules, "turkish_tokenizer", types.ModuleType("turkish_tokenizer"))
    monkeypatch.setitem(sys.modules, "turkmorph.context", types.ModuleType("turkmorph.context"))
    spec = importlib.util.spec_from_file_location(
        "review_server", Path(__file__).resolve().parents[1] / "server" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("body,length,code", [
    (b"{", "1", 400),
    (b"\xff", "1", 400),
    (b"[]", "2", 400),
    (b'{"text":42}', "11", 400),
    (b"", "-1", 400),
    (b"", "bad", 400),
    (b"", "256001", 413),
])
def test_invalid_request(app, body, length, code):
    handler = object.__new__(app.Handler)
    handler.path = "/api/tokenize"
    handler.headers = {"Content-Length": length}
    handler.rfile = io.BytesIO(body)
    responses = []
    handler._send = lambda status, *args: responses.append(status)
    handler.do_POST()
    assert responses == [code]
    assert handler.close_connection
    if code == 413:
        assert handler.rfile.tell() == 0


def test_valid_request(app):
    app.TOK = types.SimpleNamespace(vocab=[])
    app.turkmorph = lambda text, context: {"decoded": text}
    app.turkish_tokenizer = lambda text: None
    handler = object.__new__(app.Handler)
    handler.path = "/api/tokenize"
    body = b'{"text":"hello"}'
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    responses = []
    handler._send = lambda status, *args: responses.append((status, args[0]))
    handler.do_POST()
    assert responses[0][0] == 200
    assert '"decoded": "hello"' in responses[0][1]
