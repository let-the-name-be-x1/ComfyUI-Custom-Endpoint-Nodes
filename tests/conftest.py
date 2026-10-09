import base64
import io
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

import ce_nodes as nodes  # noqa: E402
from custom_endpoint import api_gemini, api_grok, api_openai, images, transport  # noqa: E402


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------
def png_bytes(width: int = 8, height: int = 8, color=(255, 0, 0)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def png_b64(width: int = 8, height: int = 8) -> str:
    return base64.b64encode(png_bytes(width, height)).decode("ascii")


def image_tensor(batch: int = 1, height: int = 8, width: int = 8, value: float = 0.5):
    return torch.full((batch, height, width, 3), float(value), dtype=torch.float32)


def mask_tensor(batch: int = 1, height: int = 8, width: int = 8, value: float = 0.0):
    return torch.full((batch, height, width), float(value), dtype=torch.float32)


# ---------------------------------------------------------------------------
# fake transport
# ---------------------------------------------------------------------------
class RawResponse:
    """Mimics the object urllib returns from an opener."""

    def __init__(self, status: int = 200, body: bytes | str = b"", headers=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.status = int(status)
        self._body = body
        self.headers = dict(headers or {})

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def json_response(payload, status: int = 200, headers=None):
    return RawResponse(status=status, body=json.dumps(payload), headers=headers)


class QueueOpener:
    """Replays queued responses (or exceptions) and records every request."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = []

    def __call__(self, request, timeout=None):
        self.calls.append(request)
        if not self.queue:
            raise AssertionError(f"unexpected extra request: {request.full_url}")
        item = self.queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    @property
    def urls(self):
        return [call.full_url for call in self.calls]


class HandlerOpener:
    """Calls a handler(url, body) -> RawResponse, for tests with many requests."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def __call__(self, request, timeout=None):
        self.calls.append(request)
        return self.handler(request)


@pytest.fixture
def install_client(monkeypatch):
    """Point the node's HTTP client at a fake opener."""

    def _install(opener):
        client = transport.HttpClient(opener=opener, sleeper=lambda _: None)
        monkeypatch.setattr(nodes, "_new_client", lambda: client)
        return client

    return _install


@pytest.fixture(autouse=True)
def _clear_credentials():
    from custom_endpoint import secrets

    secrets.clear_session_credentials()
    yield
    secrets.clear_session_credentials()


def load_package():
    """Load the package exactly the way ComfyUI does (dir name, __init__.py).

    NOTE: this produces a *second* module tree (`ComfyUI-Custom-Endpoint-Nodes.nodes`).
    Use it only for registration metadata; behavioural tests import the top-level
    `nodes` module from conftest so `isinstance` checks stay inside one tree.
    """
    import importlib.util

    name = "ComfyUI-Custom-Endpoint-Nodes"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "__init__.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def package():
    return load_package()
