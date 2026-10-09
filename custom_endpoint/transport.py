"""Minimal stdlib HTTP client (urllib) with retries and multipart support.

No third-party dependency: ``requests`` may or may not be importable depending
on the ComfyUI install, and the whole surface we need is small.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .errors import EndpointError

DEFAULT_USER_AGENT = "ComfyUI-Custom-Endpoint-Nodes/1.0"
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 522, 524})


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    url: str
    method: str
    latency_ms: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        try:
            return self.body.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - decode never raises with replace
            return ""

    def json(self) -> Any:
        text = self.text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise EndpointError(
                f"Provider returned a non-JSON body: {exc}",
                status=self.status,
                error_type="protocol",
                error_code="invalid_json",
                phase="decode",
                endpoint=self.url,
                response_excerpt=text[:800],
                latency_ms=self.latency_ms,
            ) from exc

    def header(self, name: str) -> str | None:
        target = name.lower()
        for key, value in self.headers.items():
            if key.lower() == target:
                return value
        return None

    def excerpt(self, limit: int = 800) -> str:
        return self.text[:limit]


def build_multipart(
    fields: dict[str, Any],
    files: Iterable[tuple[str, str, bytes, str]],
) -> tuple[str, bytes]:
    """Return ``(content_type, body)`` for a multipart/form-data request.

    ``files`` items are ``(field_name, filename, content, content_type)``.
    """
    boundary = "----CustomEndpoint" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        if value is None:
            continue
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        if isinstance(value, (dict, list)):
            chunks.append(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        elif isinstance(value, bool):
            chunks.append(("true" if value else "false").encode("utf-8"))
        else:
            chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    for field_name, filename, content, content_type in files:
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)


class HttpClient:
    """Sends one request and returns an HttpResponse; raises only on transport failure."""

    def __init__(
        self,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleeper or time.sleep
        self._user_agent = user_agent

    # -- public API -------------------------------------------------------
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: Any = None,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout: float = 300,
        retries: int = 0,
        backoff: float = 1.5,
        phase: str = "request",
    ) -> HttpResponse:
        request_headers = {"User-Agent": self._user_agent, "Accept": "application/json"}
        if headers:
            request_headers.update({str(k): str(v) for k, v in headers.items()})

        payload = body
        if json_body is not None:
            payload = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        if content_type:
            request_headers["Content-Type"] = content_type
        if payload is not None:
            request_headers.setdefault("Content-Length", str(len(payload)))

        attempts = max(0, int(retries)) + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            try:
                response = self._send(
                    method, url, request_headers, payload, timeout, started
                )
            except EndpointError:
                raise
            except Exception as error:  # network layer
                last_error = error
                if attempt < attempts:
                    self._sleep(backoff ** (attempt - 1))
                    continue
                raise EndpointError(
                    f"{method} {url} failed: {error}",
                    error_type="transport",
                    error_code=type(error).__name__,
                    phase=phase,
                    endpoint=url,
                    retryable=True,
                    attempts=attempt,
                ) from error

            if (
                response.status in RETRY_STATUSES
                and attempt < attempts
                and response.status != 400
            ):
                self._sleep(backoff ** (attempt - 1))
                continue
            return response

        # Unreachable: the loop either returns or raises.
        raise EndpointError(  # pragma: no cover
            f"{method} {url} failed: {last_error}",
            error_type="transport",
            phase=phase,
            endpoint=url,
            attempts=attempts,
        )

    def get(self, url: str, **kwargs: Any) -> HttpResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> HttpResponse:
        return self.request("POST", url, **kwargs)

    # -- internals --------------------------------------------------------
    def _send(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: bytes | None,
        timeout: float,
        started: float,
    ) -> HttpResponse:
        request = urllib.request.Request(
            url, data=payload, headers=headers, method=method.upper()
        )
        try:
            with self._opener(request, timeout=timeout) as raw:
                body = raw.read()
                status = int(getattr(raw, "status", None) or raw.getcode())
                response_headers = {k: v for k, v in raw.headers.items()}
        except urllib.error.HTTPError as error:
            body = error.read() if hasattr(error, "read") else b""
            status = int(error.code)
            response_headers = {k: v for k, v in (error.headers or {}).items()}
        except urllib.error.URLError as error:
            raise socket.error(str(error.reason)) from error
        return HttpResponse(
            status=status,
            headers=response_headers,
            body=body or b"",
            url=url,
            method=method.upper(),
            latency_ms=int((time.monotonic() - started) * 1000),
        )


def auth_headers(provider: Any) -> dict[str, str]:
    """Auth header for a provider.

    Gemini's native API wants ``x-goog-api-key``; OpenAI/Grok want Bearer.
    Compatible relays accept both, so we send the native one and fall back to
    Bearer only when the caller asks for it explicitly.
    """
    key = provider.credential()
    if not key:
        raise EndpointError(
            "API key is empty. Enter it on the provider node.",
            error_type="authentication",
            error_code="credential_missing",
            phase="validate",
        )
    headers: dict[str, str] = {}
    if provider.api_protocol == "gemini":
        headers["x-goog-api-key"] = key
        headers["Authorization"] = f"Bearer {key}"
    else:
        headers["Authorization"] = f"Bearer {key}"
    headers.update(provider.extra_headers)
    return headers
