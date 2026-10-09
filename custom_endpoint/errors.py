"""Error type used across the package.

Deliberately not an exception the caller is expected to catch: every call node
converts an ``EndpointError`` into a 1x1 image plus a response-log string, so
the ComfyUI queue never fails because a provider rejected a request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .redact import sanitize


@dataclass
class EndpointError(RuntimeError):
    """A provider/transport failure with structured, log-ready detail."""

    message: str = ""
    status: int | None = None
    request_id: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    phase: str | None = None
    protocol: str | None = None
    endpoint: str | None = None
    retryable: bool = False
    attempts: int = 1
    latency_ms: int | None = None
    response_excerpt: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # RuntimeError wants a message
        RuntimeError.__init__(self, self.message)

    def report(self, known_secrets: tuple[str, ...] = ()) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "message": self.message,
            "http_status": self.status,
            "request_id": self.request_id,
            "error_type": self.error_type,
            "error_code": self.error_code,
            "phase": self.phase,
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "retryable": self.retryable,
            "attempts": self.attempts,
            "latency_ms": self.latency_ms,
            "response_excerpt": self.response_excerpt,
        }
        if self.details:
            payload["details"] = self.details
        return sanitize(payload, known_secrets)

    def __str__(self) -> str:
        return json.dumps(self.report(), ensure_ascii=False, separators=(",", ":"))


def report_json(report: dict[str, Any], known_secrets: tuple[str, ...] = ()) -> str:
    """Serialize a log/report dict with secrets removed and stable key order."""
    return json.dumps(
        sanitize(report, known_secrets),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def error_json(
    error: Exception,
    *,
    known_secrets: tuple[str, ...] = (),
    phase: str | None = None,
    protocol: str | None = None,
    endpoint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Turn any exception into one log-ready JSON string."""
    if isinstance(error, EndpointError):
        report = error.report(known_secrets)
        if phase and not report.get("phase"):
            report["phase"] = phase
        if protocol and not report.get("protocol"):
            report["protocol"] = protocol
        if endpoint and not report.get("endpoint"):
            report["endpoint"] = endpoint
    else:
        report = sanitize(
            {
                "ok": False,
                "message": str(error) or error.__class__.__name__,
                "error_type": error.__class__.__name__,
                "phase": phase,
                "protocol": protocol,
                "endpoint": endpoint,
            },
            known_secrets,
        )
    if extra:
        report.update(sanitize(extra, known_secrets))
    return json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
