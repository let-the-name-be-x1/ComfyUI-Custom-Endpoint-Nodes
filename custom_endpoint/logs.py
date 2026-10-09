"""The two log outputs of every call node.

``request_log``  — exactly what was sent: method, url, headers (credentials
                   masked), body with every image/binary field replaced by a
                   size summary ("image/png 1024x1024, 512.3 KB").
``response_log`` — exactly what came back: status, latency, response headers of
                   interest, the raw body with binary blobs summarized, and the
                   parsed result (text / revised prompt / block reasons).

Both are plain JSON strings so they stay readable in a Show Text node and easy
to grep in the ComfyUI history.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .errors import report_json
from .images import describe_bytes
from .redact import redact_headers, redact_text, redact_url, sanitize

_B64_FIELD_NAMES = frozenset(
    {
        "b64_json",
        "data",
        "inlineData",
        "inline_data",
        "file_data",
        "fileData",
        "image_url",
        "video",
        "audio",
        "audio_data",
        "base64",
        "bytes",
    }
)
_DATA_URL_RE = re.compile(r"^data:([a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)
_LONG_B64_RE = re.compile(r"^[A-Za-z0-9+/=\s]{128,}$")


def summarize_binary(raw: str) -> dict[str, Any] | None:
    """If ``raw`` is a data URL or a long base64 blob, describe it instead."""
    match = _DATA_URL_RE.match(raw.strip())
    if match:
        mime, payload = match.group(1), match.group(2)
        try:
            import base64

            decoded = base64.b64decode(payload, validate=False)
            return {
                "kind": "data_url",
                "mime": mime,
                "summary": describe_bytes(decoded),
                "bytes": len(decoded),
            }
        except Exception:
            return {"kind": "data_url", "mime": mime, "summary": f"<{len(payload)} b64 chars>"}
    if _LONG_B64_RE.match(raw) and len(raw.strip()) >= 128:
        return {"kind": "base64", "summary": f"<{len(raw.strip())} b64 chars>"}
    return None


def _substitute(value: Any) -> Any:
    """Replace binary payloads with placeholders, recursively."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, str):
                if key in _B64_FIELD_NAMES:
                    summary = summarize_binary(item)
                    result[key] = summary or {
                        "kind": "base64",
                        "summary": f"<{len(item)} b64 chars>",
                    }
                    continue
                summary = summarize_binary(item)
                if summary is not None and summary["kind"] == "data_url":
                    result[key] = summary
                    continue
                result[key] = redact_text(item)
            else:
                result[key] = _substitute(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_substitute(item) for item in value]
    if isinstance(value, bytes):
        return describe_bytes(value)
    if isinstance(value, str):
        summary = summarize_binary(value)
        return summary if summary is not None else redact_text(value)
    return value


def request_log_json(
    *,
    provider: Any,
    method: str,
    url: str,
    headers: dict[str, str],
    body: Any = None,
    image_summary: dict[str, Any] | None = None,
    operation: str | None = None,
    item_index: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Build the request log string."""
    secrets = (provider.credential() if getattr(provider, "api_key", "") else "",)
    payload: dict[str, Any] = {
        "ok": True,
        "phase": "request",
        "method": method.upper(),
        "url": redact_url(url, secrets),
        "headers": redact_headers(headers, secrets),
        "provider": provider.public_dict(),
    }
    if operation:
        payload["operation"] = operation
    if item_index is not None:
        payload["item_index"] = item_index
    if body is not None:
        payload["body"] = _substitute(body)
    if image_summary:
        payload["images"] = _substitute(image_summary)
    if extra:
        payload["extra"] = _substitute(extra)
    return report_json(payload, secrets)


def response_log_json(
    *,
    provider: Any,
    response: Any,
    parsed: dict[str, Any] | None = None,
    item_index: int | None = None,
    extra: dict[str, Any] | None = None,
    ok: bool | None = None,
) -> str:
    """Build the response log string from an HttpResponse (or None)."""
    secrets = (provider.credential() if getattr(provider, "api_key", "") else "",)
    payload: dict[str, Any] = {"ok": True, "phase": "response"}
    if ok is not None:
        payload["ok"] = bool(ok)
    if item_index is not None:
        payload["item_index"] = item_index
    if response is not None:
        payload["http_status"] = response.status
        payload["latency_ms"] = response.latency_ms
        payload["url"] = redact_url(response.url, secrets)
        request_id = response.header("x-request-id") or response.header("request-id")
        if request_id:
            payload["request_id"] = redact_text(request_id, secrets)
        raw = response.text
        summary = summarize_binary(raw)
        if summary is not None:
            payload["body"] = summary
        elif raw.strip():
            body = response.json() if _looks_like_json(raw) else raw
            payload["body"] = _substitute(body) if not isinstance(body, str) else redact_text(body)
        if not response.ok:
            payload["ok"] = False
    if parsed:
        payload["parsed"] = sanitize(parsed, secrets)
    if extra:
        payload["extra"] = sanitize(extra, secrets)
    return report_json(payload, secrets)


def _looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def image_summary(names_and_frames: Iterable[tuple[str, list[bytes]]]) -> dict[str, Any]:
    """Summarize the images attached to a request."""
    result: dict[str, Any] = {}
    for name, pngs in names_and_frames:
        result[name] = [describe_bytes(png) for png in pngs]
    return result


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
