"""Shared request/response shapes for the three provider families."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import EndpointError
from .images import b64_to_bytes, bytes_look_like_image
from .transport import HttpResponse

IMAGE_MIME_BY_FORMAT = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}


@dataclass
class PreparedRequest:
    """One concrete HTTP call, ready to be sent and logged."""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    json_body: Any = None
    body: bytes | None = None
    content_type: str | None = None
    operation: str = ""
    image_summary: dict[str, Any] = field(default_factory=dict)
    log_body: Any = None  # override for the log when json_body is not what is sent

    def effective_log_body(self) -> Any:
        return self.log_body if self.log_body is not None else self.json_body


@dataclass
class ParsedResponse:
    """Provider response, decoded into the shapes a node needs."""

    images: list[bytes] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    thought_images: list[bytes] = field(default_factory=list)
    text: str = ""
    revised_prompts: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def has_images(self) -> bool:
        return bool(self.images)


def extract_error_message(payload: Any, fallback: str) -> tuple[str, str | None, str | None]:
    """Pull (message, error_type, error_code) out of any provider error body."""
    if not isinstance(payload, dict):
        return fallback, None, None
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("detail") or fallback
        code = error.get("code") or error.get("status")
        return str(message), str(error.get("type") or "") or None, _stringify(code)
    if isinstance(error, str):
        return error, None, None
    for key in ("message", "detail", "msg", "error_description"):
        if isinstance(payload.get(key), str):
            return payload[key], None, _stringify(payload.get("code"))
    if isinstance(payload.get("code"), (str, int)) and payload.get("message") is None:
        return fallback, None, str(payload["code"])
    return fallback, None, None


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def check_response(
    response: HttpResponse,
    *,
    protocol: str,
    phase: str,
    provider: Any = None,
) -> None:
    """Raise an EndpointError when the provider answered with an error status."""
    if response.ok:
        return
    payload = None
    try:
        payload = response.json()
    except EndpointError:
        payload = None
    message, error_type, error_code = extract_error_message(
        payload, f"HTTP {response.status} from the provider."
    )
    raise EndpointError(
        message,
        status=response.status,
        error_type=error_type or "provider",
        error_code=error_code,
        phase=phase,
        protocol=protocol,
        endpoint=response.url,
        retryable=response.status in {408, 429, 500, 502, 503, 504},
        latency_ms=response.latency_ms,
        response_excerpt=response.excerpt(1200),
        details={"provider": provider.public_dict()} if provider is not None else {},
    )


def download_bytes(client: Any, url: str, *, timeout: float, phase: str, protocol: str) -> bytes:
    """Fetch a result image URL.  Never forwards the API Authorization header."""
    response = client.request("GET", url, headers={}, timeout=timeout, phase=phase)
    if not response.ok:
        raise EndpointError(
            f"Could not download the result image (HTTP {response.status}).",
            status=response.status,
            error_type="download",
            phase=phase,
            protocol=protocol,
            endpoint=url,
            response_excerpt=response.excerpt(400),
        )
    return response.body


def decode_b64_image(data: str, *, phase: str, protocol: str) -> bytes:
    try:
        decoded = b64_to_bytes(data)
    except Exception as error:
        raise EndpointError(
            f"Provider returned an undecodable base64 image: {error}",
            error_type="decode",
            phase=phase,
            protocol=protocol,
        ) from error
    if not bytes_look_like_image(decoded):
        raise EndpointError(
            "Provider returned base64 data that is not a recognizable image.",
            error_type="decode",
            phase=phase,
            protocol=protocol,
            response_excerpt=decoded[:32].hex(),
        )
    return decoded


def placeholder_text(parsed: ParsedResponse) -> str:
    return parsed.text or ""
