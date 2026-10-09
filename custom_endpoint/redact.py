"""Secret redaction helpers.

Every log line this package emits (request log, response log, error report)
passes through these functions.  The rule is: never let a credential, an
Authorization header or an inline base64 image body reach a ComfyUI widget,
the queue history or a PNG metadata block.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

SENSITIVE_KEY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "authorization",
        "credential",
        "password",
        "secret",
        "token",
        "access_token",
        "x-api-key",
        "x-goog-api-key",
    }
)

REDACTED = "[REDACTED]"

_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_KEYISH_RE = re.compile(r"\b(?:sk|sess|key|xai|AIza)[-_][A-Za-z0-9._-]{8,}\b", re.IGNORECASE)
_DATA_URL_RE = re.compile(r"data:(?:image|audio|video)/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", re.IGNORECASE)
_LONG_B64_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{256,}={0,2}(?![A-Za-z0-9+/=])")
_HTTP_URL_RE = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)


def _mask_url_query(match: "re.Match[str]") -> str:
    """Replace every query value of an http(s) URL with [REDACTED]."""
    raw = match.group(0)
    trailing = ""
    while raw and raw[-1] in ".,;!?)]":
        trailing = raw[-1] + trailing
        raw = raw[:-1]
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw + trailing
    if not parsed.query:
        return raw + trailing
    masked = "&".join(
        f"{component.split('=', 1)[0]}={REDACTED}" for component in parsed.query.split("&")
    )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, masked, "")) + trailing


def redact_text(value: Any, known_secrets: tuple[str, ...] = ()) -> str:
    """Redact secrets from one string."""
    text = str(value)
    for secret in known_secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, REDACTED)
    text = _BEARER_RE.sub(f"Bearer {REDACTED}", text)
    text = _KEYISH_RE.sub(REDACTED, text)
    text = _DATA_URL_RE.sub(f"data:[binary];base64,{REDACTED}", text)
    text = _HTTP_URL_RE.sub(_mask_url_query, text)
    text = _LONG_B64_RE.sub("[REDACTED_BASE64]", text)
    return text


def redact_url(value: Any, known_secrets: tuple[str, ...] = ()) -> str:
    """Redact a URL: keep scheme/host/path, drop or mask the query values."""
    raw = str(value)
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return redact_text(raw, known_secrets)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return redact_text(raw, known_secrets)
    query = ""
    if parsed.query:
        query = "&".join(
            f"{redact_text(component.split('=', 1)[0], known_secrets)}={REDACTED}"
            for component in parsed.query.split("&")
        )
    return redact_text(
        urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, "")), known_secrets
    )


def redact_headers(headers: Any, known_secrets: tuple[str, ...] = ()) -> dict[str, str]:
    """Return the header map with credential-bearing values masked."""
    cleaned: dict[str, str] = {}
    if not isinstance(headers, dict):
        return cleaned
    for key, value in headers.items():
        name = str(key)
        if name.lower() in SENSITIVE_KEY_NAMES:
            cleaned[name] = REDACTED
        else:
            cleaned[name] = redact_text(value, known_secrets)
    return cleaned


def sanitize(value: Any, known_secrets: tuple[str, ...] = ()) -> Any:
    """Recursively replace sensitive values inside a JSON-able structure."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEY_NAMES:
                cleaned[str(key)] = REDACTED
            else:
                cleaned[str(key)] = sanitize(item, known_secrets)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize(item, known_secrets) for item in value]
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, str):
        return redact_text(value, known_secrets)
    return value
