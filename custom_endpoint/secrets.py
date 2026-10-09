"""In-memory API-key handles.

The API key widget never travels to the server inside the queued prompt.  The
frontend POSTs the key once to ``/custom-endpoint/session-credential`` and puts
the returned handle in the prompt instead; the provider node resolves the handle
back to the key at execution time.

With the "encrypt export" switch on, nothing else keeps the key: the exported
workflow file holds no endpoint and no key, so a reload requires re-entering it.
"""

from __future__ import annotations

import re
import secrets
import threading
import time

HANDLE_PREFIX = "custom-endpoint-session:"
SESSION_TTL_SECONDS = 30 * 60
MAX_SESSION_CREDENTIALS = 256

_LOCK = threading.Lock()
_CREDENTIALS: dict[str, tuple[float, str]] = {}
_HANDLE_PATTERN = re.compile(rf"^{re.escape(HANDLE_PREFIX)}[A-Za-z0-9_-]{{43}}$")


def _purge_expired(now: float) -> None:
    expired = [
        handle for handle, (expires_at, _secret) in _CREDENTIALS.items() if expires_at <= now
    ]
    for handle in expired:
        _CREDENTIALS.pop(handle, None)


def issue_session_credential(api_key: str, *, now: float | None = None) -> str:
    """Store an API key in process memory and return a non-secret handle."""
    secret = str(api_key or "").strip()
    if not secret:
        raise ValueError("api_key cannot be empty.")
    if len(secret) > 16 * 1024:
        raise ValueError("api_key is too large.")

    current = time.monotonic() if now is None else float(now)
    handle = HANDLE_PREFIX + secrets.token_urlsafe(32)
    with _LOCK:
        _purge_expired(current)
        while len(_CREDENTIALS) >= MAX_SESSION_CREDENTIALS:
            oldest = min(_CREDENTIALS, key=lambda item: _CREDENTIALS[item][0])
            _CREDENTIALS.pop(oldest, None)
        _CREDENTIALS[handle] = (current + SESSION_TTL_SECONDS, secret)
    return handle


def is_session_handle(value: str) -> bool:
    return _HANDLE_PATTERN.fullmatch(str(value or "").strip()) is not None


def consume_session_credential(value: str, *, now: float | None = None) -> str:
    """Return the plaintext key for a handle; plain keys pass through unchanged.

    A plain key is accepted so a headless API client (or a workflow restored
    from a file where the user typed the key back in) still works; the prompt
    guard is what keeps a plain key out of the queue history.
    """
    return resolve_session_credential(value, consume=True, now=now)


def resolve_session_credential(
    value: str, *, consume: bool = True, now: float | None = None
) -> str:
    """Resolve a handle to its plaintext value.

    ``consume=True`` (API keys) removes it so a handle leaked into the history
    cannot be replayed.  ``consume=False`` (endpoints) keeps it for the TTL,
    because an endpoint is not a secret and re-queueing must keep working.
    """
    candidate = str(value or "").strip()
    if not candidate.startswith(HANDLE_PREFIX):
        return candidate

    current = time.monotonic() if now is None else float(now)
    with _LOCK:
        _purge_expired(current)
        entry = _CREDENTIALS.get(candidate)
        if consume:
            _CREDENTIALS.pop(candidate, None)
    if entry is None:
        raise ValueError(
            "The temporary API credential expired or is unavailable. "
            "Re-enter the endpoint/API Key in the provider node and queue again."
        )
    return entry[1]


def clear_session_credentials() -> None:
    """Clear volatile credentials (shutdown paths and tests)."""
    with _LOCK:
        _CREDENTIALS.clear()
