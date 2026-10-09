"""Provider configuration: where to call, with which protocol, for which model.

One `ProviderConfig` is produced by the provider node and consumed by every call
node.  It is immutable and never serializes the API key (`public_dict`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .errors import EndpointError
from .secrets import consume_session_credential

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
PROTOCOL_AUTO = "auto"
PROTOCOL_OPENAI_IMAGES = "openai_images"
PROTOCOL_OPENAI_CHAT = "openai_chat"
PROTOCOL_OPENAI_RESPONSES = "openai_responses"
PROTOCOL_GEMINI = "gemini"
PROTOCOL_GROK = "grok"

API_PROTOCOLS: tuple[str, ...] = (
    PROTOCOL_AUTO,
    PROTOCOL_OPENAI_IMAGES,
    PROTOCOL_OPENAI_CHAT,
    PROTOCOL_OPENAI_RESPONSES,
    PROTOCOL_GEMINI,
    PROTOCOL_GROK,
)

PROTOCOL_LABELS = {
    PROTOCOL_AUTO: "auto",
    PROTOCOL_OPENAI_IMAGES: "openai_images (/images/generations)",
    PROTOCOL_OPENAI_CHAT: "openai_chat (/chat/completions)",
    PROTOCOL_OPENAI_RESPONSES: "openai_responses (/responses)",
    PROTOCOL_GEMINI: "gemini (/models/{model}:generateContent)",
    PROTOCOL_GROK: "grok (/images/* + /chat/completions)",
}

FAMILY_OPENAI = "openai"
FAMILY_GEMINI = "gemini"
FAMILY_GROK = "grok"

_PROTOCOL_FAMILY = {
    PROTOCOL_OPENAI_IMAGES: FAMILY_OPENAI,
    PROTOCOL_OPENAI_CHAT: FAMILY_OPENAI,
    PROTOCOL_OPENAI_RESPONSES: FAMILY_OPENAI,
    PROTOCOL_GEMINI: FAMILY_GEMINI,
    PROTOCOL_GROK: FAMILY_GROK,
}

# Friendly partner-node labels -> real API model ids.
GEMINI_MODEL_IDS = {
    "gemini 3.8 flash": "gemini-3.8-flash",
    "gemini 3.7 flash": "gemini-3.7-flash",
    "gemini 3.5 flash": "gemini-3.5-flash",
    "gemini 3.1 pro": "gemini-3.1-pro-preview",
    "gemini 3.1 flash-lite": "gemini-3.1-flash-lite-preview",
    "nano banana 2 (gemini 3.1 flash image)": "gemini-3.1-flash-image",
    "nano banana 2": "gemini-3.1-flash-image",
    "nano banana 2 lite": "gemini-3.1-flash-lite-image",
    "nano banana pro (google gemini image)": "gemini-3-pro-image",
    "nano banana (google gemini image)": "gemini-2.5-flash-image",
}
GROK_MODEL_IDS = {
    "grok-imagine-image-2.0": "grok-imagine-image-2.0",
    "grok-imagine-image-quality": "grok-imagine-image-quality",
    "grok-imagine-image-pro": "grok-imagine-image-pro",
    "grok-imagine-image": "grok-imagine-image",
}

_VERSION_SEGMENT_RE_PREFIXES = ("v1", "v2", "v3", "v1beta", "v2beta")


def normalize_model_id(value: str, protocol: str | None = None) -> str:
    """Map a friendly partner-node label to the real API id (idempotent)."""
    model = str(value or "").strip()
    if not model:
        return model
    key = " ".join(model.lower().split())
    if protocol == PROTOCOL_GEMINI:
        return GEMINI_MODEL_IDS.get(key, model)
    if protocol == PROTOCOL_GROK:
        return GROK_MODEL_IDS.get(key, model)
    return model


def infer_protocol(model: str) -> str:
    """Best-effort protocol guess for ``auto``."""
    lowered = " ".join(str(model or "").lower().split())
    if not lowered:
        return PROTOCOL_OPENAI_CHAT
    if "gpt-image" in lowered or "dall-e" in lowered:
        return PROTOCOL_OPENAI_IMAGES
    if lowered.startswith("grok-imagine-image") or "imagine-image" in lowered:
        return PROTOCOL_GROK
    if "grok" in lowered:
        return PROTOCOL_GROK
    if "gemini" in lowered or "nano-banana" in lowered or "nano banana" in lowered:
        return PROTOCOL_GEMINI
    return PROTOCOL_OPENAI_CHAT


def _path_has_version(path: str) -> bool:
    for segment in path.strip("/").split("/"):
        lowered = segment.lower()
        if lowered in _VERSION_SEGMENT_RE_PREFIXES:
            return True
    return False


def normalize_base_url(value: str, protocol: str = PROTOCOL_AUTO) -> str:
    """Trim the base URL and append the protocol's default version prefix.

    The silent-prefix trap: a relay mounted under ``/v1beta`` 404s every request
    if the client auto-fills ``/v1``.  We therefore only fill in a version
    segment when the user's URL has none at all.
    """
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parsed = urlsplit(raw)
    except ValueError as error:
        raise ValueError(f"Invalid endpoint URL: {value!r} ({error})") from error
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid endpoint URL: {value!r}")
    path = parsed.path.rstrip("/")
    if not _path_has_version(path):
        default_version = "v1beta" if protocol == PROTOCOL_GEMINI else "v1"
        path = f"{path}/{default_version}"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def join_url(base_url: str, suffix: str) -> str:
    base = str(base_url or "").rstrip("/")
    tail = str(suffix or "").lstrip("/")
    if not tail:
        return base
    return f"{base}/{tail}"


@dataclass(frozen=True)
class ProviderConfig:
    """Everything a call node needs to reach a provider."""

    base_url: str
    model: str
    api_protocol: str = PROTOCOL_AUTO
    api_key: str = field(default="", repr=False)
    timeout_sec: int = 300
    retries: int = 1
    extra_headers_json: str = field(default="", repr=False)
    extra_headers: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    encrypt_export: bool = True
    provider_name: str = "provider"
    config_error: str | None = None

    def __post_init__(self) -> None:
        if self.config_error:
            # A misconfigured provider node never fails the queue: the call
            # node turns this into a 1x1 image plus a response-log message.
            object.__setattr__(self, "model", str(self.model or "").strip())
            if self.api_protocol not in API_PROTOCOLS:
                object.__setattr__(self, "api_protocol", PROTOCOL_AUTO)
            return
        if self.api_protocol not in API_PROTOCOLS:
            raise ValueError(f"Unsupported api_protocol: {self.api_protocol}")
        object.__setattr__(self, "model", str(self.model or "").strip())
        object.__setattr__(self, "api_key", str(self.api_key or "").strip())
        object.__setattr__(self, "provider_name", str(self.provider_name or "provider").strip())
        object.__setattr__(self, "timeout_sec", max(5, int(self.timeout_sec or 300)))
        object.__setattr__(self, "retries", max(0, int(self.retries or 0)))
        object.__setattr__(self, "encrypt_export", bool(self.encrypt_export))

        protocol = self.api_protocol
        if protocol == PROTOCOL_AUTO:
            protocol = infer_protocol(self.model)
        object.__setattr__(self, "api_protocol", protocol)
        object.__setattr__(self, "model", normalize_model_id(self.model, protocol))

        headers = self._parse_headers(self.extra_headers_json)
        if not isinstance(self.extra_headers, dict) or not self.extra_headers:
            object.__setattr__(self, "extra_headers", headers)
        else:
            merged = dict(headers)
            merged.update({str(k): str(v) for k, v in self.extra_headers.items()})
            object.__setattr__(self, "extra_headers", merged)

    @staticmethod
    def _parse_headers(raw: str) -> dict[str, str]:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"extra_headers must be a JSON object: {error}") from error
        if not isinstance(parsed, dict):
            raise ValueError("extra_headers must be a JSON object.")
        return {str(key): str(value) for key, value in parsed.items()}

    # -- derived ----------------------------------------------------------
    @property
    def family(self) -> str:
        return _PROTOCOL_FAMILY.get(self.api_protocol, FAMILY_OPENAI)

    @property
    def resolved_model(self) -> str:
        return self.model

    def credential(self) -> str:
        return consume_session_credential(self.api_key)

    def validate(self) -> None:
        """Raise when the provider node is not usable (called by call nodes)."""
        if self.config_error:
            raise EndpointError(
                self.config_error,
                error_type="configuration",
                error_code="provider_invalid",
                phase="validate",
            )
        if not self.base_url:
            raise EndpointError(
                "endpoint is empty. Set the base URL on the provider node.",
                error_type="configuration",
                error_code="endpoint_missing",
                phase="validate",
            )
        if not self.model:
            raise EndpointError(
                "model is empty. Set the model on the provider node.",
                error_type="configuration",
                error_code="model_missing",
                phase="validate",
            )

    def public_dict(self) -> dict[str, Any]:
        """Diagnostics without the credential."""
        return {
            "provider_name": self.provider_name,
            "base_url": self.base_url,
            "model": self.model,
            "api_protocol": self.api_protocol,
            "family": self.family,
            "api_key_configured": bool(self.api_key),
            "encrypt_export": self.encrypt_export,
            "timeout_sec": self.timeout_sec,
            "retries": self.retries,
            "extra_header_names": sorted(self.extra_headers),
        }


def build_provider(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    api_protocol: str = PROTOCOL_AUTO,
    encrypt_export: bool = True,
    timeout_sec: int = 300,
    retries: int = 1,
    extra_headers: str = "",
    provider_name: str = "provider",
) -> ProviderConfig:
    """Validate + normalize the provider node widgets into a ProviderConfig.

    A configuration problem never raises out of the provider node: the node is
    built broken and every call node reports it as a per-item error log.
    """
    try:
        from .secrets import resolve_session_credential

        protocol = api_protocol if api_protocol in API_PROTOCOLS else PROTOCOL_AUTO
        resolved = protocol if protocol != PROTOCOL_AUTO else infer_protocol(model)
        # The prompt guard swaps the endpoint for an opaque handle (so the PNG
        # metadata never carries it); resolve it back.  consume=False: an
        # endpoint is not a secret and re-queueing must keep working.
        endpoint_value = resolve_session_credential(endpoint, consume=False)
        base_url = normalize_base_url(endpoint_value, resolved)
        config = ProviderConfig(
            base_url=base_url,
            model=model,
            api_protocol=api_protocol,
            api_key=api_key,
            timeout_sec=timeout_sec,
            retries=retries,
            extra_headers_json=extra_headers,
            encrypt_export=encrypt_export,
            provider_name=provider_name,
        )
    except Exception as error:  # noqa: BLE001 - misconfiguration is data, not a crash
        return ProviderConfig(
            base_url=str(endpoint or ""),
            model=str(model or ""),
            api_protocol=api_protocol if api_protocol in API_PROTOCOLS else PROTOCOL_AUTO,
            api_key="",
            timeout_sec=timeout_sec,
            retries=retries,
            encrypt_export=encrypt_export,
            provider_name=provider_name,
            config_error=f"Provider configuration error: {error}",
        )
    if not config.base_url:
        object.__setattr__(config, "config_error", "endpoint is empty. Set the base URL on the provider node.")
    elif not config.model:
        object.__setattr__(config, "config_error", "model is empty. Set the model on the provider node.")
    return config


def require_protocols(provider: ProviderConfig, allowed: tuple[str, ...], *, node_label: str) -> None:
    """Raise a clear EndpointError when a node is fed the wrong protocol."""
    provider.validate()
    if provider.api_protocol not in allowed:
        raise EndpointError(
            f"{node_label} needs a provider using one of {list(allowed)}, but the connected "
            f"provider resolves to protocol '{provider.api_protocol}'. Set api_protocol explicitly "
            f"on the provider node (or use a matching node).",
            error_type="configuration",
            error_code="protocol_mismatch",
            phase="validate",
            protocol=provider.api_protocol,
        )



def require_family(provider: ProviderConfig, family: str, *, node_label: str) -> None:
    """Raise a clear EndpointError when a node is fed another family's provider."""
    if provider.family != family:
        raise EndpointError(
            f"{node_label} requires a {family} provider, but the connected provider "
            f"resolves to protocol '{provider.api_protocol}' (family '{provider.family}'). "
            f"Set api_protocol explicitly on the provider node.",
            error_type="configuration",
            error_code="protocol_mismatch",
            phase="validate",
            protocol=provider.api_protocol,
        )
