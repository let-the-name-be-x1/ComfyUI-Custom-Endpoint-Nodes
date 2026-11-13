"""Qwen-Image wire protocol, in the two shapes it is actually served in.

``DIALECT_NATIVE`` (protocol ``qwen``) — Alibaba DashScope, as the ComfyUI
partner nodes use it::

    POST {base}/services/aigc/multimodal-generation/generation
    {"model": "...",
     "input": {"messages": [{"role": "user",
                            "content": [{"image": "data:image/png;base64,..."},
                                        {"text": "把 @Image1 换成蓝色"}]}]},
     "parameters": {"size": "1024*1024", "n": 1, "seed": 42,
                    "prompt_extend": true, "watermark": false,
                    "negative_prompt": "..."}}          # only when non-empty

``DIALECT_OPENAI_IMAGES`` (protocol ``qwen_images``) — what relays (one-api /
new-api / LiteLLM style) expose, because they never route the native path
(``404 Invalid URL (POST /v1/services/aigc/...)``)::

    POST {base}/images/edits          (generation: /images/generations)
    {"model": "...", "prompt": "...", "n": 1, "watermark": false,
     "prompt_extend": true, "size": "1024*1024", "seed": 42,
     "negative_prompt": "...", "image": "data:image/png;base64,..."}

``image`` is a plain string for one reference and a list for 2-3 (verified
against a live relay).  The relay answers OpenAI-style
``{"data": [{"url": "https://...signed..."}], "created": ...}``, so downloads
must not carry the Authorization header.

Both dialects share the Qwen rules ported from the partner node:

* pixel area window, aspect ratio 1:8..8:1, and ``size`` as ``"W*H"``;
* ``@Image1``-style prompt references rewritten to ``Image 1``;
* ``n`` in 1..6, ``seed`` in 0..2147483647, ``prompt_extend`` default true.

Two limits differ between the surfaces: the official partner-node path allows
2560x2560 (6,553,600 px) while the relay upstream observed in practice caps at
2048x2048 (4,194,304 px) and answers ``HTTP 500 invalid_request`` above that —
so the relay dialect validates against the smaller window up front.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Any

from .api_common import (
    ParsedResponse,
    PreparedRequest,
    check_response,
    decode_b64_image,
    download_bytes,
)
from .errors import EndpointError
from .provider import PROTOCOL_QWEN_IMAGES, join_url

DIALECT_NATIVE = "native"
DIALECT_OPENAI_IMAGES = "openai_images"

GENERATION_PATH = "services/aigc/multimodal-generation/generation"
OPENAI_EDIT_PATH = "/images/edits"
OPENAI_GENERATE_PATH = "/images/generations"

MAX_EDIT_IMAGES = 3
MIN_AREA = 262144  # 512*512
MAX_AREA = 6553600  # 2560*2560, official DashScope path
MAX_AREA_RELAY = 4194304  # 2048*2048, observed relay upstream limit
MAX_ASPECT = 8  # the API allows 1:8 .. 8:1
MAX_INPUT_BYTES = 10 * 1024 * 1024  # the API rejects decoded inputs over 10 MB
MAX_INPUT_PIXELS = 2048 * 2048  # the partner node downscales reference images to this
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
MIN_N, MAX_N = 1, 6
MAX_SEED = 2147483647

_MATCH_INPUT_ALIASES = frozenset({"match input", "match_input", "matchinput", "input", "match"})
_MISSING_ROUTE_MARKERS = ("invalid url", "unknown url", "no route", "not found route")
_IMAGE_REF_RE = re.compile(r"@image(?P<idx>\d*)(?!\w)", re.IGNORECASE | re.ASCII)


# ---------------------------------------------------------------------------
# dialect + endpoints
# ---------------------------------------------------------------------------
def dialect_for(provider: Any, dialect: str | None = None) -> str:
    """Wire shape for this provider (explicit override wins)."""
    if dialect in (DIALECT_NATIVE, DIALECT_OPENAI_IMAGES):
        return dialect
    if getattr(provider, "api_protocol", "") == PROTOCOL_QWEN_IMAGES:
        return DIALECT_OPENAI_IMAGES
    return DIALECT_NATIVE


def generation_endpoint(provider: Any) -> str:
    """DashScope native URL (base URL + generation path; a full URL is used as-is)."""
    base = str(provider.base_url).rstrip("/")
    if "/multimodal-generation/generation" in base:
        return base
    return join_url(base, GENERATION_PATH)


def openai_surface_url(provider: Any, operation: str) -> str:
    """Relay URL for ``generate`` / ``edit``."""
    base = str(provider.base_url).rstrip("/")
    path = OPENAI_EDIT_PATH if operation == "edit" else OPENAI_GENERATE_PATH
    if base.endswith(path):
        return base
    for known in (OPENAI_EDIT_PATH, OPENAI_GENERATE_PATH):
        if base.endswith(known):
            base = base[: -len(known)]
            break
    return base + path


def is_missing_route(response: Any) -> bool:
    """True when a gateway answered 'this route does not exist'.

    That is the signature of an OpenAI-compatible relay being asked for the
    DashScope native path; the caller retries on the relay surface.
    """
    if getattr(response, "status", None) != 404:
        return False
    text = str(getattr(response, "text", "") or "").lower()
    return any(marker in text for marker in _MISSING_ROUTE_MARKERS)


# ---------------------------------------------------------------------------
# size helpers (ported from the partner node)
# ---------------------------------------------------------------------------
def max_area_for(dialect: str) -> int:
    return MAX_AREA_RELAY if dialect == DIALECT_OPENAI_IMAGES else MAX_AREA


def _fit_to_size(width: int, height: int, max_area: int = MAX_AREA) -> tuple[int, int]:
    """Scale into the supported pixel area and 1:8..8:1 range, keeping the ratio."""
    import math

    width = max(1, int(width))
    height = max(1, int(height))
    if width > MAX_ASPECT * height:
        height = math.ceil(width / MAX_ASPECT)
    elif height > MAX_ASPECT * width:
        width = math.ceil(height / MAX_ASPECT)
    area = width * height
    if area < MIN_AREA:
        scale = math.sqrt(MIN_AREA / area)
        width, height = math.ceil(width * scale), math.ceil(height * scale)
    elif area > max_area:
        scale = math.sqrt(max_area / area)
        width, height = math.floor(width * scale), math.floor(height * scale)
    # rounding can push the ratio a hair past the limit; trimming only shrinks area
    return min(width, MAX_ASPECT * height), min(height, MAX_ASPECT * width)


def _size_error(message: str, max_area: int, dialect: str) -> EndpointError:
    hint = (
        f" This relay's upstream caps at {MAX_AREA_RELAY} px (2048x2048); connect the "
        f"official DashScope endpoint (protocol 'qwen') for the 2560x2560 window."
        if dialect == DIALECT_OPENAI_IMAGES
        else ""
    )
    return EndpointError(
        f"{message} (allowed pixel area {MIN_AREA} .. {max_area}, aspect 1:{MAX_ASPECT}.."
        f"{MAX_ASPECT}:1).{hint}",
        error_type="validation",
        error_code="invalid_size",
        phase="validate",
    )


def resolve_size(width: Any, height: Any, *, max_area: int = MAX_AREA, dialect: str = DIALECT_NATIVE) -> str:
    """Validate an explicit size and return the wire value ``"W*H"``."""
    try:
        width_value, height_value = int(width), int(height)
    except (TypeError, ValueError) as error:
        raise EndpointError(
            f"width and height must be integers, got {width!r}x{height!r}.",
            error_type="validation",
            error_code="invalid_size",
            phase="validate",
        ) from error
    area = width_value * height_value
    if not MIN_AREA <= area <= max_area:
        raise _size_error(
            f"Image area must be between {MIN_AREA} (512x512) and {max_area} pixels; "
            f"got {width_value}x{height_value} = {area}",
            max_area,
            dialect,
        )
    if width_value > MAX_ASPECT * height_value or height_value > MAX_ASPECT * width_value:
        raise _size_error(
            f"Aspect ratio must be between 1:{MAX_ASPECT} and {MAX_ASPECT}:1; "
            f"got {width_value}x{height_value}",
            max_area,
            dialect,
        )
    return f"{width_value}*{height_value}"


# ---------------------------------------------------------------------------
# reference images
# ---------------------------------------------------------------------------
def _image_dims(payload: bytes) -> tuple[int, int]:
    """Decoded ``(width, height)`` of a PNG/JPEG/WebP payload."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(payload)) as image:
            return int(image.width), int(image.height)
    except Exception as error:  # noqa: BLE001 - an unusable input is data
        raise EndpointError(
            f"Could not read a reference image ({error}).",
            error_type="validation",
            error_code="bad_reference_image",
            phase="validate",
        ) from error


def _upload_payload(payload: bytes) -> tuple[bytes, str]:
    """Return ``(bytes, mime)`` capped at 2048x2048 pixels and 10 MB.

    The original bytes are returned untouched when no cap applies, so the
    common case pays no re-encode.
    """
    from PIL import Image

    mime = "image/png"
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width, height = int(image.width), int(image.height)
            if width * height <= MAX_INPUT_PIXELS and len(payload) <= MAX_INPUT_BYTES:
                return payload, mime
            if width * height > MAX_INPUT_PIXELS:
                scale = (MAX_INPUT_PIXELS / float(width * height)) ** 0.5
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                image = image.resize(size, Image.LANCZOS)
            rgb = image.convert("RGB")
    except Exception as error:  # noqa: BLE001
        raise EndpointError(
            f"Could not encode a reference image ({error}).",
            error_type="validation",
            error_code="bad_reference_image",
            phase="validate",
        ) from error

    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    encoded = buffer.getvalue()
    if len(encoded) <= MAX_INPUT_BYTES:
        return encoded, mime
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue(), "image/jpeg"


def _image_data_uri(payload: bytes) -> str:
    encoded, mime = _upload_payload(payload)
    return f"data:{mime};base64," + base64.b64encode(encoded).decode("ascii")


def resolve_image_refs(prompt: str, total_images: int) -> str:
    """Rewrite ``@Image1``-style references into ``Image 1`` (1-based).

    A tag counts only at a word boundary or right after a previous tag, so
    ``@Image1@Image2`` resolves while ``user@image1.com`` passes through.
    """
    parts: list[str] = []
    pos = 0
    prev_end = -1
    for match in _IMAGE_REF_RE.finditer(prompt):
        start = match.start()
        if start > 0 and start != prev_end and (prompt[start - 1].isalnum() or prompt[start - 1] == "_"):
            continue
        idx = int(match.group("idx") or 1)
        if not 1 <= idx <= total_images:
            raise EndpointError(
                f"The prompt references @Image{idx}, but only {total_images} reference "
                f"image(s) are connected (a batched input counts once per image).",
                error_type="validation",
                error_code="invalid_image_reference",
                phase="validate",
            )
        parts.append(prompt[pos:start])
        parts.append(f"Image {idx}")
        pos = match.end()
        prev_end = match.end()
    parts.append(prompt[pos:])
    return "".join(parts)


# ---------------------------------------------------------------------------
# parameter block
# ---------------------------------------------------------------------------
def _param_values(params: dict[str, Any]) -> dict[str, Any]:
    """Validate n/seed once, for both dialects."""
    try:
        n = int(params.get("n") or 1)
        seed = int(params.get("seed") or 0)
    except (TypeError, ValueError) as error:
        raise EndpointError(
            f"n and seed must be integers ({error}).",
            error_type="validation",
            error_code="invalid_parameter",
            phase="validate",
        ) from error
    if not MIN_N <= n <= MAX_N:
        raise EndpointError(
            f"n must be between {MIN_N} and {MAX_N}, got {n}.",
            error_type="validation",
            error_code="invalid_n",
            phase="validate",
        )
    if not 0 <= seed <= MAX_SEED:
        raise EndpointError(
            f"seed must be between 0 and {MAX_SEED}, got {seed}.",
            error_type="validation",
            error_code="invalid_seed",
            phase="validate",
        )
    return {
        "n": n,
        "seed": seed,
        "prompt_extend": bool(params.get("prompt_extend", True)),
        "watermark": bool(params.get("watermark", False)),
        "negative_prompt": str(params.get("negative_prompt") or "").strip() or None,
    }


def _edit_size(params: dict[str, Any], images: list[bytes], dialect: str) -> str | None:
    max_area = max_area_for(dialect)
    mode = str(params.get("size_mode") or "match input").strip()
    lowered = " ".join(mode.lower().split())
    if lowered in _MATCH_INPUT_ALIASES:
        if not images:
            raise EndpointError(
                "size='match input' needs at least one reference image; connect an image "
                "or pick 'auto'.",
                error_type="validation",
                error_code="missing_image",
                phase="validate",
            )
        width, height = _image_dims(images[0])
        fitted_width, fitted_height = _fit_to_size(width, height, max_area)
        return f"{fitted_width}*{fitted_height}"
    if lowered in ("", "auto"):
        return None
    return resolve_size(
        params.get("width", DEFAULT_WIDTH), params.get("height", DEFAULT_HEIGHT),
        max_area=max_area, dialect=dialect,
    )


# ---------------------------------------------------------------------------
# request builders
# ---------------------------------------------------------------------------
def _native_request(
    provider: Any,
    *,
    operation: str,
    text: str,
    images: list[bytes],
    size: str | None,
    values: dict[str, Any],
    image_summary: dict[str, Any],
) -> PreparedRequest:
    from .transport import auth_headers

    content: list[dict[str, Any]] = [{"image": _image_data_uri(payload)} for payload in images]
    content.append({"text": text})
    parameters: dict[str, Any] = {}
    if size:
        parameters["size"] = size
    parameters["n"] = values["n"]
    parameters["seed"] = values["seed"]
    parameters["prompt_extend"] = values["prompt_extend"]
    parameters["watermark"] = values["watermark"]
    if values["negative_prompt"]:
        parameters["negative_prompt"] = values["negative_prompt"]
    body: dict[str, Any] = {
        "model": provider.resolved_model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": parameters,
    }
    return PreparedRequest(
        method="POST",
        url=generation_endpoint(provider),
        headers=auth_headers(provider),
        json_body=body,
        operation=operation,
        image_summary=image_summary,
    )


def _openai_request(
    provider: Any,
    *,
    operation: str,
    text: str,
    images: list[bytes],
    size: str | None,
    values: dict[str, Any],
    image_summary: dict[str, Any],
) -> PreparedRequest:
    from .transport import auth_headers

    body: dict[str, Any] = {
        "model": provider.resolved_model,
        "prompt": text,
        "n": values["n"],
        "watermark": values["watermark"],
        "prompt_extend": values["prompt_extend"],
        "seed": values["seed"],
    }
    if size:
        body["size"] = size
    if values["negative_prompt"]:
        body["negative_prompt"] = values["negative_prompt"]
    if operation == "edit":
        # one reference -> plain string, several -> list (both verified live)
        uris = [_image_data_uri(payload) for payload in images]
        body["image"] = uris[0] if len(uris) == 1 else uris
    return PreparedRequest(
        method="POST",
        url=openai_surface_url(provider, operation),
        headers=auth_headers(provider),
        json_body=body,
        operation=operation,
        image_summary=image_summary,
    )


def prepare_image_request(
    provider: Any,
    *,
    operation: str,
    prompt: str,
    images: list[bytes] | None = None,
    mask: bytes | None = None,
    params: dict[str, Any] | None = None,
    image_summary: dict[str, Any] | None = None,
    dialect: str | None = None,
) -> PreparedRequest:
    """Build one text-to-image (``generate``) or ``edit`` request."""
    params = dict(params or {})
    images = list(images or [])
    wire = dialect_for(provider, dialect)
    if mask is not None:
        raise EndpointError(
            "Qwen-Image has no mask field; describe the edit in the prompt. "
            "Remove the mask or use a GPT provider.",
            error_type="validation",
            error_code="mask_unsupported",
            phase="validate",
        )
    text = str(prompt or "")
    if not text.strip():
        raise EndpointError(
            "prompt is empty; Qwen-Image needs a prompt.",
            error_type="validation",
            error_code="empty_prompt",
            phase="validate",
        )
    values = _param_values(params)

    if operation == "edit" or images:
        if not images:
            raise EndpointError(
                "Qwen-Image editing requires at least one reference image.",
                error_type="validation",
                error_code="missing_image",
                phase="validate",
            )
        if len(images) > MAX_EDIT_IMAGES:
            raise EndpointError(
                f"Qwen-Image supports at most {MAX_EDIT_IMAGES} reference images, got "
                f"{len(images)} (a batched input counts once per image).",
                error_type="validation",
                error_code="too_many_images",
                phase="validate",
            )
        text = resolve_image_refs(text, len(images))
        size = _edit_size(params, images, wire)
        operation_name = "edit"
    else:
        size = resolve_size(
            params.get("width", DEFAULT_WIDTH), params.get("height", DEFAULT_HEIGHT),
            max_area=max_area_for(wire), dialect=wire,
        )
        operation_name = "generate"

    builder = _native_request if wire == DIALECT_NATIVE else _openai_request
    return builder(
        provider,
        operation=operation_name,
        text=text,
        images=images,
        size=size,
        values=values,
        image_summary=image_summary or {},
    )


# ---------------------------------------------------------------------------
# response (tolerant: both surfaces)
# ---------------------------------------------------------------------------
def _urls_and_texts(payload: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Extract ``(image_urls, b64_payloads, texts)`` from either response shape."""
    urls: list[str] = []
    b64: list[str] = []
    texts: list[str] = []

    def take_item(item: Any) -> None:
        if isinstance(item, str):
            if item.startswith("data:"):
                b64.append(item)
            elif item.startswith("http"):
                urls.append(item)
            return
        if not isinstance(item, dict):
            return
        if isinstance(item.get("b64_json"), str) and item["b64_json"]:
            b64.append(item["b64_json"])
        elif isinstance(item.get("url"), str) and item["url"]:
            urls.append(item["url"])
        elif isinstance(item.get("image"), str) and item["image"]:
            urls.append(item["image"])
        elif isinstance(item.get("text"), str) and item["text"]:
            texts.append(item["text"])
        elif isinstance(item.get("image_url"), dict) and item["image_url"].get("url"):
            urls.append(item["image_url"]["url"])

    for key in ("data", "images"):
        for item in payload.get(key) or []:
            take_item(item)

    output = payload.get("output")
    if isinstance(output, dict):
        for choice in output.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            for item in message.get("content") or []:
                take_item(item)
    elif isinstance(output, list):
        for item in output:
            take_item(item)
    return urls, b64, texts


def parse_image_response(
    provider: Any, response: Any, client: Any, *, timeout: float
) -> ParsedResponse:
    check_response(response, protocol=provider.api_protocol, phase="image", provider=provider)
    payload = response.json()
    if not isinstance(payload, dict):
        raise EndpointError(
            "Qwen returned no JSON object.",
            status=response.status,
            error_type="protocol",
            phase="image",
            protocol=provider.api_protocol,
            response_excerpt=response.excerpt(600),
        )

    parsed = ParsedResponse()
    if isinstance(payload.get("error"), (dict, str)):
        # OpenAI-compatible relays may report a failure with HTTP 200.
        message = payload["error"]
        message = message.get("message", "") if isinstance(message, dict) else str(message)
        raise EndpointError(
            str(message) or "Qwen returned an error object.",
            status=response.status,
            error_type="provider",
            phase="image",
            protocol=provider.api_protocol,
            response_excerpt=response.excerpt(800),
        )

    output = payload.get("output")
    if isinstance(output, dict):
        for choice in output.get("choices") or []:
            if isinstance(choice, dict) and choice.get("finish_reason"):
                parsed.details.setdefault("finish_reasons", []).append(choice["finish_reason"])

    urls, b64_payloads, texts = _urls_and_texts(payload)
    image_urls: list[str] = []
    for url in urls:
        image_urls.append(url)
        parsed.images.append(
            download_bytes(
                client, url, timeout=timeout, phase="download", protocol=provider.api_protocol
            )
        )
    for data in b64_payloads:
        parsed.images.append(
            decode_b64_image(data, phase="image", protocol=provider.api_protocol)
        )
    parsed.image_urls = image_urls
    parsed.text = "\n".join(text for text in texts if text).strip()

    if payload.get("request_id"):
        parsed.details["request_id"] = payload["request_id"]
    if payload.get("usage"):
        parsed.details["usage"] = payload["usage"]
    if payload.get("created") is not None:
        parsed.details["created"] = payload["created"]
    code, message = payload.get("code"), payload.get("message")
    if code:
        parsed.details["provider_code"] = code
    if message:
        parsed.details["provider_message"] = message

    if not parsed.images:
        reason = " - ".join(str(part) for part in (code, message) if part) or "no image in output"
        raise EndpointError(
            f"Qwen returned no image ({reason}).",
            status=response.status,
            error_type="empty_result",
            error_code=code,
            phase="image",
            protocol=provider.api_protocol,
            response_excerpt=response.excerpt(800),
            details=parsed.details,
        )
    return parsed


__all__ = [
    "DIALECT_NATIVE",
    "DIALECT_OPENAI_IMAGES",
    "GENERATION_PATH",
    "MAX_EDIT_IMAGES",
    "dialect_for",
    "generation_endpoint",
    "openai_surface_url",
    "is_missing_route",
    "max_area_for",
    "resolve_size",
    "resolve_image_refs",
    "prepare_image_request",
    "parse_image_response",
]
