"""OpenAI-family wire protocol: GPT Image (images/*), ChatGPT (/chat/completions
and /responses).

Field names mirror the ComfyUI partner nodes (`comfy_api_nodes/apis/openai.py`)
so a request built here is byte-comparable with the official one, minus the
fields the partner node never actually sends.
"""

from __future__ import annotations

import json
from typing import Any

from .api_common import (
    IMAGE_MIME_BY_FORMAT,
    ParsedResponse,
    PreparedRequest,
    check_response,
    decode_b64_image,
    download_bytes,
)
from .errors import EndpointError
from .provider import join_url
from .transport import build_multipart

IMAGE_EDIT_FIELD_SINGLE = "image"
IMAGE_EDIT_FIELD_MULTI = "image[]"


def resolve_size(params: dict[str, Any]) -> str | None:
    """Turn the size widget (+ custom width/height) into the wire value."""
    size = params.get("size")
    if size is None:
        return None
    size = str(size)
    if size != "Custom":
        return size
    width = int(params.get("custom_width") or 1024)
    height = int(params.get("custom_height") or 1024)
    if width % 16 or height % 16:
        raise EndpointError(
            f"Custom width and height must be multiples of 16, got {width}x{height}.",
            error_type="validation",
            error_code="invalid_size",
            phase="validate",
        )
    if max(width, height) > 3840:
        raise EndpointError(
            f"Custom resolution max edge must be <= 3840, got {width}x{height}.",
            error_type="validation",
            error_code="invalid_size",
            phase="validate",
        )
    ratio = max(width, height) / min(width, height)
    if ratio > 3:
        raise EndpointError(
            f"Custom resolution aspect ratio must not exceed 3:1, got {width}x{height}.",
            error_type="validation",
            error_code="invalid_size",
            phase="validate",
        )
    total = width * height
    if not 655_360 <= total <= 8_294_400:
        raise EndpointError(
            f"Custom resolution total pixels must be between 655,360 and 8,294,400, got {total}.",
            error_type="validation",
            error_code="invalid_size",
            phase="validate",
        )
    return f"{width}x{height}"


def _image_body_fields(provider: Any, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {"model": provider.resolved_model, "prompt": prompt}
    size = resolve_size(params)
    if size:
        body["size"] = size
    for key in ("quality", "background", "output_format", "moderation", "response_format"):
        value = params.get(key)
        if value is not None:
            body[key] = value
    if params.get("n") is not None:
        body["n"] = int(params["n"])
    if params.get("output_compression") is not None:
        compression = int(params["output_compression"])
        if 0 <= compression <= 100:
            body["output_compression"] = compression
    if params.get("seed") is not None:
        body["seed"] = int(params["seed"])
    return body


def prepare_image_request(
    provider: Any,
    *,
    operation: str,
    prompt: str,
    images: list[bytes] | None = None,
    mask: bytes | None = None,
    params: dict[str, Any] | None = None,
    image_summary: dict[str, Any] | None = None,
) -> PreparedRequest:
    """Build a `/images/generations` or `/images/edits` request."""
    params = dict(params or {})
    images = list(images or [])
    headers = _auth_headers(provider)

    if operation == "edit" and images:
        fields = _image_body_fields(provider, prompt, params)
        files: list[tuple[str, str, bytes, str]] = []
        field_name = IMAGE_EDIT_FIELD_SINGLE if len(images) == 1 else IMAGE_EDIT_FIELD_MULTI
        for index, payload in enumerate(images):
            files.append((field_name, f"image_{index}.png", payload, "image/png"))
        if mask is not None:
            files.append(("mask", "mask.png", mask, "image/png"))
        content_type, body = build_multipart(fields, files)
        return PreparedRequest(
            method="POST",
            url=join_url(provider.base_url, "/images/edits"),
            headers=headers,
            body=body,
            content_type=content_type,
            operation="edit",
            image_summary=image_summary or {},
            log_body={**fields, "_files": [name for name, *_ in files]},
        )

    body = _image_body_fields(provider, prompt, params)
    return PreparedRequest(
        method="POST",
        url=join_url(provider.base_url, "/images/generations"),
        headers=headers,
        json_body=body,
        operation="generate",
        image_summary=image_summary or {},
    )


def parse_image_response(
    provider: Any,
    response: Any,
    client: Any,
    *,
    timeout: float,
) -> ParsedResponse:
    check_response(response, protocol=provider.api_protocol, phase="image", provider=provider)
    payload = response.json()
    if not isinstance(payload, dict):
        raise EndpointError(
            "Provider returned no JSON object for an image request.",
            status=response.status,
            error_type="protocol",
            phase="image",
            protocol=provider.api_protocol,
            response_excerpt=response.excerpt(600),
        )
    data = payload.get("data") or []
    if not isinstance(data, list) or not data:
        raise EndpointError(
            "Provider returned no images.",
            status=response.status,
            error_type="empty_result",
            phase="image",
            protocol=provider.api_protocol,
            response_excerpt=response.excerpt(600),
        )
    parsed = ParsedResponse()
    for item in data:
        if not isinstance(item, dict):
            continue
        revised = item.get("revised_prompt")
        if isinstance(revised, str) and revised:
            parsed.revised_prompts.append(revised)
        b64 = item.get("b64_json")
        if isinstance(b64, str) and b64:
            parsed.images.append(
                decode_b64_image(b64, phase="image", protocol=provider.api_protocol)
            )
            continue
        url = item.get("url")
        if isinstance(url, str) and url:
            parsed.image_urls.append(url)
            parsed.images.append(
                download_bytes(
                    client,
                    url,
                    timeout=timeout,
                    phase="download",
                    protocol=provider.api_protocol,
                )
            )
    if not parsed.images:
        raise EndpointError(
            "Provider returned an image payload without url or b64_json.",
            status=response.status,
            error_type="empty_result",
            phase="image",
            protocol=provider.api_protocol,
        )
    usage = payload.get("usage")
    if isinstance(usage, dict):
        parsed.details["usage"] = usage
    if parsed.revised_prompts:
        parsed.details["revised_prompt"] = parsed.revised_prompts[0]
    return parsed


# ---------------------------------------------------------------------------
# Text models
# ---------------------------------------------------------------------------
def _content_parts(prompt: str, images: list[bytes], files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import base64

    parts: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for payload in images:
        parts.append(
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": "data:image/png;base64," + base64.b64encode(payload).decode("ascii"),
            }
        )
    for entry in files:
        parts.append(entry)
    return parts


def _chat_message_content(prompt: str, images: list[bytes], files: list[dict[str, Any]]) -> Any:
    import base64

    text_extras: list[str] = []
    for entry in files:
        filename = str(entry.get("filename") or "file")
        data = str(entry.get("file_data") or "")
        if filename.lower().endswith(".txt"):
            _, _, encoded = data.partition(",")
            try:
                text_extras.append(base64.b64decode(encoded).decode("utf-8", errors="replace"))
            except Exception:
                text_extras.append(f"[unreadable text file: {filename}]")
        else:
            raise EndpointError(
                f"/chat/completions cannot carry '{filename}'. Use the openai_responses "
                "protocol or a Gemini provider for PDF input.",
                error_type="validation",
                error_code="unsupported_file",
                phase="validate",
            )
    full_text = "\n\n".join([prompt] + text_extras) if text_extras else prompt
    if not images:
        return full_text
    content: list[dict[str, Any]] = [{"type": "text", "text": full_text}]
    for payload in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,"
                    + base64.b64encode(payload).decode("ascii")
                },
            }
        )
    return content


def prepare_chat_request(
    provider: Any,
    *,
    prompt: str,
    images: list[bytes] | None = None,
    files: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
    image_summary: dict[str, Any] | None = None,
) -> PreparedRequest:
    params = dict(params or {})
    images = list(images or [])
    files = list(files or [])
    headers = _auth_headers(provider)
    protocol = provider.api_protocol

    if protocol == "openai_responses":
        body: dict[str, Any] = {
            "model": provider.resolved_model,
            "input": [
                {"role": "user", "content": _content_parts(prompt, images, files)}
            ],
            "store": bool(params.get("store", False)),
            "stream": False,
        }
        if params.get("instructions"):
            body["instructions"] = params["instructions"]
        if params.get("max_output_tokens") is not None:
            body["max_output_tokens"] = int(params["max_output_tokens"])
        if params.get("truncation"):
            body["truncation"] = params["truncation"]
        if params.get("reasoning_effort"):
            body["reasoning"] = {"effort": params["reasoning_effort"]}
        return PreparedRequest(
            method="POST",
            url=join_url(provider.base_url, "/responses"),
            headers=headers,
            json_body=body,
            operation="responses",
            image_summary=image_summary or {},
        )

    body = {
        "model": provider.resolved_model,
        "messages": [
            {"role": "user", "content": _chat_message_content(prompt, images, files)}
        ],
        "stream": False,
    }
    if params.get("instructions"):
        body["messages"].insert(0, {"role": "system", "content": params["instructions"]})
    if params.get("temperature") is not None:
        body["temperature"] = float(params["temperature"])
    if params.get("top_p") is not None:
        body["top_p"] = float(params["top_p"])
    if params.get("max_output_tokens") is not None:
        body["max_tokens"] = int(params["max_output_tokens"])
    if params.get("reasoning_effort"):
        body["reasoning_effort"] = params["reasoning_effort"]
    return PreparedRequest(
        method="POST",
        url=join_url(provider.base_url, "/chat/completions"),
        headers=headers,
        json_body=body,
        operation="chat_completions",
        image_summary=image_summary or {},
    )


def parse_chat_response(provider: Any, response: Any) -> ParsedResponse:
    check_response(response, protocol=provider.api_protocol, phase="chat", provider=provider)
    payload = response.json()
    if not isinstance(payload, dict):
        raise EndpointError(
            "Provider returned no JSON object for a text request.",
            status=response.status,
            error_type="protocol",
            phase="chat",
            protocol=provider.api_protocol,
            response_excerpt=response.excerpt(600),
        )
    parsed = ParsedResponse()
    if provider.api_protocol == "openai_responses":
        output = payload.get("output") or []
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if isinstance(content, dict) and content.get("type") == "output_text":
                    texts.append(str(content.get("text") or ""))
        parsed.text = "\n".join(text for text in texts if text)
        if payload.get("error"):
            raise EndpointError(
                str(payload["error"]),
                status=response.status,
                error_type="provider",
                phase="chat",
                protocol=provider.api_protocol,
            )
    else:
        choices = payload.get("choices") or []
        if not choices:
            raise EndpointError(
                "Provider returned no choices.",
                status=response.status,
                error_type="empty_result",
                phase="chat",
                protocol=provider.api_protocol,
                response_excerpt=response.excerpt(600),
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parsed.text = "\n".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            ).strip()
        else:
            parsed.text = str(content or "")
        parsed.details["finish_reason"] = choices[0].get("finish_reason")
    if payload.get("usage"):
        parsed.details["usage"] = payload["usage"]
    if not parsed.text:
        raise EndpointError(
            "Provider returned an empty text response.",
            status=response.status,
            error_type="empty_result",
            phase="chat",
            protocol=provider.api_protocol,
            response_excerpt=response.excerpt(600),
        )
    return parsed


def _auth_headers(provider: Any) -> dict[str, str]:
    from .transport import auth_headers

    return auth_headers(provider)


__all__ = [
    "IMAGE_MIME_BY_FORMAT",
    "resolve_size",
    "prepare_image_request",
    "parse_image_response",
    "prepare_chat_request",
    "parse_chat_response",
    "json",
]
