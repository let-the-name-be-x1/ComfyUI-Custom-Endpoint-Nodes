"""xAI Grok wire protocol: OpenAI-compatible images plus chat/completions.

Grok has no mask field, so an edit is a prompt-driven transformation of the
supplied source images (``images[].url`` data URLs), exactly like the partner
node's `GrokImageEditNodeV2`.
"""

from __future__ import annotations

import base64
from typing import Any

from .api_common import (
    ParsedResponse,
    PreparedRequest,
    check_response,
    decode_b64_image,
    download_bytes,
)
from .errors import EndpointError
from .provider import join_url

MAX_EDIT_IMAGES = 3
PRO_MODEL = "grok-imagine-image-pro"


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
    from .transport import auth_headers

    params = dict(params or {})
    images = list(images or [])
    model = provider.resolved_model
    if mask is not None:
        raise EndpointError(
            "Grok has no mask field; the edit is driven by the prompt over the "
            "source image(s). Remove the mask or use a GPT/Gemini provider.",
            error_type="validation",
            error_code="mask_unsupported",
            phase="validate",
        )

    body: dict[str, Any] = {"model": model, "prompt": prompt}
    for key in ("aspect_ratio", "resolution", "quality", "response_format"):
        if params.get(key) is not None:
            body[key] = params[key]
    if params.get("resolution") is not None:
        body["resolution"] = str(params["resolution"]).lower()
    if params.get("n") is not None:
        body["n"] = int(params["n"])
    if params.get("seed") is not None:
        body["seed"] = int(params["seed"])

    if operation == "edit" or images:
        limit = 1 if model == PRO_MODEL else MAX_EDIT_IMAGES
        if not images:
            raise EndpointError(
                "Grok image editing requires at least one source image.",
                error_type="validation",
                error_code="missing_image",
                phase="validate",
            )
        if len(images) > limit:
            raise EndpointError(
                f"{model} supports at most {limit} input image(s), got {len(images)}.",
                error_type="validation",
                error_code="too_many_images",
                phase="validate",
            )
        if len(images) == 1 and body.get("aspect_ratio") not in (None, "auto"):
            raise EndpointError(
                "A custom aspect ratio is only allowed when multiple images are connected.",
                error_type="validation",
                error_code="invalid_aspect_ratio",
                phase="validate",
            )
        body["images"] = [
            {"url": "data:image/png;base64," + base64.b64encode(payload).decode("ascii")}
            for payload in images
        ]
        if body.get("aspect_ratio") == "auto":
            body.pop("aspect_ratio", None)
        return PreparedRequest(
            method="POST",
            url=join_url(provider.base_url, "/images/edits"),
            headers=auth_headers(provider),
            json_body=body,
            operation="edit",
            image_summary=image_summary or {},
        )

    return PreparedRequest(
        method="POST",
        url=join_url(provider.base_url, "/images/generations"),
        headers=auth_headers(provider),
        json_body=body,
        operation="generate",
        image_summary=image_summary or {},
    )


def parse_image_response(
    provider: Any, response: Any, client: Any, *, timeout: float
) -> ParsedResponse:
    check_response(response, protocol=provider.api_protocol, phase="image", provider=provider)
    payload = response.json()
    if not isinstance(payload, dict):
        raise EndpointError(
            "Grok returned no JSON object.",
            status=response.status,
            error_type="protocol",
            phase="image",
            protocol=provider.api_protocol,
            response_excerpt=response.excerpt(600),
        )
    data = payload.get("data") or []
    if not isinstance(data, list) or not data:
        raise EndpointError(
            "Grok returned no images.",
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
        if isinstance(item.get("revised_prompt"), str) and item["revised_prompt"]:
            parsed.revised_prompts.append(item["revised_prompt"])
        if isinstance(item.get("b64_json"), str) and item["b64_json"]:
            parsed.images.append(
                decode_b64_image(item["b64_json"], phase="image", protocol=provider.api_protocol)
            )
            continue
        if isinstance(item.get("url"), str) and item["url"]:
            parsed.image_urls.append(item["url"])
            parsed.images.append(
                download_bytes(
                    client, item["url"], timeout=timeout, phase="download", protocol=provider.api_protocol
                )
            )
    if not parsed.images:
        raise EndpointError(
            "Grok returned an image payload without url or b64_json.",
            status=response.status,
            error_type="empty_result",
            phase="image",
            protocol=provider.api_protocol,
        )
    usage = payload.get("usage")
    if isinstance(usage, dict):
        parsed.details["usage"] = usage
        if usage.get("cost_in_usd_ticks") is not None:
            parsed.details["cost_in_usd_ticks"] = usage["cost_in_usd_ticks"]
    if parsed.revised_prompts:
        parsed.details["revised_prompt"] = parsed.revised_prompts[0]
    return parsed


def prepare_chat_request(
    provider: Any,
    *,
    prompt: str,
    images: list[bytes] | None = None,
    files: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
    image_summary: dict[str, Any] | None = None,
) -> PreparedRequest:
    """Grok text models are reached through an OpenAI-compatible chat endpoint.

    The partner nodes ship no Grok LLM node; this is the documented xAI surface.
    """
    from .api_openai import _chat_message_content
    from .transport import auth_headers

    params = dict(params or {})
    images = list(images or [])
    files = list(files or [])
    messages: list[dict[str, Any]] = []
    system_prompt = str(params.get("system_prompt") or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": _chat_message_content(prompt, images, files)})
    body: dict[str, Any] = {
        "model": provider.resolved_model,
        "messages": messages,
        "stream": False,
    }
    if params.get("temperature") is not None:
        body["temperature"] = float(params["temperature"])
    if params.get("top_p") is not None:
        body["top_p"] = float(params["top_p"])
    if params.get("max_tokens") is not None:
        body["max_tokens"] = int(params["max_tokens"])
    if params.get("reasoning_effort"):
        body["reasoning_effort"] = params["reasoning_effort"]
    return PreparedRequest(
        method="POST",
        url=join_url(provider.base_url, "/chat/completions"),
        headers=auth_headers(provider),
        json_body=body,
        operation="chat_completions",
        image_summary=image_summary or {},
    )


def parse_chat_response(provider: Any, response: Any) -> ParsedResponse:
    from .api_openai import parse_chat_response as parse_openai_chat

    return parse_openai_chat(provider, response)
