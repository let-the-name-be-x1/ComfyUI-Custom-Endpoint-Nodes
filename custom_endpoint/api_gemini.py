"""Gemini wire protocol (`{base}/models/{model}:generateContent`).

Field shapes follow the partner nodes: ``systemInstruction`` is a ``Content``
object, ``responseModalities`` is an array, and ``imageConfig`` nests
``imageOutputOptions``.  Flat widget values are coerced to those wire types
here — a flat string where the schema wants an object is the classic
``400 INVALID_ARGUMENT`` on a Vertex-compatible relay.
"""

from __future__ import annotations

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


def generate_endpoint(provider: Any) -> str:
    base = str(provider.base_url).rstrip("/")
    if ":generateContent" in base:
        return base
    return f"{base}/models/{provider.resolved_model}:generateContent"


def _contents(prompt: str, images: list[bytes], files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import base64

    parts: list[dict[str, Any]] = [{"text": prompt}]
    for payload in images:
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(payload).decode("ascii"),
                }
            }
        )
    for entry in files:
        parts.append(entry)
    return [{"role": "user", "parts": parts}]


def _system_instruction(system_prompt: Any) -> dict[str, Any] | None:
    text = str(system_prompt or "").strip()
    if not text:
        return None
    if text.startswith("{"):
        import json

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and "parts" in parsed:
            return parsed
    return {"parts": [{"text": text}]}


def _thinking_config(params: dict[str, Any]) -> dict[str, Any] | None:
    level = params.get("thinking_level")
    include = params.get("include_thoughts")
    if level is None and include is None:
        return None
    config: dict[str, Any] = {}
    if level is not None:
        config["thinkingLevel"] = str(level)
    if include is not None:
        config["includeThoughts"] = bool(include)
    return config


def _image_config(params: dict[str, Any]) -> dict[str, Any] | None:
    aspect = params.get("aspect_ratio")
    resolution = params.get("resolution")
    mime = params.get("mime_type")
    compression = params.get("compression_quality")
    config: dict[str, Any] = {}
    if resolution is not None:
        config["imageSize"] = str(resolution)
    if aspect is not None and str(aspect) != "auto":
        config["aspectRatio"] = str(aspect)
    options: dict[str, Any] = {}
    if mime is not None:
        options["mimeType"] = str(mime)
    if compression is not None:
        value = int(compression)
        if 0 <= value <= 100:
            options["compressionQuality"] = value
    if options:
        options.setdefault("mimeType", "image/png")
        config["imageOutputOptions"] = options
    return config or None


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
    files = list(params.pop("files", None) or [])
    if mask is not None:
        raise EndpointError(
            "Gemini's generateContent has no mask field; use the mask as an "
            "additional reference image plus a prompt instruction instead.",
            error_type="validation",
            error_code="mask_unsupported",
            phase="validate",
        )

    generation: dict[str, Any] = {}
    modalities = params.get("response_modalities")
    if modalities is not None:
        text_value = str(modalities).upper()
        generation["responseModalities"] = (
            ["TEXT", "IMAGE"] if "+" in text_value or "," in text_value else ["IMAGE"]
        )
    image_config = _image_config(params)
    if image_config:
        generation["imageConfig"] = image_config
    thinking = _thinking_config(params)
    if thinking:
        generation["thinkingConfig"] = thinking
    if params.get("temperature") is not None:
        generation["temperature"] = float(params["temperature"])
    if params.get("top_p") is not None:
        generation["topP"] = float(params["top_p"])
    if params.get("top_k") is not None:
        generation["topK"] = int(params["top_k"])
    if params.get("max_output_tokens") is not None:
        generation["maxOutputTokens"] = int(params["max_output_tokens"])
    if params.get("seed") is not None:
        generation["seed"] = int(params["seed"])
    stop = params.get("stop_sequences")
    if stop:
        generation["stopSequences"] = list(stop)

    body: dict[str, Any] = {"contents": _contents(prompt, images, files)}
    if generation:
        body["generationConfig"] = generation
    system = _system_instruction(params.get("system_prompt"))
    if system:
        body["systemInstruction"] = system
    if params.get("safety_settings"):
        body["safetySettings"] = params["safety_settings"]

    return PreparedRequest(
        method="POST",
        url=generate_endpoint(provider),
        headers=auth_headers(provider),
        json_body=body,
        operation=operation,
        image_summary=image_summary or {},
    )


def prepare_chat_request(
    provider: Any,
    *,
    prompt: str,
    images: list[bytes] | None = None,
    files: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
    image_summary: dict[str, Any] | None = None,
) -> PreparedRequest:
    from .transport import auth_headers

    params = dict(params or {})
    request = prepare_image_request(
        provider,
        operation="generate_content",
        prompt=prompt,
        images=images,
        params={**params, "files": list(files or [])},
        image_summary=image_summary,
    )
    request.headers = auth_headers(provider)
    return request


def parse_response(provider: Any, response: Any, client: Any, *, timeout: float, want_images: bool) -> ParsedResponse:
    check_response(
        response, protocol=provider.api_protocol, phase="image" if want_images else "chat", provider=provider
    )
    payload = response.json()
    if not isinstance(payload, dict):
        raise EndpointError(
            "Provider returned no JSON object.",
            status=response.status,
            error_type="protocol",
            phase="decode",
            protocol=provider.api_protocol,
            response_excerpt=response.excerpt(600),
        )

    parsed = ParsedResponse()
    texts: list[str] = []
    for candidate in payload.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("finishReason"):
            parsed.details.setdefault("finish_reasons", []).append(candidate["finishReason"])
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                data = decode_b64_image(str(inline["data"]), phase="image", protocol=provider.api_protocol)
                if part.get("thought"):
                    parsed.thought_images.append(data)
                else:
                    parsed.images.append(data)
                continue
            file_data = part.get("fileData") or part.get("file_data")
            if isinstance(file_data, dict) and file_data.get("fileUri"):
                uri = str(file_data["fileUri"])
                parsed.image_urls.append(uri)
                parsed.images.append(
                    download_bytes(
                        client, uri, timeout=timeout, phase="download", protocol=provider.api_protocol
                    )
                )
                continue
            if isinstance(part.get("text"), str) and part["text"]:
                if part.get("thought"):
                    parsed.details.setdefault("thought_text", []).append(part["text"])
                else:
                    texts.append(part["text"])

    feedback = payload.get("promptFeedback") or {}
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        parsed.details["block_reason"] = feedback.get("blockReason")
        if feedback.get("blockReasonMessage"):
            parsed.details["block_reason_message"] = feedback["blockReasonMessage"]

    parsed.text = "\n".join(texts).strip()
    if payload.get("usageMetadata"):
        parsed.details["usage"] = payload["usageMetadata"]
    if payload.get("modelVersion"):
        parsed.details["model_version"] = payload["modelVersion"]

    if want_images and not parsed.images:
        reason = parsed.details.get("block_reason")
        message = (
            f"Gemini returned no image (blockReason={reason})."
            if reason
            else "Gemini returned no image."
        )
        raise EndpointError(
            message,
            status=response.status,
            error_type="empty_result",
            error_code=reason,
            phase="image",
            protocol=provider.api_protocol,
            response_excerpt=response.excerpt(800),
            details=parsed.details,
        )
    if not want_images and not parsed.text:
        reason = parsed.details.get("block_reason")
        raise EndpointError(
            f"Gemini returned no text (blockReason={reason})." if reason else "Gemini returned no text.",
            status=response.status,
            error_type="empty_result",
            error_code=reason,
            phase="chat",
            protocol=provider.api_protocol,
            response_excerpt=response.excerpt(800),
            details=parsed.details,
        )
    return parsed
