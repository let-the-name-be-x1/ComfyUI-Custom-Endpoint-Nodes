"""Server-side prompt guard + one small route.

Two jobs, both about keeping credentials out of persisted artifacts:

1. ``POST /custom-endpoint/session-credential`` — the frontend can hand the raw
   key to the server and only receive an opaque handle back.
2. ``guard_prompt_payload`` — hooked into ``PromptServer.on_prompt_handlers``:

   * a raw API key found in the queued prompt is *replaced* by a freshly issued
     session handle, so execution keeps working while the queue item, the
     ComfyUI history and the PNG metadata never carry the key;
   * the UI workflow attached to the request has the provider node's endpoint
     and API-key widget values cleared.  ``SaveImage`` embeds exactly this
     workflow (plus the prompt) into the PNG metadata, so this is what makes
     "image metadata never contains the endpoint or the key" unconditional.
"""

from __future__ import annotations

from typing import Any

from .secrets import is_session_handle, issue_session_credential

ROUTE_PATH = "/custom-endpoint/session-credential"
MAX_REQUEST_BYTES = 64 * 1024

PROVIDER_CLASS_TYPES = frozenset({"CustomEndpointProvider"})
PROVIDER_SECRET_WIDGET_NAMES = ("endpoint", "api_key", "base_url")
PROVIDER_SECRET_WIDGET_INDICES = (0, 1)
REDACTED = "[REDACTED]"


def credential_response(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")
    return {"credential_handle": issue_session_credential(payload.get("api_key", ""))}


def redact_exact_string(value: Any, secret: str) -> None:
    """Replace a raw credential with [REDACTED] in every string of a payload."""
    if not secret:
        return
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if isinstance(item, str):
                value[key] = item.replace(secret, REDACTED)
            else:
                redact_exact_string(item, secret)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, str):
                value[index] = item.replace(secret, REDACTED)
            else:
                redact_exact_string(item, secret)


def _scrub_widgets_values(widgets_values: Any) -> None:
    """Blank the endpoint and API-key slots of the provider node."""
    if isinstance(widgets_values, list):
        for index in PROVIDER_SECRET_WIDGET_INDICES:
            if index < len(widgets_values) and isinstance(widgets_values[index], str):
                widgets_values[index] = ""
    elif isinstance(widgets_values, dict):
        for name in PROVIDER_SECRET_WIDGET_NAMES:
            if isinstance(widgets_values.get(name), str):
                widgets_values[name] = ""


def scrub_provider_nodes(value: Any) -> None:
    """Walk a (UI-format or API-format) graph and clear provider secrets."""
    if isinstance(value, dict):
        node_type = value.get("type") or value.get("class_type")
        if node_type in PROVIDER_CLASS_TYPES:
            if "widgets_values" in value:
                _scrub_widgets_values(value["widgets_values"])
            inputs = value.get("inputs")
            if isinstance(inputs, dict) and isinstance(inputs.get("api_key"), str):
                if not is_session_handle(inputs["api_key"]):
                    inputs["api_key"] = ""
        for item in value.values():
            scrub_provider_nodes(item)
    elif isinstance(value, list):
        for item in value:
            scrub_provider_nodes(item)


def guard_prompt_payload(json_data: Any) -> Any:
    """on_prompt_handler: keep raw credentials out of everything persisted."""
    if not isinstance(json_data, dict):
        return json_data

    raw_secrets: list[str] = []
    prompt = json_data.get("prompt")
    if isinstance(prompt, dict):
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") not in PROVIDER_CLASS_TYPES:
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            for name in ("api_key", "endpoint"):
                candidate = inputs.get(name)
                if not isinstance(candidate, str) or not candidate.strip():
                    continue
                if is_session_handle(candidate):
                    continue
                # Replace the plaintext value with a one-time handle so the
                # queued prompt, the history entry and the PNG metadata stay
                # clean while the provider node can still resolve it.
                inputs[name] = issue_session_credential(candidate)
                raw_secrets.append(candidate)

    for secret in raw_secrets:
        redact_exact_string(json_data, secret)

    scrub_provider_nodes(json_data)
    return json_data


def register_routes() -> bool:
    """Register the credential route + prompt guard when running inside ComfyUI."""
    try:
        from aiohttp import web
        from server import PromptServer
    except (ImportError, AttributeError):
        return False

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return False

    route_marker = "_custom_endpoint_session_credential_route"
    if not getattr(prompt_server, route_marker, False):

        @prompt_server.routes.post(ROUTE_PATH)
        async def store_session_credential(request):  # pragma: no cover - aiohttp
            if request.content_length is not None and request.content_length > MAX_REQUEST_BYTES:
                return web.json_response(
                    {"error": "request_too_large"},
                    status=413,
                    headers={"Cache-Control": "no-store"},
                )
            try:
                result = credential_response(await request.json())
            except (ValueError, TypeError):
                return web.json_response(
                    {"error": "api_key_required"},
                    status=400,
                    headers={"Cache-Control": "no-store"},
                )
            return web.json_response(result, headers={"Cache-Control": "no-store"})

        setattr(prompt_server, route_marker, True)

    guard_marker = "_custom_endpoint_prompt_guard"
    if not getattr(prompt_server, guard_marker, False):
        prompt_server.add_on_prompt_handler(guard_prompt_payload)
        setattr(prompt_server, guard_marker, True)
    return True
