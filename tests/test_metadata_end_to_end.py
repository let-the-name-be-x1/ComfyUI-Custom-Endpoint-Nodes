"""End-to-end: a plain-credential prompt goes in, a runnable prompt + clean PNG
metadata come out.

This exercises the exact ComfyUI path: ``/prompt`` body -> on_prompt_handler ->
queue/history -> ``SaveImage`` writing ``prompt`` + ``extra_pnginfo.workflow``
into the PNG.
"""

import io
import json

from PIL import Image, PngImagePlugin

import ce_nodes as nodes
from conftest import HandlerOpener, json_response, png_b64

SECRET = "sk-live-abcdefghijklmnop1234"
ENDPOINT = "https://api.example.com"
PROVIDER = "CustomEndpointProvider"


def _frontend_payload():
    """What the ComfyUI frontend POSTs to /prompt."""
    return {
        "client_id": "test",
        "prompt": {
            "1": {
                "class_type": PROVIDER,
                "inputs": {
                    "endpoint": ENDPOINT,
                    "api_key": SECRET,
                    "model": "gpt-image-2.5-flare",
                    "api_protocol": "auto",
                    "encrypt_export": True,
                    "timeout_sec": 300,
                    "retries": 1,
                    "extra_headers": "",
                    "provider_name": "provider",
                },
            },
            "2": {
                "class_type": "CustomEndpointGPTImage",
                "inputs": {
                    "provider": ["1", 0],
                    "prompt": "a cat",
                    "size": "auto",
                    "n": 1,
                },
            },
        },
        "extra_data": {
            "extra_pnginfo": {
                "workflow": {
                    "nodes": [
                        {
                            "id": 1,
                            "type": PROVIDER,
                            "widgets_values": [
                                ENDPOINT,
                                SECRET,
                                "gpt-image-2.5-flare",
                                "auto",
                                True,
                                300,
                                1,
                                "",
                                "provider",
                            ],
                        },
                        {"id": 2, "type": "CustomEndpointGPTImage", "widgets_values": ["a cat"]},
                    ],
                    "links": [[1, 1, 0, 2, 0, "CUSTOM_ENDPOINT_PROVIDER"]],
                }
            }
        },
    }


def _write_png_like_saveimage(prompt, extra_pnginfo) -> bytes:
    """Reproduce ComfyUI's SaveImage metadata writing."""
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", json.dumps(prompt))
    for key, value in extra_pnginfo.items():
        metadata.add_text(key, json.dumps(value))
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (0, 0, 0)).save(buffer, format="PNG", pnginfo=metadata)
    return buffer.getvalue()


def test_png_metadata_never_carries_endpoint_or_key():
    from custom_endpoint import routes

    payload = _frontend_payload()
    guarded = routes.guard_prompt_payload(payload)

    png = _write_png_like_saveimage(guarded["prompt"], guarded["extra_data"]["extra_pnginfo"])
    with Image.open(io.BytesIO(png)) as image:
        prompt_metadata = image.text["prompt"]
        workflow_metadata = image.text["workflow"]

    for blob in (prompt_metadata, workflow_metadata):
        assert SECRET not in blob
        assert ENDPOINT not in blob
    # the model stays, so a reloaded workflow is still recognizable
    assert "gpt-image-2.5-flare" in workflow_metadata or "gpt-image-2.5-flare" in prompt_metadata
    # the node still executes: only handles are in the prompt
    assert "custom-endpoint-session:" in prompt_metadata


def test_guarded_prompt_still_runs(install_client):
    """The provider node resolves the handles the guard installed."""
    from custom_endpoint import routes

    guarded = routes.guard_prompt_payload(_frontend_payload())
    provider, = nodes.CustomEndpointProvider().build(**guarded["prompt"]["1"]["inputs"])
    assert provider.base_url == f"{ENDPOINT}/v1"
    assert provider.api_protocol == "openai_images"

    opener = HandlerOpener(lambda request: json_response({"data": [{"b64_json": png_b64(4, 4)}]}))
    install_client(opener)
    job_kwargs = {
        "provider": provider,
        "prompt": ["a cat"],
        "size": ["auto"],
        "custom_width": [1024],
        "custom_height": [1024],
        "background": ["不发送"],
        "quality": ["low"],
        "output_format": ["不发送"],
        "output_compression": [-1],
        "moderation": ["不发送"],
        "response_format": ["不发送"],
        "n": [1],
        "seed": [-1],
        "max_concurrency": [2],
    }
    images, texts, request_log, response_log = nodes.CustomEndpointGPTImage().execute(**job_kwargs)

    # the real key was used on the wire ...
    assert opener.calls[0].headers["Authorization"] == f"Bearer {SECRET}"
    assert tuple(images[0].shape) == (1, 4, 4, 3)
    # ... and the log still hides it
    assert SECRET not in request_log[0]
    assert SECRET not in response_log[0]
    assert json.loads(request_log[0])["headers"]["Authorization"] == "[REDACTED]"


def test_second_queue_of_the_same_graph_works(install_client):
    """The frontend re-sends the raw key each queue, so the guard re-issues handles."""
    from custom_endpoint import routes

    install_client(HandlerOpener(lambda request: json_response({"data": [{"b64_json": png_b64(4, 4)}]})))
    for _ in range(2):
        guarded = routes.guard_prompt_payload(_frontend_payload())
        provider, = nodes.CustomEndpointProvider().build(**guarded["prompt"]["1"]["inputs"])
        images, texts, request_log, response_log = nodes.CustomEndpointGPTImage().execute(
            provider=provider,
            prompt=["x"],
            size=["auto"],
            custom_width=[1024],
            custom_height=[1024],
            quality=["low"],
            n=[1],
            seed=[-1],
            max_concurrency=[1],
        )
        assert tuple(images[0].shape) == (1, 4, 4, 3)
