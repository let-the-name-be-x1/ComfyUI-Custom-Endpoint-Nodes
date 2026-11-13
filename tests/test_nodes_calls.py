"""End-to-end behaviour of the call nodes over a fake transport."""

import json
import threading
import time

import pytest

import ce_nodes as nodes
from conftest import (
    HandlerOpener,
    QueueOpener,
    RawResponse,
    content_box,
    image_tensor,
    json_response,
    mask_tensor,
    png_b64,
    png_bytes,
)

SECRET = "sk-test-1234567890"
LINK_TYPES = ("IMAGE", "MASK", "CUSTOM_ENDPOINT_PROVIDER", "CUSTOM_ENDPOINT_FILES")


def widget_defaults(cls):
    """Mimic ComfyUI: every widget arrives wrapped in a one-element list."""
    spec = cls.INPUT_TYPES()
    values = {}
    for section in ("required", "optional"):
        for name, definition in spec.get(section, {}).items():
            if definition[0] in LINK_TYPES:
                continue
            values[name] = [definition[1].get("default")]
    return values


def make_provider(**overrides):
    kwargs = {
        "endpoint": "https://api.example.com",
        "api_key": SECRET,
        "model": "gpt-image-2.5-flare",
        "api_protocol": "openai_images",
    }
    kwargs.update(overrides)
    return nodes.CustomEndpointProvider().build(**kwargs)[0]


def image_reply(request):
    return json_response({"data": [{"b64_json": png_b64(4, 4)}]})


# ---------------------------------------------------------------------------
# provider node
# ---------------------------------------------------------------------------
def test_provider_node_builds_a_config():
    provider, = nodes.CustomEndpointProvider().build(
        endpoint="https://api.example.com",
        api_key=SECRET,
        model="gpt-image-2.5-flare",
        api_protocol="auto",
    )
    assert provider.api_protocol == "openai_images"
    assert provider.base_url == "https://api.example.com/v1"


def test_provider_node_never_raises_on_empty_config():
    provider, = nodes.CustomEndpointProvider().build(
        endpoint="", api_key="", model="", api_protocol="auto"
    )
    assert provider.config_error


# ---------------------------------------------------------------------------
# GPT image
# ---------------------------------------------------------------------------
def test_gpt_image_generate(install_client):
    opener = HandlerOpener(image_reply)
    install_client(opener)
    node = nodes.CustomEndpointGPTImage()
    kwargs = widget_defaults(nodes.CustomEndpointGPTImage)
    kwargs["provider"] = make_provider()
    kwargs["prompt"] = ["a cat"]

    images, texts, request_log, response_log = node.execute(**kwargs)

    assert len(images) == 1 and len(texts) == 1
    assert tuple(images[0].shape) == (1, 4, 4, 3)
    body = json.loads(request_log[0])
    assert body["method"] == "POST"
    assert body["url"].endswith("/v1/images/generations")
    assert body["body"]["prompt"] == "a cat"
    assert "不发送" not in request_log[0]
    response = json.loads(response_log[0])
    assert response["ok"] is True
    assert response["http_status"] == 200
    assert response["parsed"]["image_count"] == 1


def test_gpt_image_edit_with_mask_uses_multipart(install_client):
    opener = HandlerOpener(image_reply)
    install_client(opener)
    node = nodes.CustomEndpointGPTImage()
    kwargs = widget_defaults(nodes.CustomEndpointGPTImage)
    kwargs["provider"] = make_provider()
    kwargs["prompt"] = ["repaint the sky"]
    kwargs["image_1"] = [image_tensor(1, 8, 8)]
    kwargs["mask"] = [mask_tensor(1, 8, 8, 0.0)]

    images, texts, request_log, response_log = node.execute(**kwargs)

    assert tuple(images[0].shape) == (1, 4, 4, 3)
    request = opener.calls[0]
    assert request.full_url.endswith("/v1/images/edits")
    assert b'name="image"' in request.data
    assert b'name="mask"' in request.data
    assert b'filename="mask.png"' in request.data
    body = json.loads(request_log[0])
    assert body["operation"] == "edit"
    assert body["body"]["_files"] == ["image", "mask"]


def test_gpt_image_batch_runs_every_prompt(install_client):
    states = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def handler(request):
        with lock:
            states["active"] += 1
            states["peak"] = max(states["peak"], states["active"])
        time.sleep(0.05)
        with lock:
            states["active"] -= 1
        return image_reply(request)

    opener = HandlerOpener(handler)
    install_client(opener)
    node = nodes.CustomEndpointGPTImage()
    kwargs = widget_defaults(nodes.CustomEndpointGPTImage)
    kwargs["provider"] = make_provider()
    kwargs["prompt"] = ["p1", "p2", "p3"]
    kwargs["max_concurrency"] = [3]

    images, texts, request_log, response_log = node.execute(**kwargs)

    assert len(opener.calls) == 3
    assert len(images) == 3 and len(request_log) == 3 and len(response_log) == 3
    prompts = [json.loads(entry)["body"]["prompt"] for entry in request_log]
    assert prompts == ["p1", "p2", "p3"]
    assert states["peak"] >= 2, "batch was not dispatched concurrently"


def test_one_failed_item_does_not_break_the_batch(install_client):
    def handler(request):
        payload = json.loads(request.data.decode("utf-8"))
        if payload["prompt"] == "bad":
            return json_response({"error": {"message": "upstream exploded"}}, status=500)
        return image_reply(request)

    install_client(HandlerOpener(handler))
    node = nodes.CustomEndpointGPTImage()
    kwargs = widget_defaults(nodes.CustomEndpointGPTImage)
    kwargs["provider"] = make_provider(retries=0)
    kwargs["prompt"] = ["ok1", "bad", "ok2"]

    images, texts, request_log, response_log = node.execute(**kwargs)

    assert tuple(images[0].shape) == (1, 4, 4, 3)
    assert tuple(images[1].shape) == (1, 1, 1, 3)
    assert tuple(images[2].shape) == (1, 4, 4, 3)
    assert "upstream exploded" in texts[1]
    assert json.loads(response_log[1])["ok"] is False
    assert json.loads(response_log[1])["parsed"]["error"] == "upstream exploded"
    # the request log still records what was sent
    assert json.loads(request_log[1])["body"]["prompt"] == "bad"


def test_network_failure_is_reported_as_data(install_client):
    install_client(QueueOpener(OSError("connection refused")))
    node = nodes.CustomEndpointGPTImage()
    kwargs = widget_defaults(nodes.CustomEndpointGPTImage)
    kwargs["provider"] = make_provider(retries=0)
    kwargs["prompt"] = ["x"]

    images, texts, request_log, response_log = node.execute(**kwargs)

    assert tuple(images[0].shape) == (1, 1, 1, 3)
    assert texts[0]
    assert json.loads(response_log[0])["ok"] is False


def test_reference_slots_must_be_gap_free(install_client):
    install_client(HandlerOpener(image_reply))
    node = nodes.CustomEndpointGPTImage()
    kwargs = widget_defaults(nodes.CustomEndpointGPTImage)
    kwargs["provider"] = make_provider()
    kwargs["prompt"] = ["x"]
    kwargs["image_1"] = None
    kwargs["image_2"] = [image_tensor(1, 8, 8)]

    images, texts, request_log, response_log = node.execute(**kwargs)

    assert tuple(images[0].shape) == (1, 1, 1, 3)
    payload = json.loads(response_log[0])
    assert payload["ok"] is False
    assert "without gaps" in json.dumps(payload)


def test_mask_without_image_is_rejected(install_client):
    install_client(HandlerOpener(image_reply))
    node = nodes.CustomEndpointGPTImage()
    kwargs = widget_defaults(nodes.CustomEndpointGPTImage)
    kwargs["provider"] = make_provider()
    kwargs["prompt"] = ["x"]
    kwargs["mask"] = [mask_tensor(1, 8, 8, 0.0)]

    images, texts, request_log, response_log = node.execute(**kwargs)
    assert tuple(images[0].shape) == (1, 1, 1, 3)
    assert "mask" in json.dumps(json.loads(response_log[0])).lower()


def test_missing_provider_is_reported_not_raised(install_client):
    install_client(QueueOpener())
    node = nodes.CustomEndpointGPTImage()
    kwargs = widget_defaults(nodes.CustomEndpointGPTImage)
    kwargs["provider"] = None
    kwargs["prompt"] = ["x"]

    images, texts, request_log, response_log = node.execute(**kwargs)
    assert tuple(images[0].shape) == (1, 1, 1, 3)
    assert texts[0]
    assert json.loads(response_log[0])["ok"] is False


def test_protocol_mismatch_points_at_the_provider_node(install_client):
    install_client(QueueOpener())
    node = nodes.CustomEndpointGeminiImage()
    kwargs = widget_defaults(nodes.CustomEndpointGeminiImage)
    kwargs["provider"] = make_provider()  # openai provider into a gemini node
    kwargs["prompt"] = ["x"]

    images, thought, texts, request_log, response_log = node.execute(**kwargs)
    assert tuple(images[0].shape) == (1, 1, 1, 3)
    assert "protocol_mismatch" in response_log[0]


def test_session_handle_is_resolved_for_every_batch_item(install_client):
    from custom_endpoint import secrets

    handle = secrets.issue_session_credential(SECRET)
    opener = HandlerOpener(image_reply)
    install_client(opener)
    node = nodes.CustomEndpointGPTImage()
    kwargs = widget_defaults(nodes.CustomEndpointGPTImage)
    kwargs["provider"] = make_provider(api_key=handle)
    kwargs["prompt"] = ["p1", "p2", "p3"]

    images, texts, request_log, response_log = node.execute(**kwargs)

    assert len(images) == 3
    assert all(json.loads(entry)["ok"] for entry in response_log)
    assert all(
        call.headers.get("Authorization") == f"Bearer {SECRET}" for call in opener.calls
    )


# ---------------------------------------------------------------------------
# Gemini image
# ---------------------------------------------------------------------------
def test_gemini_image_returns_image_text_and_thought_image(install_client):
    def handler(request):
        assert b":generateContent" in request.full_url.encode()
        return json_response(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "done"},
                                {"inlineData": {"mimeType": "image/png", "data": png_b64(6, 6)}},
                                {"thought": True, "inlineData": {"mimeType": "image/png", "data": png_b64(2, 2)}},
                            ]
                        }
                    }
                ]
            }
        )

    opener = HandlerOpener(handler)
    install_client(opener)
    node = nodes.CustomEndpointGeminiImage()
    kwargs = widget_defaults(nodes.CustomEndpointGeminiImage)
    kwargs["provider"] = make_provider(
        model="gemini-3.1-flash-image", api_protocol="gemini"
    )
    kwargs["prompt"] = ["a fox"]
    kwargs["thinking_level"] = ["HIGH"]
    kwargs["resolution"] = ["2K"]

    images, thought, texts, request_log, response_log = node.execute(**kwargs)

    assert tuple(images[0].shape) == (1, 6, 6, 3)
    assert tuple(thought[0].shape) == (1, 2, 2, 3)
    assert texts[0] == "done"
    request_body = json.loads(opener.calls[0].data.decode("utf-8"))
    assert request_body["generationConfig"]["imageConfig"]["imageSize"] == "2K"
    assert request_body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "HIGH"}
    assert json.loads(request_log[0])["url"].endswith(":generateContent")


def test_gemini_block_reason_is_reported(install_client):
    install_client(HandlerOpener(lambda request: json_response({"promptFeedback": {"blockReason": "SAFETY"}})))
    node = nodes.CustomEndpointGeminiImage()
    kwargs = widget_defaults(nodes.CustomEndpointGeminiImage)
    kwargs["provider"] = make_provider(model="gemini-3.1-flash-image", api_protocol="gemini")
    kwargs["prompt"] = ["x"]

    images, thought, texts, request_log, response_log = node.execute(**kwargs)
    assert tuple(images[0].shape) == (1, 1, 1, 3)
    assert "SAFETY" in response_log[0]


# ---------------------------------------------------------------------------
# Grok image
# ---------------------------------------------------------------------------
def test_grok_image_generate_body(install_client):
    opener = HandlerOpener(image_reply)
    install_client(opener)
    node = nodes.CustomEndpointGrokImage()
    kwargs = widget_defaults(nodes.CustomEndpointGrokImage)
    kwargs["provider"] = make_provider(model="grok-imagine-image-2.0", api_protocol="grok")
    kwargs["prompt"] = ["a rocket"]
    kwargs["number_of_images"] = [2]
    kwargs["aspect_ratio"] = ["16:9"]
    kwargs["resolution"] = ["2K"]

    images, texts, request_log, response_log = node.execute(**kwargs)

    assert tuple(images[0].shape) == (1, 4, 4, 3)
    body = json.loads(opener.calls[0].data.decode("utf-8"))
    assert body["n"] == 2
    assert body["aspect_ratio"] == "16:9"
    assert body["resolution"] == "2k"
    assert opener.calls[0].full_url.endswith("/v1/images/generations")


# ---------------------------------------------------------------------------
# Qwen image
# ---------------------------------------------------------------------------
def qwen_handler(request):
    """Serve the generation POST, then the signed-URL image download."""
    if request.get_method() == "GET" or "cdn.example.com" in request.full_url:
        return RawResponse(200, png_bytes(6, 6))
    return json_response(
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"image": "https://cdn.example.com/result.png"},
                                {"text": "ok"},
                            ],
                        },
                    }
                ]
            },
            "request_id": "req-qwen-1",
        }
    )


def test_qwen_image_generate(install_client):
    opener = HandlerOpener(qwen_handler)
    install_client(opener)
    node = nodes.CustomEndpointQwenImage()
    kwargs = widget_defaults(nodes.CustomEndpointQwenImage)
    kwargs["provider"] = make_provider(model="qwen-image-3.0-pro", api_protocol="qwen")
    kwargs["prompt"] = ["一只在雪地里的狐狸"]
    kwargs["n"] = [2]

    images, texts, request_log, response_log = node.execute(**kwargs)

    assert tuple(images[0].shape) == (1, 6, 6, 3)
    assert texts[0] == "ok"
    post = opener.calls[0]
    assert post.full_url.endswith("/v1/services/aigc/multimodal-generation/generation")
    body = json.loads(post.data.decode("utf-8"))
    assert body["model"] == "qwen-image-3.0-pro"
    assert body["parameters"]["size"] == "1024*1024"
    assert body["parameters"]["n"] == 2
    assert body["parameters"]["seed"] == 42
    assert body["input"]["messages"][0]["content"] == [{"text": "一只在雪地里的狐狸"}]
    log = json.loads(request_log[0])
    assert log["operation"] == "generate"
    assert "不发送" not in request_log[0]
    response = json.loads(response_log[0])
    assert response["ok"] is True
    assert response["parsed"]["image_count"] == 1


def test_qwen_image_edit_uses_reference_slots(install_client):
    opener = HandlerOpener(qwen_handler)
    install_client(opener)
    node = nodes.CustomEndpointQwenImageEdit()
    kwargs = widget_defaults(nodes.CustomEndpointQwenImageEdit)
    kwargs["provider"] = make_provider(model="qwen-image-3.0", api_protocol="qwen")
    kwargs["prompt"] = ["把 @Image1 的裙子换成蓝色"]
    kwargs["image_1"] = [image_tensor(1, 8, 8)]

    images, texts, request_log, response_log = node.execute(**kwargs)

    assert tuple(images[0].shape) == (1, 6, 6, 3)
    body = json.loads(opener.calls[0].data.decode("utf-8"))
    content = body["input"]["messages"][0]["content"]
    assert content[0]["image"].startswith("data:image/png;base64,")
    assert content[1] == {"text": "把 Image 1 的裙子换成蓝色"}
    # 8x8 is below the 512x512 floor, so "match input" is scaled into the window
    assert body["parameters"]["size"] == "512*512"
    log = json.loads(request_log[0])
    assert log["operation"] == "edit"
    assert len(log["images"]["reference_images"]) == 1
    assert log["images"]["reference_images"][0] > 0
    assert json.loads(response_log[0])["ok"] is True


def test_qwen_edit_without_a_reference_image_fails_soft(install_client):
    install_client(HandlerOpener(qwen_handler))
    node = nodes.CustomEndpointQwenImageEdit()
    kwargs = widget_defaults(nodes.CustomEndpointQwenImageEdit)
    kwargs["provider"] = make_provider(model="qwen-image-3.0", api_protocol="qwen")
    kwargs["prompt"] = ["x"]

    images, texts, request_log, response_log = node.execute(**kwargs)
    assert tuple(images[0].shape) == (1, 1, 1, 3)
    assert "missing_image" in response_log[0]


def test_qwen_node_rejects_another_family_provider(install_client):
    install_client(QueueOpener())
    node = nodes.CustomEndpointQwenImage()
    kwargs = widget_defaults(nodes.CustomEndpointQwenImage)
    kwargs["provider"] = make_provider()  # openai_images provider
    kwargs["prompt"] = ["x"]

    images, texts, request_log, response_log = node.execute(**kwargs)
    assert tuple(images[0].shape) == (1, 1, 1, 3)
    assert "protocol_mismatch" in response_log[0]


def test_qwen_image_edit_on_a_relay_surface(install_client):
    """qwen_images speaks /images/edits with a top-level data URL."""
    opener = HandlerOpener(qwen_handler)
    install_client(opener)
    node = nodes.CustomEndpointQwenImageEdit()
    kwargs = widget_defaults(nodes.CustomEndpointQwenImageEdit)
    kwargs["provider"] = make_provider(
        endpoint="https://relay.example.com", model="qwen-image-3.0-pro", api_protocol="qwen_images"
    )
    kwargs["prompt"] = ["把 @Image1 的手修好"]
    kwargs["image_1"] = [image_tensor(1, 8, 8)]

    images, texts, request_log, response_log = node.execute(**kwargs)

    assert tuple(images[0].shape) == (1, 6, 6, 3)
    post = opener.calls[0]
    assert post.full_url == "https://relay.example.com/v1/images/edits"
    body = json.loads(post.data.decode("utf-8"))
    assert body["image"].startswith("data:image/png;base64,")
    assert body["prompt"] == "把 Image 1 的手修好"
    assert body["size"] == "512*512"  # 8x8 fitted into the window
    assert json.loads(response_log[0])["ok"] is True


def test_qwen_native_404_falls_back_to_the_relay_surface(install_client):
    """A gateway that 404s the DashScope path is retried once on /images/edits."""
    calls = []

    def handler(request):
        calls.append(request)
        if "/multimodal-generation/generation" in request.full_url:
            return json_response(
                {
                    "error": {
                        "message": "Invalid URL (POST /v1/services/aigc/multimodal-generation/generation)",
                        "type": "invalid_request_error",
                        "param": "",
                        "code": "",
                    }
                },
                status=404,
            )
        return qwen_handler(request)

    opener = HandlerOpener(handler)
    install_client(opener)
    node = nodes.CustomEndpointQwenImageEdit()
    kwargs = widget_defaults(nodes.CustomEndpointQwenImageEdit)
    kwargs["provider"] = make_provider(
        endpoint="https://relay.example.com", model="qwen-image-3.0-pro", api_protocol="qwen"
    )
    kwargs["prompt"] = ["修一下手"]
    kwargs["image_1"] = [image_tensor(1, 8, 8)]

    images, texts, request_log, response_log = node.execute(**kwargs)

    assert tuple(images[0].shape) == (1, 6, 6, 3), texts[0]
    urls = [call.full_url for call in opener.calls]
    assert urls[0].endswith("/v1/services/aigc/multimodal-generation/generation")
    assert urls[1].endswith("/v1/images/edits")
    log = json.loads(request_log[0])
    assert log["extra"]["fallback"]["first_status"] == 404
    assert log["url"].endswith("/v1/images/edits")
    response = json.loads(response_log[0])
    assert response["ok"] is True
    assert response["extra"]["fallback"]["first_status"] == 404


def test_qwen_other_404_is_not_retried_on_another_surface(install_client):
    def handler(request):
        return json_response({"error": {"message": "model not found"}}, status=404)

    opener = HandlerOpener(handler)
    install_client(opener)
    node = nodes.CustomEndpointQwenImage()
    kwargs = widget_defaults(nodes.CustomEndpointQwenImage)
    kwargs["provider"] = make_provider(
        endpoint="https://relay.example.com", model="qwen-image-9", api_protocol="qwen", retries=0
    )
    kwargs["prompt"] = ["x"]

    images, texts, request_log, response_log = node.execute(**kwargs)
    assert tuple(images[0].shape) == (1, 1, 1, 3)
    assert len(opener.calls) == 1, "only a 'route missing' 404 may switch surfaces"
    assert "model not found" in response_log[0]


# ---------------------------------------------------------------------------
# LLM nodes
# ---------------------------------------------------------------------------
def test_gpt_chat_returns_text(install_client):
    opener = HandlerOpener(
        lambda request: json_response({"choices": [{"message": {"content": "hello there"}}]})
    )
    install_client(opener)
    node = nodes.CustomEndpointGPTChat()
    kwargs = widget_defaults(nodes.CustomEndpointGPTChat)
    kwargs["provider"] = make_provider(model="gpt-5.5", api_protocol="openai_chat")
    kwargs["prompt"] = ["hi", "hello"]

    texts, request_log, response_log = node.execute(**kwargs)

    assert texts == ["hello there", "hello there"]
    assert len(opener.calls) == 2
    assert json.loads(request_log[0])["url"].endswith("/v1/chat/completions")


def test_gemini_chat_returns_text(install_client):
    opener = HandlerOpener(
        lambda request: json_response({"candidates": [{"content": {"parts": [{"text": "答案"}]}}]})
    )
    install_client(opener)
    node = nodes.CustomEndpointGeminiChat()
    kwargs = widget_defaults(nodes.CustomEndpointGeminiChat)
    kwargs["provider"] = make_provider(model="gemini-3.1-pro-preview", api_protocol="gemini")
    kwargs["prompt"] = ["问题"]

    texts, request_log, response_log = node.execute(**kwargs)
    assert texts == ["答案"]


def test_grok_chat_returns_text(install_client):
    opener = HandlerOpener(lambda request: json_response({"choices": [{"message": {"content": "yo"}}]}))
    install_client(opener)
    node = nodes.CustomEndpointGrokChat()
    kwargs = widget_defaults(nodes.CustomEndpointGrokChat)
    kwargs["provider"] = make_provider(model="grok-4", api_protocol="grok")
    kwargs["prompt"] = ["hi"]

    texts, request_log, response_log = node.execute(**kwargs)
    assert texts == ["yo"]


# ---------------------------------------------------------------------------
# auxiliary nodes
# ---------------------------------------------------------------------------
def test_prompt_list_splits_lines():
    prompts, = nodes.CustomEndpointPromptList().split("a\n\n  b  \nc", True, True)
    assert prompts == ["a", "b", "c"]
    keep, = nodes.CustomEndpointPromptList().split("a\n\n \nc", False, False)
    assert keep == ["a", "", " ", "c"]


def test_batch_images_returns_a_padded_batch_and_the_raw_list():
    merged, raw = nodes.CustomEndpointBatchImages().merge(
        image_tensor(1, 8, 8), image_2=image_tensor(2, 8, 8)
    )
    assert tuple(merged.shape) == (3, 8, 8, 3)
    assert [tuple(frame.shape) for frame in raw] == [(1, 8, 8, 3)] * 3


def test_batch_images_never_stretches_a_different_aspect_ratio():
    wide = image_tensor(1, 4, 8, value=0.25)  # 2:1 landscape
    merged, raw = nodes.CustomEndpointBatchImages().merge(
        image_tensor(1, 8, 8, value=0.75), image_2=wide
    )
    assert tuple(merged.shape) == (2, 8, 8, 3)  # canvas = biggest frame
    assert content_box(merged[1]) == (2, 6, 0, 8)  # letterboxed, aspect kept
    assert tuple(raw[1].shape) == (1, 4, 8, 3)  # the list keeps the native size


def test_batch_images_honours_the_canvas_widgets():
    node = nodes.CustomEndpointBatchImages()
    merged, _ = node.merge(
        image_tensor(1, 4, 8), image_2=image_tensor(1, 8, 4),
        size_mode="manual", width=12, height=6, size_multiple=4,
    )
    assert tuple(merged.shape) == (2, 8, 12, 3)  # 6 -> 8, 12 -> 12
    stretched, _ = node.merge(
        image_tensor(1, 4, 8), image_2=image_tensor(1, 8, 4), fit="stretch"
    )
    assert tuple(stretched.shape) == (2, 8, 8, 3)
    assert content_box(stretched[0]) == (0, 8, 0, 8)  # stretch fills the canvas


def test_batch_images_without_inputs_returns_the_placeholder():
    merged, raw = nodes.CustomEndpointBatchImages().merge(None)
    assert tuple(merged.shape) == (1, 1, 1, 3)
    assert len(raw) == 1


def test_file_loaders_tag_their_family(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_input_directory", lambda: str(tmp_path))
    monkeypatch.setattr(nodes, "_resolve_input_path", lambda name: str(tmp_path / name))
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")

    gpt_files, = nodes.CustomEndpointGPTFiles().load("a.txt")
    assert gpt_files["family"] == "openai"
    assert gpt_files["parts"][0]["file_data"].startswith("data:text/plain;base64,")

    gemini_files, = nodes.CustomEndpointGeminiFiles().load("a.txt")
    assert gemini_files["family"] == "gemini"
    assert gemini_files["parts"][0]["inlineData"]["mimeType"] == "text/plain"


def test_file_parts_reject_the_wrong_loader_family():
    job = {"files": {"family": "openai", "parts": []}}
    from custom_endpoint.errors import EndpointError

    with pytest.raises(EndpointError) as excinfo:
        nodes._files_parts(job, "gemini")
    assert excinfo.value.error_code == "file_family_mismatch"


def test_gpt_chat_sends_gemini_loader_files_as_an_error(install_client):
    install_client(HandlerOpener(image_reply))
    node = nodes.CustomEndpointGPTChat()
    kwargs = widget_defaults(nodes.CustomEndpointGPTChat)
    kwargs["provider"] = make_provider(model="gpt-5.5", api_protocol="openai_chat")
    kwargs["prompt"] = ["x"]
    kwargs["files"] = [{"family": "gemini", "parts": [{"inlineData": {}}]}]

    texts, request_log, response_log = node.execute(**kwargs)
    assert "file_family_mismatch" in response_log[0]
