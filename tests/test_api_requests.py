"""Wire-protocol shapes for the three provider families (no network)."""

import base64
import json

import pytest

from custom_endpoint import api_gemini, api_grok, api_openai, api_qwen
from custom_endpoint.api_common import ParsedResponse
from custom_endpoint.errors import EndpointError
from custom_endpoint.transport import HttpClient, HttpResponse
from conftest import QueueOpener, RawResponse, json_response, png_b64, png_bytes


def _provider(family, model=None, endpoint="https://api.example.com"):
    from custom_endpoint.provider import build_provider

    defaults = {
        "openai_images": "gpt-image-2.5-flare",
        "openai_chat": "gpt-5.5",
        "openai_responses": "gpt-5.5",
        "gemini": "gemini-3.1-flash-image",
        "grok": "grok-imagine-image-2.0",
        "qwen": "qwen-image-3.0",
        "qwen_images": "qwen-image-3.0",
    }
    return build_provider(
        endpoint=endpoint,
        api_key="sk-test-1234567890",
        model=model or defaults[family],
        api_protocol=family,
    )


def _client(*responses):
    opener = QueueOpener(*responses)
    return HttpClient(opener=opener, sleeper=lambda _: None), opener


def _body_of(request):
    if request.data:
        try:
            return json.loads(request.data.decode("utf-8"))
        except json.JSONDecodeError:
            return request.data.decode("utf-8", errors="replace")
    return None


# ---------------------------------------------------------------------------
# OpenAI images
# ---------------------------------------------------------------------------
def test_openai_generate_url_and_body():
    request = api_openai.prepare_image_request(
        _provider("openai_images"),
        operation="generate",
        prompt="a red cube",
        params={"size": "1024x1024", "quality": "high", "n": 2},
    )
    assert request.method == "POST"
    assert request.url == "https://api.example.com/v1/images/generations"
    assert request.json_body == {
        "model": "gpt-image-2.5-flare",
        "prompt": "a red cube",
        "size": "1024x1024",
        "quality": "high",
        "n": 2,
    }
    assert request.headers["Authorization"] == "Bearer sk-test-1234567890"


def test_openai_omits_unset_fields():
    request = api_openai.prepare_image_request(
        _provider("openai_images"), operation="generate", prompt="x", params={"n": 1}
    )
    assert set(request.json_body) == {"model", "prompt", "n"}


def test_openai_custom_size_is_validated():
    request = api_openai.prepare_image_request(
        _provider("openai_images"),
        operation="generate",
        prompt="x",
        params={"size": "Custom", "custom_width": 2048, "custom_height": 1152},
    )
    assert request.json_body["size"] == "2048x1152"

    for bad in ((1000, 1024), (3840, 1000), (1024, 1024 * 0 + 8000)):
        with pytest.raises(EndpointError) as excinfo:
            api_openai.prepare_image_request(
                _provider("openai_images"),
                operation="generate",
                prompt="x",
                params={"size": "Custom", "custom_width": bad[0], "custom_height": bad[1]},
            )
        assert excinfo.value.error_code == "invalid_size"


def test_openai_edit_is_multipart_with_image_and_mask():
    request = api_openai.prepare_image_request(
        _provider("openai_images"),
        operation="edit",
        prompt="repaint",
        images=[png_bytes(8, 8)],
        mask=png_bytes(8, 8, (0, 0, 0)),
        params={"size": "auto", "n": 1},
    )
    assert request.url.endswith("/images/edits")
    assert request.content_type.startswith("multipart/form-data")
    assert b'name="image"' in request.body
    assert b'name="mask"' in request.body
    assert b'filename="mask.png"' in request.body
    assert b'name="prompt"' in request.body


def test_openai_edit_uses_array_field_for_multiple_images():
    request = api_openai.prepare_image_request(
        _provider("openai_images"),
        operation="edit",
        prompt="p",
        images=[png_bytes(), png_bytes()],
        params={"n": 1},
    )
    assert request.body.count(b'name="image[]"') == 2
    assert b'name="image"' not in request.body


def test_openai_parse_b64_and_url_and_revised_prompt():
    client, opener = _client(
        json_response(
            {
                "data": [
                    {"b64_json": png_b64(4, 4), "revised_prompt": "better prompt"},
                    {"url": "https://cdn.example.com/a.png"},
                ]
            }
        ),
        RawResponse(200, png_bytes(6, 6)),
    )
    provider = _provider("openai_images")
    parsed = api_openai.parse_image_response(provider, client_response(client, opener), client, timeout=30)
    assert len(parsed.images) == 2
    assert parsed.revised_prompts == ["better prompt"]
    assert parsed.details["revised_prompt"] == "better prompt"
    assert opener.urls[1] == "https://cdn.example.com/a.png"
    # the download must not carry the API key
    assert "Authorization" not in opener.calls[1].headers


def client_response(client, opener):
    return client.request("POST", "https://api.example.com/v1/images/generations", json_body={})


def test_openai_parse_error_body_is_reported():
    client, opener = _client(
        json_response({"error": {"message": "quota exceeded", "code": "rate_limit"}}, status=429)
    )
    response = client_response(client, opener)
    with pytest.raises(EndpointError) as excinfo:
        api_openai.parse_image_response(_provider("openai_images"), response, client, timeout=30)
    assert "quota exceeded" in excinfo.value.message
    assert excinfo.value.status == 429
    assert excinfo.value.retryable is True


def test_openai_parse_empty_payload_raises():
    client, opener = _client(json_response({"data": []}))
    with pytest.raises(EndpointError) as excinfo:
        api_openai.parse_image_response(_provider("openai_images"), client_response(client, opener), client, timeout=30)
    assert excinfo.value.error_type == "empty_result"


# ---------------------------------------------------------------------------
# OpenAI text
# ---------------------------------------------------------------------------
def test_openai_chat_completions_shape():
    request = api_openai.prepare_chat_request(
        _provider("openai_chat"),
        prompt="hi",
        images=[png_bytes(4, 4)],
        params={"temperature": 0.5, "max_output_tokens": 64, "store": False},
    )
    assert request.url == "https://api.example.com/v1/chat/completions"
    body = request.json_body
    assert body["model"] == "gpt-5.5"
    assert body["stream"] is False
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "hi"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 64


def test_openai_chat_appends_txt_files_and_rejects_pdf():
    txt = base64.b64encode(b"context text").decode()
    files = [{"type": "input_file", "filename": "a.txt", "file_data": "data:text/plain;base64," + txt}]
    request = api_openai.prepare_chat_request(_provider("openai_chat"), prompt="q", files=files, params={})
    assert "context text" in request.json_body["messages"][0]["content"]

    pdf = [{"type": "input_file", "filename": "a.pdf", "file_data": "data:application/pdf;base64,AAAA"}]
    with pytest.raises(EndpointError) as excinfo:
        api_openai.prepare_chat_request(_provider("openai_chat"), prompt="q", files=pdf, params={})
    assert excinfo.value.error_code == "unsupported_file"


def test_openai_responses_shape_includes_files_and_reasoning():
    pdf = [{"type": "input_file", "filename": "a.pdf", "file_data": "data:application/pdf;base64,AAAA"}]
    request = api_openai.prepare_chat_request(
        _provider("openai_responses"),
        prompt="summarize",
        files=pdf,
        params={"instructions": "be terse", "max_output_tokens": 128, "reasoning_effort": "high",
                "truncation": "auto", "store": False},
    )
    assert request.url == "https://api.example.com/v1/responses"
    body = request.json_body
    assert body["store"] is False
    assert body["instructions"] == "be terse"
    assert body["max_output_tokens"] == 128
    assert body["reasoning"] == {"effort": "high"}
    assert body["truncation"] == "auto"
    types = [part["type"] for part in body["input"][0]["content"]]
    assert types == ["input_text", "input_file"]


def test_openai_chat_parse_text():
    client, opener = _client(
        json_response({"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]})
    )
    response = client.request("POST", "https://api.example.com/v1/chat/completions", json_body={})
    parsed = api_openai.parse_chat_response(_provider("openai_chat"), response)
    assert parsed.text == "hello"
    assert parsed.details["finish_reason"] == "stop"


def test_openai_responses_parse_text():
    client, opener = _client(
        json_response({"output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}]})
    )
    response = client.request("POST", "https://api.example.com/v1/responses", json_body={})
    parsed = api_openai.parse_chat_response(_provider("openai_responses"), response)
    assert parsed.text == "answer"


def test_openai_chat_parse_empty_is_an_error():
    client, _ = _client(json_response({"choices": []}))
    response = client.request("POST", "https://api.example.com/v1/chat/completions", json_body={})
    with pytest.raises(EndpointError):
        api_openai.parse_chat_response(_provider("openai_chat"), response)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def test_gemini_image_body_and_url():
    provider = _provider("gemini")
    request = api_gemini.prepare_image_request(
        provider,
        operation="generate",
        prompt="a fox",
        images=[png_bytes(4, 4)],
        params={
            "aspect_ratio": "16:9",
            "resolution": "2K",
            "thinking_level": "HIGH",
            "response_modalities": "IMAGE+TEXT",
            "system_prompt": "sys",
        },
    )
    assert request.url == "https://api.example.com/v1beta/models/gemini-3.1-flash-image:generateContent"
    body = request.json_body
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "a fox"}
    assert parts[1]["inlineData"]["mimeType"] == "image/png"
    assert request.headers["x-goog-api-key"] == "sk-test-1234567890"
    generation = body["generationConfig"]
    assert generation["responseModalities"] == ["TEXT", "IMAGE"]
    assert generation["imageConfig"] == {"imageSize": "2K", "aspectRatio": "16:9"}
    assert generation["thinkingConfig"] == {"thinkingLevel": "HIGH"}
    assert body["systemInstruction"] == {"parts": [{"text": "sys"}]}


def test_gemini_system_instruction_passes_through_objects():
    request = api_gemini.prepare_image_request(
        _provider("gemini"),
        operation="generate",
        prompt="x",
        params={"system_prompt": '{"parts":[{"text":"nested"}],"role":null}'},
    )
    assert request.json_body["systemInstruction"]["parts"] == [{"text": "nested"}]


def test_gemini_auto_aspect_ratio_is_omitted():
    request = api_gemini.prepare_image_request(
        _provider("gemini"), operation="generate", prompt="x", params={"aspect_ratio": "auto", "resolution": "1K"}
    )
    image_config = request.json_body["generationConfig"]["imageConfig"]
    assert "aspectRatio" not in image_config
    assert image_config["imageSize"] == "1K"


def test_gemini_mask_is_rejected_with_a_hint():
    with pytest.raises(EndpointError) as excinfo:
        api_gemini.prepare_image_request(
            _provider("gemini"), operation="generate", prompt="x", mask=png_bytes()
        )
    assert excinfo.value.error_code == "mask_unsupported"


def test_gemini_parse_images_thoughts_text_and_block():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "here you go"},
                        {"inlineData": {"mimeType": "image/png", "data": png_b64(5, 5)}},
                        {"thought": True, "inlineData": {"mimeType": "image/png", "data": png_b64(3, 3)}},
                        {"thought": True, "text": "thinking..."},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"totalTokenCount": 12},
    }
    client, opener = _client(json_response(payload))
    response = client.request("POST", "https://api.example.com/v1beta/models/x:generateContent", json_body={})
    provider = _provider("gemini")
    parsed = api_gemini.parse_response(provider, response, client, timeout=30, want_images=True)
    assert len(parsed.images) == 1
    assert len(parsed.thought_images) == 1
    assert parsed.text == "here you go"
    assert parsed.details["thought_text"] == ["thinking..."]
    assert parsed.details["usage"]["totalTokenCount"] == 12


def test_gemini_blocked_response_is_an_error_with_the_reason():
    client, opener = _client(
        json_response({"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "nope"}})
    )
    response = client.request("POST", "https://api.example.com/v1beta/models/x:generateContent", json_body={})
    with pytest.raises(EndpointError) as excinfo:
        api_gemini.parse_response(_provider("gemini"), response, client, timeout=30, want_images=True)
    assert excinfo.value.error_code == "SAFETY"
    assert "SAFETY" in excinfo.value.message


def test_gemini_chat_uses_generate_content_without_image_config():
    request = api_gemini.prepare_chat_request(
        _provider("gemini"),
        prompt="hi",
        params={"max_output_tokens": 512, "temperature": 0.2, "thinking_level": "LOW"},
    )
    body = request.json_body
    generation = body["generationConfig"]
    assert generation["maxOutputTokens"] == 512
    assert generation["temperature"] == 0.2
    assert "imageConfig" not in generation
    assert generation["thinkingConfig"] == {"thinkingLevel": "LOW"}


def test_gemini_chat_parse_text_only():
    client, _ = _client(json_response({"candidates": [{"content": {"parts": [{"text": "answer"}]}}]}))
    response = client.request("POST", "https://api.example.com/v1beta/models/x:generateContent", json_body={})
    parsed = api_gemini.parse_response(_provider("gemini"), response, client, timeout=30, want_images=False)
    assert parsed.text == "answer"
    assert parsed.images == []


# ---------------------------------------------------------------------------
# Grok
# ---------------------------------------------------------------------------
def test_grok_generate_body():
    request = api_grok.prepare_image_request(
        _provider("grok"),
        operation="generate",
        prompt="a rocket",
        params={"aspect_ratio": "16:9", "resolution": "2K", "quality": "medium", "n": 2, "seed": 7},
    )
    assert request.url == "https://api.example.com/v1/images/generations"
    body = request.json_body
    assert body["model"] == "grok-imagine-image-2.0"
    assert body["aspect_ratio"] == "16:9"
    assert body["resolution"] == "2k"
    assert body["quality"] == "medium"
    assert body["n"] == 2
    assert body["seed"] == 7
    assert "images" not in body


def test_grok_edit_sends_data_urls():
    request = api_grok.prepare_image_request(
        _provider("grok"),
        operation="edit",
        prompt="make it blue",
        images=[png_bytes(4, 4), png_bytes(4, 4)],
        params={"resolution": "1K", "n": 1, "aspect_ratio": "16:9"},
    )
    assert request.url == "https://api.example.com/v1/images/edits"
    assert len(request.json_body["images"]) == 2
    assert request.json_body["images"][0]["url"].startswith("data:image/png;base64,")


def test_grok_pro_model_accepts_one_image_only():
    with pytest.raises(EndpointError) as excinfo:
        api_grok.prepare_image_request(
            _provider("grok", model="grok-imagine-image-pro"),
            operation="edit",
            prompt="p",
            images=[png_bytes(), png_bytes()],
            params={"n": 1},
        )
    assert excinfo.value.error_code == "too_many_images"


def test_grok_single_image_rejects_a_custom_aspect_ratio():
    with pytest.raises(EndpointError) as excinfo:
        api_grok.prepare_image_request(
            _provider("grok"),
            operation="edit",
            prompt="p",
            images=[png_bytes()],
            params={"n": 1, "aspect_ratio": "16:9"},
        )
    assert excinfo.value.error_code == "invalid_aspect_ratio"


def test_grok_edit_requires_an_image():
    with pytest.raises(EndpointError) as excinfo:
        api_grok.prepare_image_request(_provider("grok"), operation="edit", prompt="p", params={"n": 1})
    assert excinfo.value.error_code == "missing_image"


def test_grok_mask_is_rejected():
    with pytest.raises(EndpointError) as excinfo:
        api_grok.prepare_image_request(_provider("grok"), operation="generate", prompt="p", mask=png_bytes())
    assert excinfo.value.error_code == "mask_unsupported"


def test_grok_parse_images_and_cost():
    client, _ = _client(
        json_response(
            {
                "data": [{"b64_json": png_b64(4, 4), "revised_prompt": "revised"}],
                "usage": {"cost_in_usd_ticks": 123},
            }
        )
    )
    response = client.request("POST", "https://api.example.com/v1/images/generations", json_body={})
    parsed = api_grok.parse_image_response(_provider("grok"), response, client, timeout=30)
    assert len(parsed.images) == 1
    assert parsed.details["cost_in_usd_ticks"] == 123


def test_grok_chat_shape_and_parse():
    request = api_grok.prepare_chat_request(
        _provider("grok", model="grok-4"),
        prompt="hi",
        params={"system_prompt": "be brief", "temperature": 0.1, "max_tokens": 32},
    )
    assert request.url == "https://api.example.com/v1/chat/completions"
    body = request.json_body
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["messages"][1]["content"] == "hi"
    assert body["max_tokens"] == 32

    client, _ = _client(json_response({"choices": [{"message": {"content": "hey"}}]}))
    response = client.request("POST", "https://api.example.com/v1/chat/completions", json_body={})
    parsed = api_grok.parse_chat_response(_provider("grok", model="grok-4"), response)
    assert parsed.text == "hey"


def test_parsed_response_defaults():
    parsed = ParsedResponse()
    assert parsed.has_images is False
    assert parsed.images == [] and parsed.text == ""


# ---------------------------------------------------------------------------
# Qwen-Image (DashScope multimodal-generation)
# ---------------------------------------------------------------------------
def test_qwen_generate_body_and_url():
    request = api_qwen.prepare_image_request(
        _provider("qwen"),
        operation="generate",
        prompt="一只在雪地里的狐狸",
        params={
            "width": 1328,
            "height": 1328,
            "n": 2,
            "seed": 7,
            "prompt_extend": True,
            "watermark": False,
            "negative_prompt": "模糊",
        },
    )
    assert request.method == "POST"
    assert request.url == (
        "https://api.example.com/v1/services/aigc/multimodal-generation/generation"
    )
    assert request.headers["Authorization"] == "Bearer sk-test-1234567890"
    body = request.json_body
    assert body["model"] == "qwen-image-3.0"
    assert body["input"]["messages"][0]["role"] == "user"
    assert body["input"]["messages"][0]["content"] == [{"text": "一只在雪地里的狐狸"}]
    assert body["parameters"] == {
        "size": "1328*1328",
        "n": 2,
        "seed": 7,
        "prompt_extend": True,
        "watermark": False,
        "negative_prompt": "模糊",
    }


def test_qwen_generate_omits_an_empty_negative_prompt():
    request = api_qwen.prepare_image_request(
        _provider("qwen"),
        operation="generate",
        prompt="x",
        params={"width": 1024, "height": 1024, "n": 1, "seed": 42, "negative_prompt": ""},
    )
    assert "negative_prompt" not in request.json_body["parameters"]
    assert request.json_body["parameters"]["prompt_extend"] is True
    assert request.json_body["parameters"]["watermark"] is False


def test_qwen_rejects_a_prompt_outside_the_area_window():
    for width, height in ((512, 512), (2560, 2560), (1024, 1536), (2048, 1152)):
        request = api_qwen.prepare_image_request(
            _provider("qwen"),
            operation="generate",
            prompt="x",
            params={"width": width, "height": height, "n": 1, "seed": 1},
        )
        assert request.json_body["parameters"]["size"] == f"{width}*{height}"

    for width, height in ((256, 256), (2560, 2560 + 16), (512, 256)):
        with pytest.raises(EndpointError) as excinfo:
            api_qwen.prepare_image_request(
                _provider("qwen"),
                operation="generate",
                prompt="x",
                params={"width": width, "height": height, "n": 1, "seed": 1},
            )
        assert excinfo.value.error_code == "invalid_size"


def test_qwen_rejects_an_extreme_aspect_ratio():
    with pytest.raises(EndpointError) as excinfo:
        api_qwen.prepare_image_request(
            _provider("qwen"),
            operation="generate",
            prompt="x",
            params={"width": 2560, "height": 256, "n": 1, "seed": 1},
        )
    assert excinfo.value.error_code == "invalid_size"


def test_qwen_edit_sends_images_first_then_text():
    request = api_qwen.prepare_image_request(
        _provider("qwen"),
        operation="edit",
        prompt="把 @Image1 的裙子换成蓝色，保持 @Image2 的背景",
        images=[png_bytes(1000, 600), png_bytes(600, 600)],
        params={"size_mode": "match input", "n": 1, "seed": 3},
    )
    content = request.json_body["input"]["messages"][0]["content"]
    assert [list(item) for item in content] == [["image"], ["image"], ["text"]]
    assert content[0]["image"].startswith("data:image/png;base64,")
    assert content[2]["text"] == "把 Image 1 的裙子换成蓝色，保持 Image 2 的背景"
    # match input: the first reference image's own size, already inside the window
    assert request.json_body["parameters"]["size"] == "1000*600"
    assert request.operation == "edit"


def test_qwen_edit_match_input_grows_tiny_images_into_the_window():
    request = api_qwen.prepare_image_request(
        _provider("qwen"),
        operation="edit",
        prompt="p",
        images=[png_bytes(8, 8)],
        params={"size_mode": "match input", "n": 1, "seed": 1},
    )
    # 8x8 = 64 px is below 512x512, so it is scaled up to exactly 512x512
    assert request.json_body["parameters"]["size"] == "512*512"


def test_qwen_edit_auto_and_custom_sizes():
    auto_request = api_qwen.prepare_image_request(
        _provider("qwen"),
        operation="edit",
        prompt="p",
        images=[png_bytes(64, 64)],
        params={"size_mode": "auto", "n": 1, "seed": 1},
    )
    assert "size" not in auto_request.json_body["parameters"]

    custom_request = api_qwen.prepare_image_request(
        _provider("qwen"),
        operation="edit",
        prompt="p",
        images=[png_bytes(64, 64)],
        params={"size_mode": "custom", "width": 2048, "height": 1152, "n": 1, "seed": 1},
    )
    assert custom_request.json_body["parameters"]["size"] == "2048*1152"


def test_qwen_edit_reference_validation():
    with pytest.raises(EndpointError) as missing:
        api_qwen.prepare_image_request(
            _provider("qwen"), operation="edit", prompt="p", params={"n": 1, "seed": 1}
        )
    assert missing.value.error_code == "missing_image"

    with pytest.raises(EndpointError) as too_many:
        api_qwen.prepare_image_request(
            _provider("qwen"),
            operation="edit",
            prompt="p",
            images=[png_bytes() for _ in range(4)],
            params={"size_mode": "auto", "n": 1, "seed": 1},
        )
    assert too_many.value.error_code == "too_many_images"

    with pytest.raises(EndpointError) as bad_ref:
        api_qwen.prepare_image_request(
            _provider("qwen"),
            operation="edit",
            prompt="用 @Image3 的风格",
            images=[png_bytes()],
            params={"size_mode": "auto", "n": 1, "seed": 1},
        )
    assert bad_ref.value.error_code == "invalid_image_reference"


def test_qwen_leaves_email_addresses_and_plain_text_alone():
    assert api_qwen.resolve_image_refs("mail me at user@image1.com", 1) == (
        "mail me at user@image1.com"
    )
    # adjacent tags all resolve, an unnumbered @image means the first image
    assert api_qwen.resolve_image_refs("@Image1@image 和 @Image2", 2) == (
        "Image 1Image 1 和 Image 2"
    )


def test_qwen_rejects_a_mask_empty_prompt_and_bad_n():
    with pytest.raises(EndpointError) as masked:
        api_qwen.prepare_image_request(
            _provider("qwen"),
            operation="generate",
            prompt="p",
            mask=png_bytes(),
            params={"width": 1024, "height": 1024, "n": 1, "seed": 1},
        )
    assert masked.value.error_code == "mask_unsupported"

    with pytest.raises(EndpointError) as empty:
        api_qwen.prepare_image_request(
            _provider("qwen"),
            operation="generate",
            prompt="   ",
            params={"width": 1024, "height": 1024, "n": 1, "seed": 1},
        )
    assert empty.value.error_code == "empty_prompt"

    with pytest.raises(EndpointError) as bad_n:
        api_qwen.prepare_image_request(
            _provider("qwen"),
            operation="generate",
            prompt="p",
            params={"width": 1024, "height": 1024, "n": 7, "seed": 1},
        )
    assert bad_n.value.error_code == "invalid_n"


def test_qwen_parse_images_and_text():
    payload = {
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"image": "https://cdn.example.com/a.png"},
                            {"text": "已经换好了"},
                        ],
                    },
                }
            ]
        },
        "request_id": "req-qwen-1",
        "usage": {"image_count": 1},
    }
    client, opener = _client(
        json_response(payload),
        RawResponse(200, png_bytes(6, 6)),
    )
    response = client.request("POST", api_qwen.generation_endpoint(_provider("qwen")), json_body={})
    parsed = api_qwen.parse_image_response(_provider("qwen"), response, client, timeout=30)
    assert len(parsed.images) == 1
    assert parsed.image_urls == ["https://cdn.example.com/a.png"]
    assert parsed.text == "已经换好了"
    assert parsed.details["request_id"] == "req-qwen-1"
    assert parsed.details["finish_reasons"] == ["stop"]
    # the download must not carry the API key (signed OSS URLs)
    assert opener.urls[1] == "https://cdn.example.com/a.png"
    assert "Authorization" not in opener.calls[1].headers


def test_qwen_http_error_body_is_reported():
    client, _ = _client(
        json_response(
            {"code": "InvalidApiKey", "message": "Invalid API-key provided.", "request_id": "r1"},
            status=401,
        )
    )
    response = client.request("POST", api_qwen.generation_endpoint(_provider("qwen")), json_body={})
    with pytest.raises(EndpointError) as excinfo:
        api_qwen.parse_image_response(_provider("qwen"), response, client, timeout=30)
    assert excinfo.value.status == 401
    assert "Invalid API-key" in excinfo.value.message
    assert excinfo.value.error_code == "InvalidApiKey"


def test_qwen_empty_output_reports_the_provider_code():
    client, _ = _client(
        json_response(
            {
                "output": None,
                "code": "DataInspectionFailed",
                "message": "Output data may contain inappropriate content.",
                "request_id": "r2",
            }
        )
    )
    response = client.request("POST", api_qwen.generation_endpoint(_provider("qwen")), json_body={})
    with pytest.raises(EndpointError) as excinfo:
        api_qwen.parse_image_response(_provider("qwen"), response, client, timeout=30)
    assert excinfo.value.error_type == "empty_result"
    assert excinfo.value.error_code == "DataInspectionFailed"
    assert "DataInspectionFailed" in excinfo.value.message


def test_qwen_full_url_endpoint_is_not_doubled():
    provider = _provider(
        "qwen",
        endpoint="https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    )
    assert api_qwen.generation_endpoint(provider) == (
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    )


# ---------------------------------------------------------------------------
# Qwen via an OpenAI-compatible relay (/images/*), the shape relays expose
# ---------------------------------------------------------------------------
def test_qwen_images_edit_body():
    request = api_qwen.prepare_image_request(
        _provider("qwen_images", endpoint="https://relay.example.com"),
        operation="edit",
        prompt="把 @Image1 的手修好",
        images=[png_bytes(1000, 600)],
        params={
            "size_mode": "match input",
            "n": 1,
            "seed": 42,
            "prompt_extend": False,
            "watermark": False,
            "negative_prompt": "畸形",
        },
    )
    assert request.url == "https://relay.example.com/v1/images/edits"
    body = request.json_body
    assert body["model"] == "qwen-image-3.0"
    assert body["prompt"] == "把 Image 1 的手修好"
    assert body["image"].startswith("data:image/png;base64,")
    assert body["size"] == "1000*600"
    assert body["n"] == 1 and body["seed"] == 42
    assert body["prompt_extend"] is False and body["watermark"] is False
    assert body["negative_prompt"] == "畸形"
    assert "input" not in body and "parameters" not in body


def test_qwen_images_generation_body_and_multiple_references():
    provider = _provider("qwen_images", endpoint="https://relay.example.com")

    generate = api_qwen.prepare_image_request(
        provider,
        operation="generate",
        prompt="一只猫",
        params={"width": 1024, "height": 1024, "n": 1, "seed": 7},
    )
    assert generate.url == "https://relay.example.com/v1/images/generations"
    assert "image" not in generate.json_body
    assert generate.json_body["size"] == "1024*1024"

    edit = api_qwen.prepare_image_request(
        provider,
        operation="edit",
        prompt="融合",
        images=[png_bytes(600, 600), png_bytes(600, 600), png_bytes(600, 600)],
        params={"size_mode": "auto", "n": 1, "seed": 7},
    )
    assert isinstance(edit.json_body["image"], list) and len(edit.json_body["image"]) == 3
    assert "size" not in edit.json_body, "auto size must be omitted"


def test_qwen_images_uses_the_smaller_area_window():
    relay = _provider("qwen_images", endpoint="https://relay.example.com")
    native = _provider("qwen")

    # 2560x2560 is fine on the official path, rejected on the relay surface
    assert api_qwen.prepare_image_request(
        native, operation="generate", prompt="x",
        params={"width": 2560, "height": 2560, "n": 1, "seed": 1},
    ).json_body["parameters"]["size"] == "2560*2560"

    with pytest.raises(EndpointError) as excinfo:
        api_qwen.prepare_image_request(
            relay, operation="generate", prompt="x",
            params={"width": 2560, "height": 2560, "n": 1, "seed": 1},
        )
    assert excinfo.value.error_code == "invalid_size"
    assert "2048x2048" in excinfo.value.message

    # match input is fitted into the relay window instead of failing
    request = api_qwen.prepare_image_request(
        relay, operation="edit", prompt="p",
        images=[png_bytes(2200, 2200)],
        params={"size_mode": "match input", "n": 1, "seed": 1},
    )
    assert request.json_body["size"] == "2048*2048"


def test_qwen_missing_route_detection():
    invalid_url = HttpResponse(
        status=404,
        headers={},
        body=json.dumps(
            {"error": {"message": "Invalid URL (POST /v1/services/aigc/multimodal-generation/generation)",
                       "type": "invalid_request_error"}}
        ).encode(),
        url="u",
        method="POST",
        latency_ms=1,
    )
    assert api_qwen.is_missing_route(invalid_url) is True

    other_404 = HttpResponse(status=404, headers={}, body=b'{"error":{"message":"model not found"}}',
                             url="u", method="POST", latency_ms=1)
    assert api_qwen.is_missing_route(other_404) is False
    assert api_qwen.is_missing_route(HttpResponse(status=200, headers={}, body=b"{}", url="u",
                                                  method="POST", latency_ms=1)) is False


def test_qwen_images_parse_openai_style_response():
    payload = {
        "created": 1793801744,
        "data": [{"url": "https://cdn.example.com/a.png"}, {"url": "https://cdn.example.com/b.png"}],
    }
    client, opener = _client(
        json_response(payload),
        RawResponse(200, png_bytes(8, 8)),
        RawResponse(200, png_bytes(8, 8)),
    )
    provider = _provider("qwen_images", endpoint="https://relay.example.com")
    response = client.request("POST", "https://relay.example.com/v1/images/edits", json_body={})
    parsed = api_qwen.parse_image_response(provider, response, client, timeout=30)
    assert len(parsed.images) == 2
    assert parsed.image_urls == ["https://cdn.example.com/a.png", "https://cdn.example.com/b.png"]
    assert parsed.details["created"] == 1793801744
    assert opener.urls[1:] == ["https://cdn.example.com/a.png", "https://cdn.example.com/b.png"]
    assert all("Authorization" not in call.headers for call in opener.calls[1:])


def test_qwen_error_object_with_http_200_is_raised():
    client, _ = _client(
        json_response({"error": {"message": "The model service is temporarily unavailable.",
                                 "type": "new_api_error"}})
    )
    provider = _provider("qwen_images", endpoint="https://relay.example.com")
    response = client.request("POST", "https://relay.example.com/v1/images/edits", json_body={})
    with pytest.raises(EndpointError) as excinfo:
        api_qwen.parse_image_response(provider, response, client, timeout=30)
    assert "temporarily unavailable" in excinfo.value.message
