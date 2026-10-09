"""Wire-protocol shapes for the three provider families (no network)."""

import base64
import json

import pytest

from custom_endpoint import api_gemini, api_grok, api_openai
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
