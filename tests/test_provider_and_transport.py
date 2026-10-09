"""Provider configuration, endpoint normalization, credentials and transport."""

import json

import pytest

from custom_endpoint import provider as P
from custom_endpoint import secrets, transport
from custom_endpoint.errors import EndpointError
from custom_endpoint.transport import HttpResponse
from conftest import QueueOpener, RawResponse, json_response


# ---------------------------------------------------------------------------
# base url
# ---------------------------------------------------------------------------
def test_normalize_base_url_fills_the_version_prefix():
    assert P.normalize_base_url("https://a.com", "openai_images") == "https://a.com/v1"
    assert P.normalize_base_url("https://a.com/", "openai_images") == "https://a.com/v1"
    assert P.normalize_base_url("http://127.0.0.1:8317", "gemini") == "http://127.0.0.1:8317/v1beta"


def test_normalize_base_url_keeps_an_explicit_version():
    assert P.normalize_base_url("https://a.com/v1beta", "gemini") == "https://a.com/v1beta"
    assert P.normalize_base_url("https://a.com/openai/v1", "openai_images") == "https://a.com/openai/v1"
    assert P.normalize_base_url("https://a.com/api/v2", "openai_chat") == "https://a.com/api/v2"


def test_normalize_base_url_strips_query_and_adds_scheme():
    assert P.normalize_base_url("a.com/api?x=1", "openai_images") == "https://a.com/api/v1"


def test_normalize_base_url_rejects_non_http():
    with pytest.raises(ValueError):
        P.normalize_base_url("ftp://a.com", "openai_images")
    assert P.normalize_base_url("", "openai_images") == ""


# ---------------------------------------------------------------------------
# protocol inference + model ids
# ---------------------------------------------------------------------------
def test_infer_protocol_from_model_name():
    assert P.infer_protocol("gpt-image-2.5-flare") == P.PROTOCOL_OPENAI_IMAGES
    assert P.infer_protocol("dall-e-3") == P.PROTOCOL_OPENAI_IMAGES
    assert P.infer_protocol("gemini-3.1-flash-image") == P.PROTOCOL_GEMINI
    assert P.infer_protocol("Nano Banana 2 LitE") == P.PROTOCOL_GEMINI
    assert P.infer_protocol("grok-imagine-image-2.0") == P.PROTOCOL_GROK
    assert P.infer_protocol("grok-4") == P.PROTOCOL_GROK
    assert P.infer_protocol("gpt-5.5") == P.PROTOCOL_OPENAI_CHAT
    assert P.infer_protocol("") == P.PROTOCOL_OPENAI_CHAT


def test_friendly_gemini_labels_map_to_api_ids():
    assert P.normalize_model_id("Nano Banana 2 (Gemini 3.1 Flash Image)", P.PROTOCOL_GEMINI) == "gemini-3.1-flash-image"
    assert P.normalize_model_id("Nano Banana 2 Lite", P.PROTOCOL_GEMINI) == "gemini-3.1-flash-lite-image"
    assert P.normalize_model_id("Gemini 3.1 Pro", P.PROTOCOL_GEMINI) == "gemini-3.1-pro-preview"
    assert P.normalize_model_id("my-private-model", P.PROTOCOL_GEMINI) == "my-private-model"
    assert P.normalize_model_id("grok-imagine-image-pro", P.PROTOCOL_GROK) == "grok-imagine-image-pro"


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------
def test_build_provider_resolves_auto_and_normalizes():
    provider = P.build_provider(
        endpoint="https://api.test", api_key="sk-x", model="Nano Banana 2", api_protocol="auto"
    )
    assert provider.api_protocol == P.PROTOCOL_GEMINI
    assert provider.model == "gemini-3.1-flash-image"
    assert provider.base_url == "https://api.test/v1beta"
    assert provider.family == P.FAMILY_GEMINI


def test_build_provider_never_raises_for_misconfiguration():
    provider = P.build_provider(endpoint="", api_key="", model="", api_protocol="auto")
    assert provider.config_error
    with pytest.raises(EndpointError) as excinfo:
        provider.validate()
    assert excinfo.value.error_code == "provider_invalid"


def test_build_provider_reports_a_missing_model():
    provider = P.build_provider(endpoint="https://a.com", api_key="k", model="   ")
    assert "model is empty" in provider.config_error


def test_extra_headers_json_roundtrip_and_failure():
    provider = P.build_provider(
        endpoint="https://a.com", api_key="k", model="gpt-image-2", extra_headers='{"X-A":"1"}'
    )
    assert provider.extra_headers == {"X-A": "1"}
    broken = P.build_provider(
        endpoint="https://a.com", api_key="k", model="gpt-image-2", extra_headers="{oops"
    )
    assert broken.config_error and "extra_headers" in broken.config_error


def test_rejects_unknown_protocol():
    with pytest.raises(ValueError):
        P.ProviderConfig(base_url="https://a.com/v1", model="m", api_protocol="nope")


def test_require_protocols_rejects_a_foreign_provider():
    provider = P.build_provider(
        endpoint="https://a.com", api_key="k", model="gemini-3.1-flash-image", api_protocol="gemini"
    )
    with pytest.raises(EndpointError) as excinfo:
        P.require_protocols(provider, ("openai_images",), node_label="GPT 图像")
    assert excinfo.value.error_code == "protocol_mismatch"


def test_public_dict_hides_the_credential():
    provider = P.build_provider(endpoint="https://a.com", api_key="sk-plain", model="gpt-image-2")
    public = provider.public_dict()
    assert public["api_key_configured"] is True
    assert "api_key" not in public
    assert "sk-plain" not in json.dumps(public)


# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------
def test_session_handle_is_single_use():
    handle = secrets.issue_session_credential("sk-secret")
    assert secrets.is_session_handle(handle)
    provider = P.build_provider(
        endpoint="https://a.com", api_key=handle, model="gpt-image-2", api_protocol="openai_images"
    )
    assert provider.credential() == "sk-secret"
    with pytest.raises(ValueError):
        provider.credential()


def test_plain_key_passes_through_unchanged():
    provider = P.build_provider(endpoint="https://a.com", api_key="sk-plain", model="gpt-image-2")
    assert provider.credential() == "sk-plain"
    assert provider.credential() == "sk-plain"


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------
def test_retries_on_5xx_then_succeeds():
    opener = QueueOpener(RawResponse(503, "nope"), json_response({"ok": True}))
    client = transport.HttpClient(opener=opener, sleeper=lambda _: None)
    response = client.request("POST", "https://a.com/x", json_body={"a": 1}, retries=1)
    assert response.ok
    assert response.json() == {"ok": True}
    assert len(opener.calls) == 2


def test_network_failure_raises_a_structured_error():
    opener = QueueOpener(OSError("connection refused"))
    client = transport.HttpClient(opener=opener, sleeper=lambda _: None)
    with pytest.raises(EndpointError) as excinfo:
        client.request("GET", "https://a.com/x", retries=0)
    assert excinfo.value.error_type == "transport"
    assert excinfo.value.retryable is True


def test_error_status_is_returned_not_raised():
    opener = QueueOpener(json_response({"error": {"message": "bad"}}, status=400))
    client = transport.HttpClient(opener=opener, sleeper=lambda _: None)
    response = client.request("POST", "https://a.com/x", json_body={})
    assert response.ok is False
    assert response.status == 400


def test_http_response_helpers():
    response = HttpResponse(
        status=200,
        headers={"X-Request-Id": "r1"},
        body=b'{"a":1}',
        url="https://a.com/x",
        method="POST",
        latency_ms=5,
    )
    assert response.header("x-request-id") == "r1"
    assert response.json() == {"a": 1}
    assert response.excerpt() == '{"a":1}'


def test_non_json_body_raises_a_decode_error():
    response = HttpResponse(status=200, headers={}, body=b"<html>", url="u", method="GET", latency_ms=1)
    with pytest.raises(EndpointError) as excinfo:
        response.json()
    assert excinfo.value.error_code == "invalid_json"


def test_build_multipart_shape():
    content_type, body = transport.build_multipart(
        {"model": "m", "n": 2, "flag": True},
        [("image", "i.png", b"\x89PNGdata", "image/png")],
    )
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="model"' in body
    assert b'name="n"' in body and b"2" in body
    assert b"true" in body
    assert b'filename="i.png"' in body
    assert b"\x89PNGdata" in body


def test_auth_headers_per_protocol():
    gemini = P.build_provider(
        endpoint="https://a.com", api_key="sk-1", model="gemini-3.1-flash-image", api_protocol="gemini"
    )
    headers = transport.auth_headers(gemini)
    assert headers["x-goog-api-key"] == "sk-1"
    assert headers["Authorization"] == "Bearer sk-1"

    openai = P.build_provider(
        endpoint="https://a.com", api_key="sk-2", model="gpt-image-2", api_protocol="openai_images"
    )
    assert transport.auth_headers(openai) == {"Authorization": "Bearer sk-2"}

    extra = P.build_provider(
        endpoint="https://a.com",
        api_key="sk-3",
        model="gpt-image-2",
        api_protocol="openai_images",
        extra_headers='{"X-Trace":"1"}',
    )
    assert transport.auth_headers(extra)["X-Trace"] == "1"


def test_missing_credential_is_reported_as_an_error():
    provider = P.build_provider(
        endpoint="https://a.com", api_key="", model="gpt-image-2", api_protocol="openai_images"
    )
    with pytest.raises(EndpointError) as excinfo:
        transport.auth_headers(provider)
    assert excinfo.value.error_code == "credential_missing"
