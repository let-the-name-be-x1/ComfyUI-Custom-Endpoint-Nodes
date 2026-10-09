"""Redaction + the two log outputs."""

import json

from custom_endpoint.errors import EndpointError, error_json, report_json
from custom_endpoint.logs import (
    image_summary,
    request_log_json,
    response_log_json,
    summarize_binary,
)
from custom_endpoint.provider import build_provider
from custom_endpoint.redact import redact_headers, redact_text, redact_url, sanitize
from conftest import RawResponse, json_response, png_b64

SECRET = "sk-live-abcdefghijklmnop1234"


def _provider(**overrides):
    kwargs = dict(
        endpoint="https://api.example.com/v1",
        api_key=SECRET,
        model="gpt-image-2",
        api_protocol="openai_images",
    )
    kwargs.update(overrides)
    return build_provider(**kwargs)


def test_redact_text_masks_credentials_and_binary():
    assert SECRET not in redact_text(f"key={SECRET}")
    assert "Bearer [REDACTED]" in redact_text("Authorization: Bearer abc.def.ghi")
    assert "[REDACTED]" in redact_text("data:image/png;base64," + "A" * 400)
    assert "[REDACTED_BASE64]" in redact_text("A" * 300)
    assert "a=[REDACTED]" in redact_text("https://host/path?a=1&b=2")


def test_redact_url_keeps_path_and_masks_query():
    masked = redact_url("https://api.example.com/v1/models?key=secret&x=1")
    assert masked.startswith("https://api.example.com/v1/models?key=[REDACTED]")
    assert "secret" not in masked


def test_redact_headers_masks_auth_family():
    headers = redact_headers(
        {
            "Authorization": f"Bearer {SECRET}",
            "x-goog-api-key": SECRET,
            "Content-Type": "application/json",
        }
    )
    assert headers["Authorization"] == "[REDACTED]"
    assert headers["x-goog-api-key"] == "[REDACTED]"
    assert headers["Content-Type"] == "application/json"


def test_sanitize_is_recursive_and_keeps_shape():
    cleaned = sanitize({"api_key": SECRET, "nested": {"token": "abc", "list": ["Bearer x"]}})
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["nested"]["token"] == "[REDACTED]"
    assert cleaned["nested"]["list"] == ["Bearer [REDACTED]"]


def test_endpoint_error_report_never_leaks_the_key():
    error = EndpointError("boom", status=429, error_code="rate_limit", phase="image")
    payload = error.report(known_secrets=(SECRET,))
    assert payload["ok"] is False
    assert payload["http_status"] == 429
    assert payload["error_code"] == "rate_limit"
    assert SECRET not in json.dumps(payload, ensure_ascii=False)
    assert SECRET not in error.__str__()


def test_summarize_binary_detects_data_urls_and_blobs():
    summary = summarize_binary("data:image/png;base64," + png_b64())
    assert summary and summary["kind"] == "data_url"
    assert "png" in summary["summary"]
    assert summarize_binary("A" * 200)["kind"] == "base64"
    assert summarize_binary("hello") is None


def test_request_log_excludes_image_bytes_and_credentials():
    provider = _provider()
    body = {
        "model": "gpt-image-2",
        "prompt": "a cat",
        "images": [{"url": "data:image/png;base64," + png_b64(16, 16)}],
    }
    log = request_log_json(
        provider=provider,
        method="POST",
        url="https://api.example.com/v1/images/edits",
        headers={"Authorization": f"Bearer {SECRET}", "Content-Type": "application/json"},
        body=body,
        image_summary=image_summary([("image", [b"\x89PNG\r\n\x1a\n" + b"x" * 100])]),
        operation="edit",
        item_index=2,
    )
    payload = json.loads(log)
    assert payload["method"] == "POST"
    assert payload["operation"] == "edit"
    assert payload["item_index"] == 2
    assert payload["headers"]["Authorization"] == "[REDACTED]"
    assert SECRET not in log
    assert "base64," not in log
    assert payload["body"]["prompt"] == "a cat"
    assert payload["images"]["image"][0].startswith("image")
    assert payload["provider"]["api_key_configured"] is True


def test_response_log_reports_status_latency_and_parsed_text():
    provider = _provider()
    response = RawResponse(
        status=400,
        body=json.dumps({"error": {"message": "bad request", "code": "invalid"}}),
        headers={"x-request-id": "req-1"},
    )
    from custom_endpoint.transport import HttpResponse

    http = HttpResponse(
        status=response.status,
        headers=response.headers,
        body=response.read(),
        url="https://api.example.com/v1/images/generations",
        method="POST",
        latency_ms=123,
    )
    log = response_log_json(provider=provider, response=http, parsed={"error": "bad request"})
    payload = json.loads(log)
    assert payload["ok"] is False
    assert payload["http_status"] == 400
    assert payload["latency_ms"] == 123
    assert payload["request_id"] == "req-1"
    assert payload["body"]["error"]["message"] == "bad request"
    assert payload["parsed"]["error"] == "bad request"


def test_response_log_summarizes_b64_image_payloads():
    provider = _provider()
    from custom_endpoint.transport import HttpResponse

    http = HttpResponse(
        status=200,
        headers={},
        body=json.dumps({"data": [{"b64_json": png_b64(32, 32)}]}).encode(),
        url="https://api.example.com/v1/images/generations",
        method="POST",
        latency_ms=10,
    )
    log = response_log_json(provider=provider, response=http)
    payload = json.loads(log)
    assert payload["body"]["data"][0]["b64_json"]["kind"] == "base64"
    assert len(log) < 2000
    assert png_b64(32, 32) not in log


def test_error_json_handles_plain_exceptions():
    payload = json.loads(error_json(ValueError("nope"), phase="validate"))
    assert payload["ok"] is False
    assert payload["message"] == "nope"
    assert payload["error_type"] == "ValueError"


def test_report_json_sorts_keys_for_stable_diffs():
    assert report_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
