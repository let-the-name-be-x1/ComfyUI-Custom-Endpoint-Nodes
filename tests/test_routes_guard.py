"""The on_prompt guard: what reaches the queue, the history and the PNG metadata."""

import json

from custom_endpoint import routes, secrets

SECRET = "sk-live-abcdefghijklmnop1234"
PROVIDER = "CustomEndpointProvider"
WIDGETS = [
    "https://api.example.com/v1",  # endpoint  (index 0)
    SECRET,                        # api_key   (index 1)
    "gpt-image-2.5-flare",         # model
    "auto",
    True,
    300,
    1,
    "",
    "provider",
]


def _payload(raw_key=SECRET):
    return {
        "prompt": {
            "1": {
                "class_type": PROVIDER,
                "inputs": {
                    "endpoint": "https://api.example.com/v1",
                    "api_key": raw_key,
                    "model": "gpt-image-2.5-flare",
                    "api_protocol": "auto",
                },
            },
            "2": {"class_type": "KSampler", "inputs": {"seed": 1}},
        },
        "extra_data": {
            "extra_pnginfo": {
                "workflow": {
                    "nodes": [
                        {"id": 1, "type": PROVIDER, "widgets_values": list(WIDGETS)},
                        {
                            "id": 2,
                            "type": "CustomEndpointGPTImage",
                            "widgets_values": ["a cat"],
                        },
                    ]
                }
            }
        },
    }


def test_raw_key_is_replaced_by_a_handle():
    payload = _payload()
    result = routes.guard_prompt_payload(payload)
    assert result is payload
    handle = result["prompt"]["1"]["inputs"]["api_key"]
    assert secrets.is_session_handle(handle)
    assert secrets.consume_session_credential(handle) == SECRET
    assert SECRET not in json.dumps(result, ensure_ascii=False)


def test_existing_handle_is_left_alone():
    handle = secrets.issue_session_credential(SECRET)
    payload = _payload(raw_key=handle)
    result = routes.guard_prompt_payload(payload)
    assert result["prompt"]["1"]["inputs"]["api_key"] == handle


def test_workflow_widgets_lose_endpoint_and_key_but_keep_the_model():
    payload = _payload()
    result = routes.guard_prompt_payload(payload)
    widgets = result["extra_data"]["extra_pnginfo"]["workflow"]["nodes"][0]["widgets_values"]
    assert widgets[0] == ""
    assert widgets[1] == ""
    assert widgets[2] == "gpt-image-2.5-flare"
    assert widgets[8] == "provider"
    serialized = json.dumps(result, ensure_ascii=False)
    assert SECRET not in serialized
    assert "https://api.example.com/v1" not in serialized


def test_prompt_inputs_only_carry_handles():
    payload = _payload()
    result = routes.guard_prompt_payload(payload)
    inputs = result["prompt"]["1"]["inputs"]
    assert secrets.is_session_handle(inputs["api_key"])
    assert secrets.is_session_handle(inputs["endpoint"])
    assert inputs["model"] == "gpt-image-2.5-flare"


def test_other_nodes_are_untouched():
    payload = _payload()
    result = routes.guard_prompt_payload(payload)
    other = result["extra_data"]["extra_pnginfo"]["workflow"]["nodes"][1]
    assert other["widgets_values"] == ["a cat"]
    assert result["prompt"]["2"]["inputs"]["seed"] == 1


def test_key_copied_elsewhere_is_redacted():
    payload = _payload()
    payload["extra_data"]["note"] = f"remember {SECRET} ok"
    result = routes.guard_prompt_payload(payload)
    assert SECRET not in json.dumps(result, ensure_ascii=False)
    assert "[REDACTED]" in result["extra_data"]["note"]


def test_empty_key_is_left_alone():
    payload = {"prompt": {"1": {"class_type": PROVIDER, "inputs": {"api_key": ""}}}}
    result = routes.guard_prompt_payload(payload)
    assert result["prompt"]["1"]["inputs"]["api_key"] == ""


def test_non_dict_payload_passes_through():
    assert routes.guard_prompt_payload(None) is None
    assert routes.guard_prompt_payload("nope") == "nope"


def test_widgets_values_as_mapping_is_supported():
    payload = {
        "prompt": {
            "1": {"class_type": PROVIDER, "inputs": {"api_key": secrets.issue_session_credential(SECRET)}}
        },
        "extra_data": {
            "extra_pnginfo": {
                "workflow": {
                    "nodes": [
                        {
                            "id": 1,
                            "type": PROVIDER,
                            "widgets_values": {"endpoint": "https://a/v1", "api_key": SECRET},
                        }
                    ]
                }
            }
        },
    }
    result = routes.guard_prompt_payload(payload)
    widgets = result["extra_data"]["extra_pnginfo"]["workflow"]["nodes"][0]["widgets_values"]
    assert widgets["endpoint"] == "" and widgets["api_key"] == ""


def test_provider_also_exposed_as_api_format_inputs_are_cleared():
    payload = {
        "prompt": {
            "1": {"class_type": PROVIDER, "inputs": {"api_key": secrets.issue_session_credential(SECRET)}}
        }
    }
    result = routes.guard_prompt_payload(payload)
    # The handle survives: execution still needs it.
    assert secrets.is_session_handle(result["prompt"]["1"]["inputs"]["api_key"])


def test_credential_response_issues_handles():
    handle = routes.credential_response({"api_key": SECRET})["credential_handle"]
    assert secrets.is_session_handle(handle)
    assert secrets.consume_session_credential(handle) == SECRET


def test_credential_response_rejects_empty_and_non_dict():
    import pytest

    with pytest.raises(ValueError):
        routes.credential_response({"api_key": ""})
    with pytest.raises(ValueError):
        routes.credential_response(["nope"])


def test_register_routes_is_a_noop_outside_comfyui():
    # pytest runs outside ComfyUI, so the import of `server` fails and this
    # must return False instead of raising.
    assert routes.register_routes() is False
