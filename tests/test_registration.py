"""Node registration + INPUT_TYPES contract."""

EXPECTED_NODES = {
    "CustomEndpointProvider",
    "CustomEndpointGPTImage",
    "CustomEndpointGeminiImage",
    "CustomEndpointGrokImage",
    "CustomEndpointQwenImage",
    "CustomEndpointQwenImageEdit",
    "CustomEndpointAPIDetailer",
    "CustomEndpointGPTChat",
    "CustomEndpointGeminiChat",
    "CustomEndpointGrokChat",
    "CustomEndpointGPTFiles",
    "CustomEndpointGeminiFiles",
    "CustomEndpointBatchImages",
    "CustomEndpointPromptList",
}

CALL_NODES = (
    "CustomEndpointGPTImage",
    "CustomEndpointGeminiImage",
    "CustomEndpointGrokImage",
    "CustomEndpointQwenImage",
    "CustomEndpointQwenImageEdit",
    "CustomEndpointAPIDetailer",
    "CustomEndpointGPTChat",
    "CustomEndpointGeminiChat",
    "CustomEndpointGrokChat",
)


def test_class_mappings_are_exact(package):
    assert set(package.NODE_CLASS_MAPPINGS) == EXPECTED_NODES
    assert set(package.NODE_DISPLAY_NAME_MAPPINGS) == EXPECTED_NODES


def test_display_names_are_non_empty_and_chinese(package):
    for node_id, label in package.NODE_DISPLAY_NAME_MAPPINGS.items():
        assert label.strip(), node_id
        assert "自定义端点" in label


def test_web_directory(package):
    assert package.WEB_DIRECTORY == "./custom_endpoint/web"


def test_provider_widget_order_puts_secrets_first(package):
    cls = package.NODE_CLASS_MAPPINGS["CustomEndpointProvider"]
    spec = cls.INPUT_TYPES()
    order = list(spec["required"]) + list(spec["optional"])
    # routes.py blanks widgets_values[0] and [1]; keep this contract stable.
    assert order[0] == "endpoint"
    assert order[1] == "api_key"
    assert order[2] == "model"
    assert spec["optional"]["encrypt_export"][1]["default"] is True
    assert "CUSTOM_ENDPOINT_PROVIDER" in cls.RETURN_TYPES


def test_every_call_node_is_a_concurrent_batch_node(package):
    for node_id in CALL_NODES:
        cls = package.NODE_CLASS_MAPPINGS[node_id]
        assert cls.INPUT_IS_LIST is True, node_id
        assert len(cls.OUTPUT_IS_LIST) == len(cls.RETURN_TYPES), node_id
        assert all(cls.OUTPUT_IS_LIST), node_id
        assert "request_log" in cls.RETURN_NAMES, node_id
        assert "response_log" in cls.RETURN_NAMES, node_id


def test_call_nodes_declare_the_widget_superset(package):
    gemini_image = package.NODE_CLASS_MAPPINGS["CustomEndpointGeminiImage"].INPUT_TYPES()
    required = set(gemini_image["required"])
    assert {"aspect_ratio", "resolution", "thinking_level", "response_modalities"} <= required
    assert "max_concurrency" in required

    gpt_image = package.NODE_CLASS_MAPPINGS["CustomEndpointGPTImage"].INPUT_TYPES()
    assert {"size", "custom_width", "custom_height", "quality", "background"} <= set(gpt_image["required"])
    assert "mask" in gpt_image["optional"]

    gpt_chat = package.NODE_CLASS_MAPPINGS["CustomEndpointGPTChat"].INPUT_TYPES()
    assert {"instructions", "truncation", "reasoning_effort"} <= set(gpt_chat["required"])

    qwen_image = package.NODE_CLASS_MAPPINGS["CustomEndpointQwenImage"].INPUT_TYPES()
    assert {"prompt", "negative_prompt", "width", "height", "n", "seed",
            "prompt_extend", "watermark"} <= set(qwen_image["required"])
    assert "optional" not in qwen_image or not qwen_image["optional"]

    qwen_edit = package.NODE_CLASS_MAPPINGS["CustomEndpointQwenImageEdit"].INPUT_TYPES()
    assert {"size_mode", "width", "height", "negative_prompt"} <= set(qwen_edit["required"])
    assert list(qwen_edit["optional"]) == ["image_1", "image_2", "image_3"]
    assert qwen_edit["required"]["size_mode"][0] == ("match input", "auto", "custom")


def test_detailer_declares_its_own_param_surface(package):
    cls = package.NODE_CLASS_MAPPINGS["CustomEndpointAPIDetailer"]
    spec = cls.INPUT_TYPES()
    required = spec["required"]
    assert {"image", "mask", "prompt", "crop_factor", "target_resolution",
            "feather", "drop_size", "mask_grow", "paste_area",
            "mask_guide"} <= set(required)
    assert required["mask_guide"][0] == (
        "none", "outline", "dim outside", "tint inside", "native mask",
    )
    assert required["mask_guide"][1]["default"] == "none"
    assert required["paste_area"][0] == ("crop box", "mask")
    assert required["paste_area"][1]["default"] == "crop box"
    assert cls.RETURN_NAMES == (
        "images", "texts", "request_log", "response_log", "cropped",
    )
    assert cls.OUTPUT_IS_LIST == (True, True, True, True, True)
    assert set(cls.ALLOWED_PROTOCOLS) == {
        "gemini", "openai_images", "grok", "qwen", "qwen_images",
    }


def test_every_protocol_is_offered_by_the_provider(package):
    spec = package.NODE_CLASS_MAPPINGS["CustomEndpointProvider"].INPUT_TYPES()
    options = spec["required"]["api_protocol"][0]
    assert isinstance(options, tuple)
    for protocol in ("auto", "openai_images", "openai_chat", "openai_responses", "gemini", "grok",
                     "qwen", "qwen_images"):
        assert protocol in options, protocol


def test_qwen_nodes_accept_both_qwen_surfaces(package):
    for node_id in ("CustomEndpointQwenImage", "CustomEndpointQwenImageEdit"):
        cls = package.NODE_CLASS_MAPPINGS[node_id]
        assert set(cls.ALLOWED_PROTOCOLS) == {"qwen", "qwen_images"}, node_id


def test_omittable_combos_start_with_omit_sentinel(package):
    from custom_endpoint.widgets import OMIT

    gpt_image = package.NODE_CLASS_MAPPINGS["CustomEndpointGPTImage"].INPUT_TYPES()
    for name in ("background", "quality", "output_format", "moderation", "response_format"):
        options = gpt_image["required"][name][0]
        assert isinstance(options, tuple), name
        assert options[0] == OMIT, name

    gemini_image = package.NODE_CLASS_MAPPINGS["CustomEndpointGeminiImage"].INPUT_TYPES()
    for name in ("resolution", "thinking_level", "response_modalities", "mime_type"):
        assert gemini_image["required"][name][0][0] == OMIT, name


def test_gemini_image_exposes_thought_image_output(package):
    cls = package.NODE_CLASS_MAPPINGS["CustomEndpointGeminiImage"]
    assert cls.RETURN_NAMES[1] == "thought_images"
    assert cls.RETURN_TYPES[1] == "IMAGE"


def test_aux_nodes_are_wired(package):
    files = package.NODE_CLASS_MAPPINGS["CustomEndpointGPTFiles"]
    assert files.RETURN_TYPES == ("CUSTOM_ENDPOINT_FILES",)
    gemini_files = package.NODE_CLASS_MAPPINGS["CustomEndpointGeminiFiles"]
    assert gemini_files.FAMILY == "gemini"
    prompt_list = package.NODE_CLASS_MAPPINGS["CustomEndpointPromptList"]
    assert prompt_list.OUTPUT_IS_LIST == (True,)
