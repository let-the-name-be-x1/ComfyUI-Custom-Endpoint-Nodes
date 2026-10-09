"""Generate the bundled example workflows.

Run with the ComfyUI python from the package root:

    python tools/make_examples.py

The generator derives every node's `inputs`, `outputs` and `widgets_values`
from the node classes themselves (INPUT_TYPES / RETURN_TYPES), so an example
cannot drift away from the real node definitions; `tests/test_example_workflows.py`
re-checks the result.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ce_nodes as nodes  # noqa: E402

WIDGET_TYPES = ("STRING", "INT", "FLOAT", "BOOLEAN")
OUTPUT_DIR = os.path.join(ROOT, "example_workflows")


def merged_spec(class_type: str) -> list[tuple[str, tuple]]:
    """Declared inputs in order: [(name, definition), ...]."""
    spec = nodes.NODE_CLASS_MAPPINGS[class_type].INPUT_TYPES()
    items: list[tuple[str, tuple]] = []
    for section in ("required", "optional"):
        for name, definition in spec.get(section, {}).items():
            items.append((name, definition))
    return items


def is_widget(definition) -> bool:
    if not isinstance(definition, (list, tuple)) or not definition:
        return False
    if isinstance(definition[0], (list, tuple)):
        return True
    return definition[0] in WIDGET_TYPES


def widget_type_name(definition) -> str:
    if isinstance(definition[0], (list, tuple)):
        return "COMBO"
    return definition[0]


def widget_names(class_type: str) -> list[str]:
    return [name for name, definition in merged_spec(class_type) if is_widget(definition)]


def widget_values_for(class_type: str, overrides: dict) -> list:
    values = []
    for name, definition in merged_spec(class_type):
        if not is_widget(definition):
            continue
        values.append(overrides[name] if name in overrides else definition[1].get("default"))
    return values


def build_node(node_id, class_type, pos, *, widgets="default", links_in=None, links_out=None, size=(360, 200)):
    """Build one serialized node entry straight from the node definition."""
    links_in = dict(links_in or {})
    links_out = dict(links_out or {})

    if widgets == "default":
        values = widget_values_for(class_type, {})
    else:
        values = widget_values_for(class_type, widgets)

    inputs = []
    for name, definition in merged_spec(class_type):
        if is_widget(definition):
            inputs.append(
                {
                    "label": name,
                    "name": name,
                    "type": widget_type_name(definition),
                    "link": links_in.get(name),
                    "widget": {"name": name},
                }
            )
        else:
            inputs.append(
                {"label": name, "name": name, "type": definition[0], "link": links_in.get(name)}
            )

    cls = nodes.NODE_CLASS_MAPPINGS[class_type]
    outputs = []
    for index, (type_name, out_name) in enumerate(zip(cls.RETURN_TYPES, cls.RETURN_NAMES)):
        outputs.append(
            {
                "label": out_name,
                "name": out_name,
                "type": type_name,
                "links": links_out.get(out_name) or [],
                "slot_index": index,
            }
        )
    return {
        "id": node_id,
        "type": class_type,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": node_id - 1,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {"Node name for S&R": class_type, "cnr_id": class_type},
        "widgets_values": values,
    }


def builtin_node(node_id, class_type, pos, widgets, *, inputs, outputs, size=(300, 120)):
    return {
        "id": node_id,
        "type": class_type,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": node_id - 1,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {"Node name for S&R": class_type},
        "widgets_values": widgets,
    }


def widget_input(name, type_name="STRING"):
    return {"label": name, "name": name, "type": type_name, "link": None, "widget": {"name": name}}


def socket_input(name, type_name, link):
    return {"label": name, "name": name, "type": type_name, "link": link}


def save_image(node_id, pos, link, prefix):
    return builtin_node(
        node_id,
        "SaveImage",
        pos,
        [prefix],
        inputs=[socket_input("images", "IMAGE", link), widget_input("filename_prefix")],
        outputs=[],
    )


def workflow(nodes_list, links, *, last_node_id):
    return {
        "id": "00000000-0000-0000-0000-0000000000c1",
        "revision": 0,
        "last_node_id": last_node_id,
        "last_link_id": len(links),
        "nodes": nodes_list,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 1.0, "offset": [0, 0]}, "frontendVersion": "1.0.0"},
        "version": 0.4,
    }


PROVIDER_COMMON = {
    "endpoint": "https://your-relay.example.com/v1",
    "api_key": "",
    "encrypt_export": True,
}


def build_gpt_image_example():
    nodes_list = [
        build_node(
            1,
            "CustomEndpointProvider",
            [40, 120],
            widgets={**PROVIDER_COMMON, "model": "gpt-image-2.5-flare", "api_protocol": "openai_images"},
            links_out={"provider": [1]},
        ),
        build_node(
            2,
            "CustomEndpointGPTImage",
            [500, 120],
            widgets={
                "prompt": "a red fox sitting in tall grass, morning light",
                "size": "1024x1024",
                "quality": "high",
            },
            links_in={"provider": 1},
            links_out={"images": [2]},
            size=(380, 420),
        ),
        save_image(5, [980, 140], 2, "custom_endpoint/gpt-image"),
    ]
    links = [[1, 1, 0, 2, 0, "CUSTOM_ENDPOINT_PROVIDER"], [2, 2, 0, 5, 0, "IMAGE"]]
    return workflow(nodes_list, links, last_node_id=5)


def build_gemini_batch_example():
    nodes_list = [
        build_node(
            1,
            "CustomEndpointProvider",
            [40, 60],
            widgets={
                "endpoint": "https://your-relay.example.com",
                "api_key": "",
                "model": "gemini-3.1-flash-image",
                "api_protocol": "gemini",
                "encrypt_export": True,
            },
            links_out={"provider": [1]},
        ),
        build_node(
            3,
            "CustomEndpointPromptList",
            [40, 400],
            widgets={"text": "a fox in the snow\na fox at the beach\na fox on the moon"},
            links_out={"prompts": [2]},
            size=(360, 200),
        ),
        build_node(
            2,
            "CustomEndpointGeminiImage",
            [500, 100],
            widgets={
                "prompt": "",
                "aspect_ratio": "16:9",
                "resolution": "2K",
                "thinking_level": "HIGH",
                "system_prompt": nodes.GEMINI_IMAGE_SYS_PROMPT,
            },
            links_in={"provider": 1, "prompt": 2},
            links_out={"images": [3]},
            size=(400, 480),
        ),
        save_image(5, [1000, 120], 3, "custom_endpoint/gemini-batch"),
    ]
    links = [
        [1, 1, 0, 2, 0, "CUSTOM_ENDPOINT_PROVIDER"],
        [2, 3, 0, 2, 1, "STRING"],
        [3, 2, 0, 5, 0, "IMAGE"],
    ]
    return workflow(nodes_list, links, last_node_id=5)


def build_chat_example():
    nodes_list = [
        build_node(
            1,
            "CustomEndpointProvider",
            [40, 120],
            widgets={**PROVIDER_COMMON, "model": "gpt-5.5", "api_protocol": "openai_chat"},
            links_out={"provider": [1]},
        ),
        build_node(
            2,
            "CustomEndpointGPTChat",
            [500, 120],
            widgets={
                "prompt": "用一句话解释扩散模型中的 Soret 效应。",
                "instructions": "你是燃烧学领域的专家。",
            },
            links_in={"provider": 1},
            links_out={"texts": [2]},
            size=(400, 360),
        ),
        builtin_node(
            5,
            "PreviewAny",
            [1000, 140],
            [],
            inputs=[socket_input("source", "*", 2)],
            outputs=[],
        ),
    ]
    links = [[1, 1, 0, 2, 0, "CUSTOM_ENDPOINT_PROVIDER"], [2, 2, 0, 5, 0, "STRING"]]
    return workflow(nodes_list, links, last_node_id=5)


EXAMPLES = {
    "custom-endpoint-gpt-image.json": build_gpt_image_example,
    "custom-endpoint-gemini-batch.json": build_gemini_batch_example,
    "custom-endpoint-gpt-chat.json": build_chat_example,
}


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename, builder in EXAMPLES.items():
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(builder(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
