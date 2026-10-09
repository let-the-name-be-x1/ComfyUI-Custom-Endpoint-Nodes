"""Runs the front-end "encrypt export" scrubber under node.js.

The module under test imports ComfyUI's `scripts/app.js`; the test builds a
minimal stub tree so the real file is executed unmodified.
"""

import json
import os
import shutil
import subprocess

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(PACKAGE_ROOT, "custom_endpoint", "web")
NODE = shutil.which("node")

RUNNER = r"""
import { scrubSerializedWorkflow, PROVIDER_TYPE, SECRET_WIDGET_NAMES } from "./custom_endpoint/web/export_scrub.js";

const SECRET = "sk-live-abcdefghijklmnop1234";
const ENDPOINT = "https://api.example.com/v1";
const WIDGETS = ["endpoint", "api_key", "model", "api_protocol", "encrypt_export",
                 "timeout_sec", "retries", "extra_headers", "provider_name"];

function liveNode(encrypt) {
    return {
        id: 1,
        widgets: WIDGETS.map((name) => ({ name, value: name === "encrypt_export" ? encrypt : "" })),
    };
}
function graphWith(node) {
    return { getNodeById: (id) => (id === 1 ? node : undefined) };
}
function serialized() {
    return {
        nodes: [
            { id: 1, type: PROVIDER_TYPE,
              widgets_values: [ENDPOINT, SECRET, "gpt-image-2.5-flare", "openai_images", true, 300, 1, "", "provider"] },
            { id: 2, type: "CustomEndpointGPTImage", widgets_values: ["a cat"] },
        ],
    };
}

const failures = [];
function check(label, condition) {
    if (!condition) failures.push(label);
}

check("widget names", SECRET_WIDGET_NAMES.includes("endpoint") && SECRET_WIDGET_NAMES.includes("api_key"));

const on = scrubSerializedWorkflow(serialized(), graphWith(liveNode(true)));
check("endpoint cleared when encrypt on", on.nodes[0].widgets_values[0] === "");
check("key cleared when encrypt on", on.nodes[0].widgets_values[1] === "");
check("model kept", on.nodes[0].widgets_values[2] === "gpt-image-2.5-flare");
check("protocol kept", on.nodes[0].widgets_values[3] === "openai_images");
check("other node untouched", on.nodes[1].widgets_values[0] === "a cat");

const off = scrubSerializedWorkflow(serialized(), graphWith(liveNode(false)));
check("endpoint kept when encrypt off", off.nodes[0].widgets_values[0] === ENDPOINT);
check("key kept when encrypt off", off.nodes[0].widgets_values[1] === SECRET);

const shifted = { nodes: [{ id: 1, type: PROVIDER_TYPE,
    widgets_values: ["gpt-image-2.5-flare", "openai_images", true, ENDPOINT, SECRET] }] };
scrubSerializedWorkflow(shifted, null);
check("pattern fallback clears url", !shifted.nodes[0].widgets_values.includes(ENDPOINT));
check("pattern fallback clears key", !shifted.nodes[0].widgets_values.includes(SECRET));
check("pattern fallback keeps model", shifted.nodes[0].widgets_values[0] === "gpt-image-2.5-flare");

const dictForm = { nodes: [{ id: 1, type: PROVIDER_TYPE,
    widgets_values: { endpoint: ENDPOINT, api_key: SECRET, model: "gpt-image-2.5-flare" } }] };
scrubSerializedWorkflow(dictForm, graphWith(liveNode(true)));
check("mapping endpoint cleared", dictForm.nodes[0].widgets_values.endpoint === "");
check("mapping key cleared", dictForm.nodes[0].widgets_values.api_key === "");
check("mapping model kept", dictForm.nodes[0].widgets_values.model === "gpt-image-2.5-flare");

const noNodes = { nodes: [] };
check("empty graph is safe", scrubSerializedWorkflow(noNodes, null) === noNodes);
check("garbage is safe", scrubSerializedWorkflow({}, null) !== undefined);

if (failures.length) {
    console.error("FAILED: " + failures.join("; "));
    process.exit(1);
}
console.log("OK");
"""


def _stub_tree(root) -> None:
    (root / "scripts").mkdir(parents=True)
    (root / "custom_endpoint" / "web").mkdir(parents=True)
    (root / "scripts" / "app.js").write_text(
        "export const app = { graph: null, registerExtension() {} };\n", encoding="utf-8"
    )
    shutil.copy(
        os.path.join(WEB_DIR, "export_scrub.js"),
        root / "custom_endpoint" / "web" / "export_scrub.js",
    )
    (root / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    (root / "run.mjs").write_text(RUNNER, encoding="utf-8")


@pytest.mark.skipif(NODE is None, reason="node.js is not installed")
def test_export_scrub_behaviour(tmp_path):
    root = tmp_path / "tree"
    _stub_tree(root)
    result = subprocess.run(
        [NODE, str(root / "run.mjs")], capture_output=True, text=True, timeout=60, cwd=str(root)
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


def test_web_directory_contains_the_three_extensions():
    names = sorted(name for name in os.listdir(WEB_DIR) if name.endswith(".js"))
    assert names == ["export_scrub.js", "grow_image_inputs.js", "provider_widget.js"]


def test_extension_files_register_themselves():
    for name in ("export_scrub.js", "grow_image_inputs.js", "provider_widget.js"):
        text = open(os.path.join(WEB_DIR, name), encoding="utf-8").read()
        assert "app.registerExtension(" in text, name
        assert 'from "../../scripts/app.js"' in text, name


def test_grow_inputs_matches_the_declared_slots():
    """grow_image_inputs.js derives its maximum from nodeData, not a hardcode."""
    text = open(os.path.join(WEB_DIR, "grow_image_inputs.js"), encoding="utf-8").read()
    assert "declaredMax" in text
    assert "input?.optional" in text
    import ce_nodes as nodes

    for class_type in ("CustomEndpointGPTImage", "CustomEndpointGeminiImage", "CustomEndpointGrokImage"):
        spec = nodes.NODE_CLASS_MAPPINGS[class_type].INPUT_TYPES()
        slots = [name for name in spec["optional"] if name.startswith("image_")]
        assert slots == [f"image_{index}" for index in range(1, 7)], class_type
