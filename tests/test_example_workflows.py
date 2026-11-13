"""Structural validation of the bundled example workflows.

`example_workflows/` holds hand-maintained exports from a real ComfyUI, so they
may reference third-party node packs (rgthree, pysssss, easy-use, …).  Only this
package's own nodes are checked against the live node definitions; the rest of
the file is checked for internal consistency (links, ids, no leaked secrets).
"""

from __future__ import annotations

import glob
import json
import os
import re

import pytest

import ce_nodes as nodes

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_workflows"
)
EXAMPLES = sorted(
    os.path.basename(path) for path in glob.glob(os.path.join(EXAMPLES_DIR, "*.json"))
)

WIDGET_TYPES = ("STRING", "INT", "FLOAT", "BOOLEAN")
# The ComfyUI frontend appends one extra value after a seed widget
# ("fixed" / "randomize" / "increment" / "decrement").
CONTROL_WIDGETS = {"seed"}
CONTROL_VALUES = {"fixed", "randomize", "increment", "decrement"}
# Nodes that legitimately do not take a provider input.
NO_PROVIDER_INPUT = {
    "CustomEndpointBatchImages",
    "CustomEndpointPromptList",
    "CustomEndpointGPTFiles",
    "CustomEndpointGeminiFiles",
}


def _load(filename):
    with open(os.path.join(EXAMPLES_DIR, filename), encoding="utf-8") as handle:
        return json.load(handle)


def _declared(class_type):
    """``[(name, definition), ...]`` in INPUT_TYPES declaration order."""
    spec = nodes.NODE_CLASS_MAPPINGS[class_type].INPUT_TYPES()
    items = []
    for section in ("required", "optional"):
        for name, definition in spec.get(section, {}).items():
            items.append((name, definition))
    return items


def _is_widget(definition):
    if not isinstance(definition, (list, tuple)) or not definition:
        return False
    if isinstance(definition[0], (list, tuple)):
        return True
    return definition[0] in WIDGET_TYPES


def _widget_names(class_type):
    return [name for name, definition in _declared(class_type) if _is_widget(definition)]


def _our_nodes(payload):
    for entry in payload["nodes"]:
        if entry["type"] in nodes.NODE_CLASS_MAPPINGS:
            yield entry


def test_examples_directory_is_not_empty():
    assert EXAMPLES, f"no example workflow found in {EXAMPLES_DIR}"


@pytest.mark.parametrize("filename", EXAMPLES)
def test_example_is_a_valid_workflow(filename):
    payload = _load(filename)
    assert payload["nodes"], filename
    assert isinstance(payload["links"], list)
    ids = [entry["id"] for entry in payload["nodes"]]
    assert len(ids) == len(set(ids)), f"{filename}: duplicate node ids"
    assert payload.get("last_node_id", max(ids)) >= max(ids)


@pytest.mark.parametrize("filename", EXAMPLES)
def test_our_node_types_are_registered(filename):
    """A node type starting with ``CustomEndpoint`` must exist in this package.

    Third-party types are allowed: these examples are real working graphs, not
    minimal demos, and deliberately reuse the author's usual helper nodes.
    """
    for entry in _load(filename)["nodes"]:
        if entry["type"].startswith("CustomEndpoint"):
            assert entry["type"] in nodes.NODE_CLASS_MAPPINGS, f"{filename}: {entry['type']}"


@pytest.mark.parametrize("filename", EXAMPLES)
def test_widget_values_line_up_with_the_declaration(filename):
    for entry in _our_nodes(_load(filename)):
        values = list(entry.get("widgets_values") or [])
        index = 0
        for name in _widget_names(entry["type"]):
            assert index < len(values), (
                f"{filename}:{entry['type']} has {len(values)} widget values, "
                f"expected one per declared widget"
            )
            index += 1
            if name in CONTROL_WIDGETS and index < len(values) and values[index] in CONTROL_VALUES:
                index += 1  # control_after_generate
        assert index == len(values), (
            f"{filename}:{entry['type']} has extra widget values {values[index:]!r}"
        )


@pytest.mark.parametrize("filename", EXAMPLES)
def test_serialized_inputs_are_declared(filename):
    """Every serialized input must exist, and only widgets may carry a ``widget`` key."""
    for entry in _our_nodes(_load(filename)):
        declared = dict(_declared(entry["type"]))
        names = [item["name"] for item in entry["inputs"]]
        assert len(names) == len(set(names)), f"{filename}:{entry['type']} duplicate inputs"
        for item in entry["inputs"]:
            assert item["name"] in declared, f"{filename}:{entry['type']} unknown input {item['name']}"
            assert ("widget" in item) is _is_widget(declared[item["name"]]), (
                f"{filename}:{entry['type']} input {item['name']} is not serialized like its declaration"
            )


def _type_matches(socket_type, link_type):
    """``*`` (AnyType) matches everything; some packs declare "FLOAT,INT,BOOLEAN"."""
    allowed = {part.strip() for part in str(socket_type).split(",")}
    return "*" in allowed or "*" in (link_type,) or link_type in allowed


@pytest.mark.parametrize("filename", EXAMPLES)
def test_links_point_at_existing_nodes_and_slots(filename):
    payload = _load(filename)
    by_id = {entry["id"]: entry for entry in payload["nodes"]}
    for link in payload["links"]:
        assert isinstance(link, list) and len(link) == 6, f"{filename}: unexpected link {link!r}"
        link_id, origin_id, origin_slot, target_id, target_slot, type_name = link
        assert origin_id in by_id, f"link {link_id}: unknown origin node {origin_id}"
        assert target_id in by_id, f"link {link_id}: unknown target node {target_id}"
        origin = by_id[origin_id]
        assert origin_slot < len(origin["outputs"]), f"link {link_id}: origin slot out of range"
        assert _type_matches(origin["outputs"][origin_slot]["type"], type_name), (
            f"link {link_id}: output type"
        )
        target = by_id[target_id]
        assert target_slot < len(target["inputs"]), f"link {link_id}: target slot out of range"
        assert _type_matches(target["inputs"][target_slot]["type"], type_name), (
            f"link {link_id}: input type"
        )


@pytest.mark.parametrize("filename", EXAMPLES)
def test_every_call_node_is_wired_to_a_provider(filename):
    for entry in _our_nodes(_load(filename)):
        if entry["type"] in NO_PROVIDER_INPUT or entry["type"] == "CustomEndpointProvider":
            continue
        provider = next(item for item in entry["inputs"] if item["name"] == "provider")
        assert provider.get("link") is not None, f"{filename}:{entry['type']} has no provider"


@pytest.mark.parametrize("filename", EXAMPLES)
def test_examples_ship_no_credentials_or_local_paths(filename):
    raw = json.dumps(_load(filename), ensure_ascii=False)
    assert not re.search(r"sk-[A-Za-z0-9]", raw), "an API key looking value leaked into an example"
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", raw), "an absolute local path leaked into an example"
    assert "custom-endpoint-session:" not in raw
    for entry in _our_nodes(_load(filename)):
        if entry["type"] != "CustomEndpointProvider":
            continue
        widgets = entry["widgets_values"]
        assert widgets[1] == "", f"{filename}: example must ship an empty API key"
        assert widgets[0] == "" or widgets[0].startswith("https://your-relay.example.com"), (
            f"{filename}: example must ship an empty or placeholder endpoint, got {widgets[0]!r}"
        )
