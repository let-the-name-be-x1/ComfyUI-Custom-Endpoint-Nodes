"""Structural validation of the bundled example workflows."""

import json
import os
import re

import pytest

import ce_nodes as nodes
from tools.make_examples import EXAMPLES, is_widget, merged_spec, widget_values_for

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_workflows")
BUILTIN_NODE_TYPES = {"SaveImage", "PreviewAny"}


def _load(filename):
    path = os.path.join(EXAMPLES_DIR, filename)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.parametrize("filename", sorted(EXAMPLES))
def test_example_exists_and_is_valid_json(filename):
    payload = _load(filename)
    assert payload["nodes"], filename
    assert isinstance(payload["links"], list)


@pytest.mark.parametrize("filename", sorted(EXAMPLES))
def test_every_node_type_is_known(filename):
    for entry in _load(filename)["nodes"]:
        assert entry["type"] in nodes.NODE_CLASS_MAPPINGS or entry["type"] in BUILTIN_NODE_TYPES, entry["type"]


@pytest.mark.parametrize("filename", sorted(EXAMPLES))
def test_our_nodes_declare_the_right_number_of_widgets(filename):
    for entry in _load(filename)["nodes"]:
        class_type = entry["type"]
        if class_type not in nodes.NODE_CLASS_MAPPINGS:
            continue
        spec = nodes.NODE_CLASS_MAPPINGS[class_type].INPUT_TYPES()
        expected = 0
        for section in ("required", "optional"):
            for name, definition in spec.get(section, {}).items():
                if is_widget(definition):
                    expected += 1
        assert len(entry["widgets_values"]) == expected, (
            f"{filename}:{class_type} has {len(entry['widgets_values'])} widget values, "
            f"expected {expected} ({[n for n, d in spec.get('required', {}).items() if is_widget(d)]})"
        )


@pytest.mark.parametrize("filename", sorted(EXAMPLES))
def test_links_point_at_existing_nodes_and_slots(filename):
    payload = _load(filename)
    by_id = {entry["id"]: entry for entry in payload["nodes"]}
    for link in payload["links"]:
        link_id, origin_id, origin_slot, target_id, target_slot, type_name = link
        assert origin_id in by_id, f"link {link_id}: unknown origin node {origin_id}"
        assert target_id in by_id, f"link {link_id}: unknown target node {target_id}"
        origin = by_id[origin_id]
        assert origin_slot < len(origin["outputs"]), f"link {link_id}: origin slot out of range"
        assert origin["outputs"][origin_slot]["type"] == type_name, f"link {link_id}: output type"
        target = by_id[target_id]
        assert target_slot < len(target["inputs"]), f"link {link_id}: target slot out of range"
        target_type = target["inputs"][target_slot]["type"]
        # "*" is ComfyUI's AnyType (PreviewAny) and accepts anything.
        assert target_type in (type_name, "*"), f"link {link_id}: input type"


@pytest.mark.parametrize("filename", sorted(EXAMPLES))
def test_examples_ship_no_credentials_or_local_paths(filename):
    raw = json.dumps(_load(filename), ensure_ascii=False)
    assert not re.search(r"sk-[A-Za-z0-9]", raw), "an API key looking value leaked into an example"
    assert not re.search(r"\b[A-Za-z]:[\\\\/]", raw), "an absolute local path leaked into an example"
    assert "custom-endpoint-session:" not in raw
    for entry in _load(filename)["nodes"]:
        if entry["type"] == "CustomEndpointProvider":
            widgets = entry["widgets_values"]
            assert widgets[1] == "", "example must ship an empty API key"
            assert widgets[0].startswith("https://your-relay.example.com")


@pytest.mark.parametrize("filename", sorted(EXAMPLES))
def test_inputs_are_serialized_in_declaration_order(filename):
    """ComfyUI restores node.inputs positionally; order must match INPUT_TYPES."""
    for entry in _load(filename)["nodes"]:
        class_type = entry["type"]
        if class_type not in nodes.NODE_CLASS_MAPPINGS:
            continue
        declared = [name for name, _ in merged_spec(class_type)]
        serialized = [item["name"] for item in entry["inputs"]]
        assert serialized == declared, f"{filename}:{class_type} {serialized} != {declared}"


def test_widget_values_helper_matches_the_nodes():
    values = widget_values_for("CustomEndpointProvider", {"endpoint": "x"})
    assert values[0] == "x"
    assert len(values) == 9  # endpoint, api_key, model, api_protocol + 5 optional widgets
