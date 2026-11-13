"""API detailer: region math, mask guides, and the per-region call loop."""

import base64
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

import ce_nodes as nodes
from custom_endpoint import detailer
from conftest import (
    HandlerOpener,
    RawResponse,
    image_tensor,
    json_response,
    mask_tensor,
    png_b64,
)

SECRET = "sk-test-1234567890"
LINK_TYPES = ("IMAGE", "MASK", "CUSTOM_ENDPOINT_PROVIDER", "CUSTOM_ENDPOINT_FILES")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def widget_defaults(cls):
    """Mimic ComfyUI: every widget arrives wrapped in a one-element list."""
    spec = cls.INPUT_TYPES()
    values = {}
    for section in ("required", "optional"):
        for name, definition in spec.get(section, {}).items():
            if definition[0] in LINK_TYPES:
                continue
            values[name] = [definition[1].get("default")]
    return values


def make_provider(**overrides):
    kwargs = {
        "endpoint": "https://api.example.com",
        "api_key": SECRET,
        "model": "gemini-3.1-flash-image",
        "api_protocol": "gemini",
    }
    kwargs.update(overrides)
    return nodes.CustomEndpointProvider().build(**kwargs)[0]


def square_mask(size=32, boxes=((10, 22),), batch=1):
    mask = torch.zeros((batch, size, size), dtype=torch.float32)
    for start, end in boxes:
        mask[:, start:end, start:end] = 1.0
    return mask


def gemini_image_reply():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": png_b64(4, 4)}}
                    ]
                }
            }
        ]
    }
    return json_response(payload)


def detailer_kwargs(mask=None, **overrides):
    kwargs = widget_defaults(nodes.CustomEndpointAPIDetailer)
    kwargs.update(
        {
            "provider": make_provider(),
            "image": [image_tensor(1, 32, 32, 0.25)],
            "mask": [square_mask() if mask is None else mask],
            "prompt": ["a clean hand"],
            "feather": [0],
        }
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_make_crop_region_expands_around_the_centre():
    assert detailer.make_crop_region(1000, 800, (100, 100, 200, 200), 3.0) == (0, 0, 300, 300)
    assert detailer.make_crop_region(1000, 800, (900, 700, 950, 750), 3.0) == (850, 650, 1000, 800)
    assert detailer.make_crop_region(1000, 800, (950, 100, 1000, 200), 3.0) == (850, 0, 1000, 300)


def test_connected_components_finds_blobs_and_areas():
    binary = np.zeros((32, 32), dtype=bool)
    binary[2:7, 3:8] = True
    binary[20:25, 20:25] = True
    assert detailer.connected_components(binary) == [
        (3, 2, 8, 7, 25),
        (20, 20, 25, 25, 25),
    ]


def test_connected_components_uses_eight_connectivity():
    binary = np.zeros((8, 8), dtype=bool)
    binary[2, 2] = True
    binary[3, 3] = True
    assert len(detailer.connected_components(binary)) == 1


def test_connected_components_of_an_empty_mask_is_empty():
    assert detailer.connected_components(np.zeros((4, 4), dtype=bool)) == []


def test_regions_from_mask_drops_small_blobs():
    image = np.zeros((64, 64, 3), dtype=np.float32)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[8:24, 8:24] = 1.0
    mask[50, 50:52] = 1.0
    regions = detailer.regions_from_mask(image, mask, crop_factor=1.0, drop_size=10)
    assert len(regions) == 1
    assert regions[0].crop_region == (8, 8, 24, 24)
    assert regions[0].crop.shape == (16, 16, 3)


def test_mask_grow_expands_the_paste_area():
    image = np.zeros((64, 64, 3), dtype=np.float32)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[20:40, 20:40] = 1.0
    regions = detailer.regions_from_mask(image, mask, crop_factor=1.0, drop_size=10, mask_grow=4)
    assert regions[0].crop_region == (16, 16, 44, 44)
    assert regions[0].mask_crop[0, 0] == pytest.approx(1.0)


def test_resize_region_targets_the_longest_side():
    crop = np.zeros((40, 20, 3), dtype=np.float32)
    mask = np.ones((40, 20), dtype=np.float32)
    resized_crop, resized_mask = detailer.resize_region(crop, mask, 64)
    assert resized_crop.shape[:2] == (64, 32)
    assert resized_mask.shape == (64, 32)
    untouched, _ = detailer.resize_region(crop, mask, 0)
    assert untouched.shape[:2] == (40, 20)


def test_aspect_ratio_snaps_to_the_nearest_supported():
    assert detailer.aspect_ratio_for(1024, 1024) == "1:1"
    assert detailer.aspect_ratio_for(1024, 768) == "4:3"
    assert detailer.aspect_ratio_for(300, 900) == "1:4"
    assert detailer.aspect_ratio_for(1920, 1080) == "16:9"


# ---------------------------------------------------------------------------
# mask guides + compositing
# ---------------------------------------------------------------------------
def test_paint_guide_outline_marks_the_rim_only():
    crop = np.full((32, 32, 3), 0.5, dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[10:22, 10:22] = 1.0
    painted = detailer.paint_guide(detailer.to_uint8(crop), mask, detailer.GUIDE_OUTLINE)
    assert tuple(int(value) for value in painted[16, 16]) == (128, 128, 128)
    ring = [int(value) for value in painted[8, 16]]
    assert ring[0] > 200 and ring[1] > 180 and ring[2] < 60


def test_paint_guide_dim_and_tint_change_the_right_side():
    crop = np.full((16, 16, 3), 1.0, dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 4:12] = 1.0
    dimmed = detailer.paint_guide(detailer.to_uint8(crop), mask, detailer.GUIDE_DIM)
    assert dimmed[0, 0, 0] < 100
    assert dimmed[6, 6, 0] > 200
    tinted = detailer.paint_guide(detailer.to_uint8(crop), mask, detailer.GUIDE_TINT)
    assert tinted[6, 6, 2] > tinted[6, 6, 0]
    assert tuple(int(value) for value in tinted[0, 0]) == (255, 255, 255)


def test_feathered_alpha_softens_both_sides_of_the_edge():
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[16:, :] = 1.0
    alpha = detailer.feathered_alpha(mask, 4)
    assert alpha[0, 16] < 0.02
    assert 0.05 < alpha[14, 16] < 0.5
    assert 0.5 < alpha[18, 16] < 0.95
    assert alpha[31, 16] > 0.98


def test_composite_respects_the_mask():
    composed = np.zeros((32, 32, 3), dtype=np.float32)
    region = detailer.DetailRegion(
        (0, 0, 32, 32),
        np.zeros((32, 32, 3), dtype=np.float32),
        square_mask()[0].numpy(),
    )
    detailer.composite(composed, region, np.ones((32, 32, 3), dtype=np.float32), 0)
    assert composed[15, 15].tolist() == [1.0, 1.0, 1.0]
    assert composed[10, 10].tolist() == [1.0, 1.0, 1.0]
    assert composed[2, 2].tolist() == [0.0, 0.0, 0.0]


def test_composite_resizes_the_model_answer():
    composed = np.zeros((40, 40, 3), dtype=np.float32)
    region = detailer.DetailRegion(
        (10, 10, 30, 30),
        np.zeros((20, 20, 3), dtype=np.float32),
        np.ones((20, 20), dtype=np.float32),
    )
    detailer.composite(composed, region, np.full((8, 8, 3), 0.5, dtype=np.float32), 0)
    assert composed[20, 20].tolist() == pytest.approx([0.5, 0.5, 0.5], abs=0.01)
    assert composed[2, 2].tolist() == [0.0, 0.0, 0.0]


def test_native_mask_png_is_transparent_where_the_mask_is_white():
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    image = Image.open(io.BytesIO(detailer.native_mask_png(mask)))
    assert image.mode == "RGBA"
    alpha = np.asarray(image)[..., 3]
    assert alpha[4, 4] == 0
    assert alpha[0, 0] == 255


def test_frames_of_accepts_torch_batches():
    frames = detailer.frames_of(torch.zeros((2, 8, 8, 3), dtype=torch.float32))
    assert len(frames) == 2 and frames[0].shape == (8, 8, 3)
    masks = detailer.masks_of(torch.ones((2, 8, 8), dtype=torch.float32))
    assert len(masks) == 2 and masks[0].shape == (8, 8)


def test_rect_edge_alpha_fades_inside_the_crop_border():
    composed = np.zeros((40, 40, 3), dtype=np.float32)
    region = detailer.DetailRegion(
        (10, 10, 30, 30),
        np.zeros((20, 20, 3), dtype=np.float32),
        np.ones((20, 20), dtype=np.float32),
    )
    detailer.composite(
        composed, region, np.ones((20, 20, 3), dtype=np.float32), 4, detailer.PASTE_CROP
    )
    assert composed[20, 20].tolist() == [1.0, 1.0, 1.0]
    assert composed[10, 10][0] < 0.1
    assert 0.6 < composed[12, 20][0] < 0.75
    assert composed[14, 20][0] > 0.95
    assert composed[2, 2].tolist() == [0.0, 0.0, 0.0]


def test_rect_edge_alpha_ignores_the_image_border():
    composed = np.zeros((40, 40, 3), dtype=np.float32)
    region = detailer.DetailRegion(
        (0, 0, 20, 20),
        np.zeros((20, 20, 3), dtype=np.float32),
        np.ones((20, 20), dtype=np.float32),
    )
    detailer.composite(
        composed, region, np.ones((20, 20, 3), dtype=np.float32), 4, detailer.PASTE_CROP
    )
    assert composed[0, 0].tolist() == [1.0, 1.0, 1.0]
    assert composed[10, 10].tolist() == [1.0, 1.0, 1.0]
    assert composed[10, 19][0] < 0.1
    assert composed[19, 10][0] < 0.1


def test_guide_marker_sits_outside_the_paste_band():
    mask = np.zeros((96, 96), dtype=np.float32)
    mask[24:72, 24:72] = 1.0
    crop = detailer.to_uint8(np.full((96, 96, 3), 0.5, dtype=np.float32))
    alpha = detailer.feathered_alpha(mask, 5)

    def ring_of(painted):
        return (
            (painted[..., 0] > 200) & (painted[..., 1] > 180) & (painted[..., 2] < 60)
        )

    # pad=0 (the old placement): the marker sits inside the alpha ramp
    old = ring_of(detailer.paint_guide(crop, mask, detailer.GUIDE_OUTLINE, 0))
    assert alpha[old].max() > 0.2

    # default placement: pushed past the ramp
    new = ring_of(
        detailer.paint_guide(crop, mask, detailer.GUIDE_OUTLINE, detailer.guide_pad(5))
    )
    assert new.any()
    assert alpha[new].max() < 0.05


# ---------------------------------------------------------------------------
# node behaviour
# ---------------------------------------------------------------------------
def test_detailer_pastes_the_whole_crop_by_default(install_client):
    opener = HandlerOpener(lambda request: gemini_image_reply())
    install_client(opener)

    images, texts, _, _, cropped = nodes.CustomEndpointAPIDetailer().execute(
        **detailer_kwargs()
    )

    frame = images[0][0].numpy()
    assert frame[16, 16].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=0.01)
    # outside the mask but inside the crop: replaced as well
    assert frame[2, 2].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=0.01)
    report = json.loads(texts[0])
    assert report["paste_area"] == detailer.PASTE_CROP
    assert len(cropped) == 1 and tuple(cropped[0].shape) == (1, 4, 4, 3)


def test_detailer_pastes_only_the_mask_when_asked(install_client):
    opener = HandlerOpener(lambda request: gemini_image_reply())
    install_client(opener)

    images, texts, request_log, response_log, _ = nodes.CustomEndpointAPIDetailer().execute(
        **detailer_kwargs(**{"paste_area": ["mask"], "mask_guide": ["outline"]})
    )

    assert len(images) == 1
    frame = images[0][0].numpy()
    assert frame[16, 16].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=0.01)
    assert frame[2, 2].tolist() == pytest.approx([0.25, 0.25, 0.25], abs=0.01)

    payload = json.loads(opener.calls[0].data.decode("utf-8"))
    parts = payload["contents"][0]["parts"]
    assert parts[0]["text"].startswith("a clean hand")
    assert "yellow outline" in parts[0]["text"]
    assert len(parts) == 2 and parts[1]["inlineData"]["mimeType"] == "image/png"
    assert payload["generationConfig"]["imageConfig"]["aspectRatio"] == "1:1"
    assert payload["generationConfig"]["imageConfig"]["imageOutputOptions"]["mimeType"] == "image/png"

    request_entries = json.loads(request_log[0])
    assert isinstance(request_entries, list) and len(request_entries) == 1
    assert request_entries[0]["operation"] == "edit"
    assert request_entries[0]["url"].endswith(":generateContent")
    assert json.loads(response_log[0])[0]["parsed"]["image_count"] == 1
    assert json.loads(texts[0])["painted"] == 1


def test_crop_box_paste_drops_the_guide(install_client):
    opener = HandlerOpener(lambda request: gemini_image_reply())
    install_client(opener)

    _, texts, _, _, _ = nodes.CustomEndpointAPIDetailer().execute(
        **detailer_kwargs(**{"mask_guide": ["outline"], "target_resolution": [32]})
    )

    payload = json.loads(opener.calls[0].data.decode("utf-8"))
    parts = payload["contents"][0]["parts"]
    assert parts[0]["text"] == "a clean hand"
    sent = Image.open(io.BytesIO(base64.b64decode(parts[1]["inlineData"]["data"])))
    array = np.asarray(sent.convert("RGB"), dtype=np.int16)
    assert not ((array[..., 0] > 200) & (array[..., 1] > 180) & (array[..., 2] < 60)).any()
    assert json.loads(texts[0])["note"]


def test_detailer_calls_once_per_connected_region(install_client):
    opener = HandlerOpener(lambda request: gemini_image_reply())
    install_client(opener)

    mask = square_mask(boxes=((4, 10), (20, 26)))
    images, texts, _, _, _ = nodes.CustomEndpointAPIDetailer().execute(
        **detailer_kwargs(mask=mask)
    )

    assert len(opener.calls) == 2
    report = json.loads(texts[0])
    assert report["regions"] == 2 and report["painted"] == 2
    crop_regions = [item["crop_region"] for item in report["detail"][0]["items"]]
    assert crop_regions[0] != crop_regions[1]
    assert tuple(images[0].shape) == (1, 32, 32, 3)


def test_detailer_resizes_the_sent_crop(install_client):
    opener = HandlerOpener(lambda request: gemini_image_reply())
    install_client(opener)

    nodes.CustomEndpointAPIDetailer().execute(
        **detailer_kwargs(**{"target_resolution": [16]})
    )

    payload = json.loads(opener.calls[0].data.decode("utf-8"))
    data = payload["contents"][0]["parts"][1]["inlineData"]["data"]
    sent = Image.open(io.BytesIO(base64.b64decode(data)))
    assert sent.size == (16, 16)


def test_detailer_passes_through_when_the_mask_is_empty(install_client):
    opener = HandlerOpener(lambda request: gemini_image_reply())
    install_client(opener)

    images, texts, _, _, _ = nodes.CustomEndpointAPIDetailer().execute(
        **detailer_kwargs(mask=mask_tensor(1, 32, 32, 0.0))
    )

    assert opener.calls == []
    assert tuple(images[0].shape) == (1, 32, 32, 3)
    assert json.loads(texts[0])["regions"] == 0


def test_detailer_handles_image_batches_with_one_mask(install_client):
    opener = HandlerOpener(lambda request: gemini_image_reply())
    install_client(opener)

    batch = torch.cat([image_tensor(1, 32, 32, 0.25), image_tensor(1, 32, 32, 0.5)], dim=0)
    images, texts, _, _, _ = nodes.CustomEndpointAPIDetailer().execute(
        **detailer_kwargs(image=[batch])
    )

    assert len(opener.calls) == 2
    assert tuple(images[0].shape) == (2, 32, 32, 3)
    assert json.loads(texts[0])["frames"] == 2


def test_detailer_falls_back_to_one_pixel_when_every_region_fails(install_client):
    opener = HandlerOpener(lambda request: gemini_image_reply())
    install_client(opener)

    images, texts, _, _, _ = nodes.CustomEndpointAPIDetailer().execute(
        **detailer_kwargs(**{"mask_guide": ["native mask"], "paste_area": ["mask"]})
    )

    assert opener.calls == []
    assert tuple(images[0].shape) == (1, 1, 1, 3)
    assert "native mask" in texts[0]


def test_detailer_uses_the_native_mask_on_gpt_images(install_client):
    opener = HandlerOpener(lambda request: json_response({"data": [{"b64_json": png_b64(4, 4)}]}))
    install_client(opener)

    provider = make_provider(model="gpt-image-2.5-flare", api_protocol="openai_images")
    images, texts, _, _, _ = nodes.CustomEndpointAPIDetailer().execute(
        **detailer_kwargs(provider=provider, **{"mask_guide": ["native mask"], "paste_area": ["mask"]})
    )

    request = opener.calls[0]
    assert request.full_url.endswith("/v1/images/edits")
    assert b'name="mask"' in request.data
    assert tuple(images[0].shape) == (1, 32, 32, 3)
    assert json.loads(texts[0])["painted"] == 1


def test_detailer_turns_http_failures_into_a_one_pixel_image(install_client):
    opener = HandlerOpener(
        lambda request: json_response({"error": {"message": "boom"}}, status=500)
    )
    install_client(opener)

    images, texts, request_log, response_log, _ = nodes.CustomEndpointAPIDetailer().execute(
        **detailer_kwargs()
    )

    assert tuple(images[0].shape) == (1, 1, 1, 3)
    assert "boom" in texts[0] or "500" in texts[0]
    assert json.loads(response_log[0])[-1]["ok"] is False
    assert opener.calls  # the request was actually attempted (and retried)


def test_detailer_rejects_a_mask_of_the_wrong_size(install_client):
    opener = HandlerOpener(lambda request: gemini_image_reply())
    install_client(opener)

    images, texts, _, _, _ = nodes.CustomEndpointAPIDetailer().execute(
        **detailer_kwargs(mask=mask_tensor(1, 8, 8, 1.0))
    )

    assert opener.calls == []
    assert tuple(images[0].shape) == (1, 1, 1, 3)
    assert "MASK size" in texts[0]
