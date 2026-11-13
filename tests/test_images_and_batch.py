"""Image conversion, placeholders and the batch/concurrency helper."""

import io
import threading
import time

import pytest
from PIL import Image

from custom_endpoint import batch, images
from conftest import content_box, image_tensor, mask_tensor, png_bytes


def test_png_roundtrip_shape_and_range():
    tensor = images.png_bytes_to_tensor(png_bytes(10, 6))
    assert tuple(tensor.shape) == (1, 6, 10, 3)
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_image_to_pngs_expands_batches_and_caps_frames():
    pngs = images.image_to_pngs(image_tensor(batch=3, height=4, width=4))
    assert len(pngs) == 3
    assert all(payload.startswith(b"\x89PNG") for payload in pngs)
    assert len(images.image_to_pngs(image_tensor(batch=5), max_frames=3)) == 3


def test_frame_count_is_batch_aware():
    assert images.frame_count(image_tensor(batch=2)) == 2
    assert images.frame_count(image_tensor(batch=1)[0]) == 1
    assert images.frame_count("not a tensor") == 0


def test_mask_to_rgba_png_marks_the_editable_area():
    # mask 1 = keep, mask 0 = repaint (ComfyUI convention) -> alpha inverted
    kept = Image.open(io.BytesIO(images.mask_to_rgba_png(mask_tensor(1, 8, 8, 1.0), 8, 8)))
    assert kept.mode == "RGBA"
    assert kept.getpixel((0, 0))[3] == 0
    repaint = Image.open(io.BytesIO(images.mask_to_rgba_png(mask_tensor(1, 8, 8, 0.0), 8, 8)))
    assert repaint.getpixel((0, 0))[3] == 255


def test_mask_is_resized_to_the_target_image():
    png = images.mask_to_rgba_png(mask_tensor(1, 4, 4), 8, 6)
    # height=8, width=6 -> PIL reports (width, height)
    assert Image.open(io.BytesIO(png)).size == (6, 8)


def test_one_pixel_placeholder_is_graph_valid():
    tensor = images.one_pixel_image()
    assert tuple(tensor.shape) == (1, 1, 1, 3)
    assert tensor.dtype.is_floating_point


def test_stack_images_aligns_mixed_sizes():
    stacked = images.stack_images([image_tensor(1, 8, 8), image_tensor(2, 4, 4)])
    assert tuple(stacked.shape) == (3, 8, 8, 3)


def test_stack_images_empty_returns_placeholder():
    assert tuple(images.stack_images([]).shape) == (1, 1, 1, 3)


# ---------------------------------------------------------------------------
# canvas geometry: different sizes must never be stretched by default
# ---------------------------------------------------------------------------
def test_contain_keeps_the_aspect_ratio():
    """A 4x8 landscape frame in an 8x8 canvas: padded top/bottom, never squashed."""
    wide = image_tensor(1, 4, 8, 0.25)
    stacked = images.stack_images([image_tensor(1, 8, 8, 0.75), wide])
    assert tuple(stacked.shape) == (2, 8, 8, 3)
    assert content_box(stacked[1], pad=0.0) == (2, 6, 0, 8)  # 4 rows tall, full width
    assert float(stacked[1, 0].abs().max()) == 0.0  # padded rows are the pad colour


def test_contain_pads_when_the_canvas_is_wider():
    tall = image_tensor(1, 8, 4, 0.25)
    stacked = images.stack_images([image_tensor(1, 8, 8, 0.75), tall])
    assert content_box(stacked[1], pad=0.0) == (0, 8, 2, 6)


def test_stretch_is_opt_in_and_still_squashes():
    wide = image_tensor(1, 4, 8, 0.25)
    stacked = images.stack_images(
        [image_tensor(1, 8, 8, 0.75), wide], fit=images.FIT_STRETCH
    )
    # the whole canvas is content -> the 2:1 frame was stretched to 1:1
    assert content_box(stacked[1], pad=0.0) == (0, 8, 0, 8)


def test_cover_crops_the_overflow_instead_of_padding():
    wide = image_tensor(1, 4, 8, 0.25)
    wide[:, :, 0:1] = 1.0  # marker on the left edge, cropped away by cover
    stacked = images.stack_images(
        [image_tensor(1, 8, 8, 0.75), wide], fit=images.FIT_COVER
    )
    assert content_box(stacked[1], pad=0.0) == (0, 8, 0, 8)  # no pad bars at all
    assert float(stacked[1].min()) > 0.2  # the marker is gone -> cropping happened


def test_pad_colour_and_position_are_honoured():
    wide = image_tensor(1, 4, 8, 0.25)
    stacked = images.stack_images(
        [image_tensor(1, 8, 8, 0.75), wide], position=images.POS_TOP_LEFT, pad_color="white"
    )
    assert content_box(stacked[1], pad=1.0) == (0, 4, 0, 8)
    assert float(stacked[1, 7].min()) == 1.0  # bottom rows are white


def test_size_mode_first_and_manual():
    frames = [image_tensor(1, 4, 8, 0.5), image_tensor(1, 8, 4, 0.5)]
    first = images.stack_images(frames, size_mode=images.SIZE_MODE_FIRST)
    assert tuple(first.shape) == (2, 4, 8, 3)
    manual = images.stack_images(frames, size_mode=images.SIZE_MODE_MANUAL, width=10, height=6)
    assert tuple(manual.shape) == (2, 6, 10, 3)
    # width/height = 0 -> that side falls back to the biggest frame
    half = images.stack_images(frames, size_mode=images.SIZE_MODE_MANUAL, height=12)
    assert tuple(half.shape) == (2, 12, 8, 3)


def test_size_multiple_rounds_the_canvas_up():
    frames = [image_tensor(1, 4, 8, 0.5), image_tensor(1, 8, 4, 0.5)]
    stacked = images.stack_images(frames, multiple=6)  # 8x8 canvas -> 12x12
    assert tuple(stacked.shape) == (2, 12, 12, 3)


def test_resolve_canvas_is_pure_geometry():
    frames = [image_tensor(1, 4, 8), image_tensor(1, 8, 4)]
    assert images.resolve_canvas(frames) == (8, 8)
    assert images.resolve_canvas(frames, width=7) == (8, 7)  # explicit width wins
    assert images.resolve_canvas(frames, multiple=16) == (16, 16)
    assert images.resolve_canvas([], size_mode=images.SIZE_MODE_FIRST) == (1, 1)


def test_stack_images_accepts_three_dimensional_frames():
    frame = image_tensor(1, 4, 6)[0]  # [H,W,3]
    stacked = images.stack_images([frame, frame])
    assert tuple(stacked.shape) == (2, 4, 6, 3)


def test_decode_rejects_non_images():
    assert images.bytes_look_like_image(png_bytes())
    with pytest.raises(ValueError):
        images.png_bytes_to_tensor(b"definitely not an image")


def test_broadcast_semantics():
    assert batch.broadcast("a", 3) == ["a", "a", "a"]
    assert batch.broadcast(["a", "b", "c"], 3) == ["a", "b", "c"]
    assert batch.broadcast(["a"], 3) == ["a", "a", "a"]
    with pytest.raises(ValueError):
        batch.broadcast(["a", "b"], 3)


def test_resolve_batch_mixes_lists_and_scalars():
    length, aligned = batch.resolve_batch({"prompt": ["a", "b", "c"], "n": [2], "provider": object()})
    assert length == 3
    assert aligned["n"] == [2, 2, 2]
    assert aligned["prompt"] == ["a", "b", "c"]
    assert len(aligned["provider"]) == 3


def test_resolve_batch_without_lists_is_a_single_job():
    length, aligned = batch.resolve_batch({"prompt": "solo"})
    assert length == 1
    assert aligned["prompt"] == ["solo"]


def test_run_concurrent_keeps_order_and_isolates_failures():
    def worker(item):
        if item == 2:
            raise RuntimeError("boom")
        return item * 10

    results = batch.run_concurrent(worker, [1, 2, 3], 3)
    assert results[0] == 10
    assert isinstance(results[1], RuntimeError)
    assert results[2] == 30


def test_run_concurrent_actually_overlaps():
    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def worker(_):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        return True

    batch.run_concurrent(worker, list(range(4)), 4)
    assert state["peak"] >= 2, "calls did not overlap: concurrency is not working"


def test_run_concurrent_single_worker_is_serial():
    assert batch.run_concurrent(lambda item: item + 1, [1], 4) == [2]
    assert batch.run_concurrent(lambda item: item + 1, [], 4) == []
