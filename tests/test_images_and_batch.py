"""Image conversion, placeholders and the batch/concurrency helper."""

import io
import threading
import time

import pytest
from PIL import Image

from custom_endpoint import batch, images
from conftest import image_tensor, mask_tensor, png_bytes


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
