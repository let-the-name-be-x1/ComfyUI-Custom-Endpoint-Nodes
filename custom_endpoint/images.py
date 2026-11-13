"""Image <-> bytes conversion plus the failure placeholder.

Every returned image is normalized to a 3-channel float tensor ``[N, H, W, 3]``
in ``0..1`` so batches coming back from different providers can be stacked and
so a 1x1 placeholder can be mixed into the same output list.
"""

from __future__ import annotations

import base64
import io
from typing import Any

_IMAGE_MAGIC = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",  # jpeg
    b"RIFF",  # webp (RIFF....WEBP)
    b"GIF8",
    b"BM",
)

MAX_INPUT_IMAGES = 9


def _torch():
    import torch  # local import keeps non-image modules torch-free

    return torch


def is_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "dim")


def frame_count(value: Any) -> int:
    """Number of frames in an IMAGE value (batch aware)."""
    if not is_tensor(value):
        return 0
    shape = tuple(int(dimension) for dimension in value.shape)
    if len(shape) == 4:
        return shape[0]
    if len(shape) == 3:
        return 1
    return 0


def iter_frames(value: Any):
    """Yield ``[1, H, W, 3]`` tensors for every frame of an IMAGE value."""
    torch = _torch()
    if not is_tensor(value):
        return
    tensor = value
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    for index in range(int(tensor.shape[0])):
        frame = tensor[index : index + 1]
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        elif frame.shape[-1] == 1:
            frame = frame.repeat(1, 1, 1, 3)
        if not frame.is_floating_point():
            frame = frame.float() / 255.0
        yield frame.clamp(0.0, 1.0).contiguous()


def downscale(frame, total_pixels: int):
    """Downscale a ``[1,H,W,3]`` tensor so ``H*W <= total_pixels`` (aspect kept)."""
    torch = _torch()
    if total_pixels <= 0:
        return frame
    height, width = int(frame.shape[1]), int(frame.shape[2])
    if height * width <= total_pixels:
        return frame
    scale = (total_pixels / float(height * width)) ** 0.5
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    resized = torch.nn.functional.interpolate(
        frame.permute(0, 3, 1, 2), size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    return resized.permute(0, 2, 3, 1).contiguous()


def frame_to_png_bytes(frame, *, total_pixels: int = 0) -> bytes:
    """Encode one ``[1,H,W,3]`` float frame as PNG bytes."""
    from PIL import Image
    import numpy as np

    if total_pixels:
        frame = downscale(frame, total_pixels)
    array = (frame.squeeze(0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_pngs(value: Any, *, max_frames: int = MAX_INPUT_IMAGES, total_pixels: int = 0) -> list[bytes]:
    """Encode an IMAGE value (batch aware) into a list of PNG byte strings."""
    pngs: list[bytes] = []
    for index, frame in enumerate(iter_frames(value)):
        if index >= max_frames:
            break
        pngs.append(frame_to_png_bytes(frame, total_pixels=total_pixels))
    return pngs


def image_to_data_urls(value: Any, *, max_frames: int = MAX_INPUT_IMAGES) -> list[str]:
    return [
        "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        for png in image_to_pngs(value, max_frames=max_frames)
    ]


def mask_to_rgba_png(mask: Any, height: int, width: int) -> bytes:
    """Convert a ComfyUI MASK (``[B,H,W]`` or ``[H,W]``) to the RGBA PNG the
    OpenAI ``/images/edits`` endpoint expects (transparent = editable area)."""
    import numpy as np
    from PIL import Image

    torch = _torch()
    if mask is None:
        raise ValueError("mask is required.")
    tensor = mask
    if tensor.dim() == 3:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().float()
    if tuple(tensor.shape) != (int(height), int(width)):
        resized = torch.nn.functional.interpolate(
            tensor.view(1, 1, tensor.shape[0], tensor.shape[1]),
            size=(int(height), int(width)),
            mode="nearest",
        )
        tensor = resized.view(int(height), int(width))
    alpha = (1.0 - tensor.clamp(0.0, 1.0)).numpy()
    rgba = np.zeros((int(height), int(width), 4), dtype=np.uint8)
    rgba[:, :, 3] = (alpha * 255.0).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def bytes_look_like_image(data: bytes) -> bool:
    return bool(data) and any(data.startswith(magic) for magic in _IMAGE_MAGIC)


def png_bytes_to_tensor(data: bytes):
    """Decode one image (any PIL-readable format) into ``[1,H,W,3]`` float."""
    from PIL import Image
    import numpy as np

    if not bytes_look_like_image(data):
        raise ValueError(
            "The provider returned data that is not a recognizable image "
            f"(first bytes: {data[:16]!r})."
        )
    image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return _torch().from_numpy(array).unsqueeze(0)


# ---------------------------------------------------------------------------
# Batch geometry: one tensor batch has ONE canvas size
# ---------------------------------------------------------------------------
# A ComfyUI "batch" exists in two very different shapes:
#
#   * one ``[B, H, W, 3]`` tensor  -> every frame shares H/W (torch has no
#     ragged dimensions, so this form cannot hold 16:9 and 9:16 at once);
#   * a python list of ``[1, H, W, 3]`` tensors (``OUTPUT_IS_LIST``) -> each
#     item keeps its own size and downstream nodes are called once per item.
#
# This module only builds the tensor form, so merging frames of different sizes
# MUST pick a canvas and a way to fit every frame into it.  Fitting by
# ``stretch`` (what this package and the cores nodes used to do) changes the
# aspect ratio of every frame whose shape differs from the canvas:
SIZE_MODE_MAX = "max size"       # canvas = widest x tallest frame, nothing shrinks
SIZE_MODE_FIRST = "first image"  # canvas = the first frame (the historical behaviour)
SIZE_MODE_MANUAL = "manual"      # canvas = width/height (0 = max size for that side)
SIZE_MODES = (SIZE_MODE_MAX, SIZE_MODE_FIRST, SIZE_MODE_MANUAL)

FIT_CONTAIN = "contain"  # keep aspect, pad the leftover      (no distortion, no loss)
FIT_COVER = "cover"      # keep aspect, crop the overflow     (no distortion, loses edges)
FIT_STRETCH = "stretch"  # squash into the canvas            (the old, distorting behaviour)
FIT_MODES = (FIT_CONTAIN, FIT_COVER, FIT_STRETCH)

POS_CENTER = "center"
POS_TOP_LEFT = "top left"
POS_TOP_RIGHT = "top right"
POS_BOTTOM_LEFT = "bottom left"
POS_BOTTOM_RIGHT = "bottom right"
POSITIONS = (POS_CENTER, POS_TOP_LEFT, POS_TOP_RIGHT, POS_BOTTOM_LEFT, POS_BOTTOM_RIGHT)

PAD_COLORS = {"black": 0.0, "gray": 0.5, "white": 1.0}
PAD_COLOR_NAMES = tuple(PAD_COLORS)


def round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to the next multiple of ``multiple`` (1 = unchanged)."""
    step = max(1, int(multiple or 1))
    return max(step, ((int(value) + step - 1) // step) * step)


def _offsets(free_height: int, free_width: int, position: str) -> tuple[int, int]:
    """Top/left offsets of a smaller box inside a larger one."""
    words = str(position or POS_CENTER).replace("-", " ").replace("_", " ").split()
    if "top" in words:
        top = 0
    elif "bottom" in words:
        top = max(0, int(free_height))
    else:
        top = max(0, int(free_height)) // 2
    if "left" in words:
        left = 0
    elif "right" in words:
        left = max(0, int(free_width))
    else:
        left = max(0, int(free_width)) // 2
    return top, left


def pad_value(pad_color: Any) -> float:
    """``"black"``/``"gray"``/``"white"`` (or a float) -> the fill value."""
    if isinstance(pad_color, str):
        return float(PAD_COLORS.get(pad_color.strip().lower(), 0.0))
    try:
        return float(pad_color)
    except (TypeError, ValueError):
        return 0.0


def _resize(frame, height: int, width: int):
    """Resample a ``[1,H,W,3]`` frame (antialiased when shrinking)."""
    torch = _torch()
    height, width = max(1, int(height)), max(1, int(width))
    if int(frame.shape[1]) == height and int(frame.shape[2]) == width:
        return frame
    shrinking = height * width < int(frame.shape[1]) * int(frame.shape[2])
    resized = torch.nn.functional.interpolate(
        frame.permute(0, 3, 1, 2),
        size=(height, width),
        mode="bilinear" if shrinking else "bicubic",
        align_corners=False,
        antialias=shrinking,
    )
    return resized.permute(0, 2, 3, 1)


def resolve_canvas(
    frames: list[Any],
    *,
    size_mode: str = SIZE_MODE_MAX,
    width: int = 0,
    height: int = 0,
    multiple: int = 1,
) -> tuple[int, int]:
    """``(height, width)`` of the canvas all frames will be fitted into."""
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        return 1, 1
    if str(size_mode) == SIZE_MODE_FIRST:
        canvas_h, canvas_w = int(frames[0].shape[1]), int(frames[0].shape[2])
    else:
        canvas_h = max(int(frame.shape[1]) for frame in frames)
        canvas_w = max(int(frame.shape[2]) for frame in frames)
    if int(width or 0) > 0:
        canvas_w = int(width)
    if int(height or 0) > 0:
        canvas_h = int(height)
    return round_up(canvas_h, multiple), round_up(canvas_w, multiple)


def fit_to_canvas(
    frame,
    height: int,
    width: int,
    *,
    fit: str = FIT_CONTAIN,
    position: str = POS_CENTER,
    pad: float = 0.0,
):
    """Fit one ``[1,H,W,3]`` frame into a ``height x width`` canvas."""
    torch = _torch()
    height, width = max(1, int(height)), max(1, int(width))
    source_h, source_w = int(frame.shape[1]), int(frame.shape[2])
    mode = str(fit or FIT_CONTAIN)

    if mode == FIT_STRETCH:
        return _resize(frame, height, width).contiguous()

    if mode == FIT_COVER:
        scale = max(width / float(source_w), height / float(source_h))
        scaled_w = max(width, int(round(source_w * scale)))
        scaled_h = max(height, int(round(source_h * scale)))
        resized = _resize(frame, scaled_h, scaled_w)
        top, left = _offsets(scaled_h - height, scaled_w - width, position)
        return resized[:, top : top + height, left : left + width, :].contiguous()

    # contain: inscribe the whole frame, pad the leftover
    scale = min(width / float(source_w), height / float(source_h))
    inner_w = min(width, max(1, int(round(source_w * scale))))
    inner_h = min(height, max(1, int(round(source_h * scale))))
    resized = _resize(frame, inner_h, inner_w)
    top, left = _offsets(height - inner_h, width - inner_w, position)
    bottom = height - inner_h - top
    right = width - inner_w - left
    if top or bottom or left or right:
        resized = torch.nn.functional.pad(
            resized, (0, 0, left, right, top, bottom), mode="constant", value=float(pad)
        )
    return resized.contiguous()


def stack_images(
    tensors: list[Any],
    *,
    size_mode: str = SIZE_MODE_MAX,
    width: int = 0,
    height: int = 0,
    fit: str = FIT_CONTAIN,
    position: str = POS_CENTER,
    pad_color: Any = "black",
    multiple: int = 1,
):
    """Stack frames into one ``[N,H,W,3]`` batch on a shared canvas.

    Frames are **never** stretched unless ``fit=FIT_STRETCH`` is asked for: by
    default the canvas is the biggest frame and smaller ones are fitted into it
    with their aspect ratio intact (``contain`` pads, ``cover`` crops).
    """
    torch = _torch()
    frames = [frame for value in tensors for frame in iter_frames(value)]
    if not frames:
        return one_pixel_image()
    canvas_h, canvas_w = resolve_canvas(
        frames, size_mode=size_mode, width=width, height=height, multiple=multiple
    )
    fill = pad_value(pad_color)
    fitted = [
        fit_to_canvas(frame, canvas_h, canvas_w, fit=fit, position=position, pad=fill)
        for frame in frames
    ]
    return torch.cat(fitted, dim=0)


def one_pixel_image():
    """The failure placeholder: a 1x1 black image that keeps the graph valid."""
    return _torch().zeros((1, 1, 1, 3), dtype=_torch().float32)


def describe_bytes(data: bytes, *, limit: int = 40) -> str:
    """Human summary of image bytes for logs ('image/png 8x8, 0.2 KB')."""
    if data is None:
        return "<none>"
    size_kb = len(data) / 1024.0
    kind = "unknown"
    dimensions = ""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            kind = (image.format or "image").lower()
            dimensions = f" {image.width}x{image.height}"
    except Exception:
        if bytes_look_like_image(data):
            kind = "image"
    label = f"{kind}{dimensions}, {size_kb:.1f} KB"
    return label if len(label) <= limit else label[: limit - 3] + "..."


def b64_to_bytes(data: str) -> bytes:
    text = str(data or "").strip()
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    return base64.b64decode(text)
