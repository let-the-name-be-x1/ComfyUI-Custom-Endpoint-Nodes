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


def stack_images(tensors: list[Any]):
    """Stack a list of ``[1,H,W,3]`` tensors, resizing later ones to the first."""
    torch = _torch()
    if not tensors:
        return one_pixel_image()
    frames = [tensor if tensor.dim() == 4 else tensor.unsqueeze(0) for tensor in tensors]
    reference = frames[0]
    height, width = int(reference.shape[1]), int(reference.shape[2])
    normalized = []
    for frame in frames:
        if tuple(frame.shape[-1:]) != (3,):
            frame = frame[..., :3]
        if int(frame.shape[1]) != height or int(frame.shape[2]) != width:
            resized = torch.nn.functional.interpolate(
                frame.permute(0, 3, 1, 2),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            frame = resized.permute(0, 2, 3, 1)
        normalized.append(frame.contiguous())
    return torch.cat(normalized, dim=0)


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
