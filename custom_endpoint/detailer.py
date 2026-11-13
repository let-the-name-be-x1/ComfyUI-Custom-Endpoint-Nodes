"""Region math for the API detailer node: crop -> repaint -> paste.

The node layer (``ce_nodes.CustomEndpointAPIDetailer``) owns the HTTP calls;
everything geometric lives here so it can be tested with numpy alone.

Geometry mirrors Impact Pack's ``MaskDetailer``:

1. the mask is split into 8-connected regions (one API call per region);
2. each region's bounding box is expanded by ``crop_factor`` to keep context
   around the edit, and clamped to the image;
3. the crop is resized so its longest side equals the detail resolution;
4. the model's answer is blended back with a feathered alpha
   (``out = orig * (1 - a) + edited * a``), either over the whole crop
   (``PASTE_CROP``, the default: the ramp sits on the crop border) or over the
   mask only (``PASTE_MASK``: the ramp sits on the mask outline).

Unlike ``MaskDetailer`` there is no sampler: a hosted edit model always repaints
the whole crop it receives, so the mask has to be *communicated* to the model.
That is what the guide painters below do (``GUIDE_*``); the default is
``GUIDE_NONE``, which sends the crop untouched and leaves everything to the
prompt.

All arrays are float32 in 0..1 unless the name says ``_u8``.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Guide modes: how the mask is shown to a model that has no mask field.
# ---------------------------------------------------------------------------
GUIDE_NONE = "none"
GUIDE_OUTLINE = "outline"
GUIDE_DIM = "dim outside"
GUIDE_TINT = "tint inside"
GUIDE_NATIVE = "native mask"

GUIDES = (GUIDE_NONE, GUIDE_OUTLINE, GUIDE_DIM, GUIDE_TINT, GUIDE_NATIVE)

# Where the model's answer is blended back:
#   crop box -> the whole crop is pasted, the alpha ramp sits on the crop border
#   mask     -> only the mask is pasted, the ramp sits on the mask outline
PASTE_CROP = "crop box"
PASTE_MASK = "mask"
PASTE_AREAS = (PASTE_CROP, PASTE_MASK)

# Appended to the user prompt when a visual guide is used.  Kept in English:
# every hosted edit model is strongest there, and the sentence has to read as an
# instruction, not as image content.
GUIDE_HINTS = {
    GUIDE_OUTLINE: (
        "Only edit the area enclosed by the thin yellow outline; every pixel outside "
        "that outline must stay exactly as it is."
    ),
    GUIDE_DIM: (
        "Only edit the bright area; the dimmed area around it must stay exactly as it is."
    ),
    GUIDE_TINT: (
        "Only edit the area covered by the translucent blue tint, and remove that tint "
        "from the result."
    ),
    GUIDE_NATIVE: (
        "Only edit the transparent area of the supplied mask; keep everything else unchanged."
    ),
    GUIDE_NONE: "",
}

MAX_DETAIL_RESOLUTION = 4096
MASK_THRESHOLD = 0.5

# Aspect ratios accepted by the Gemini image config; the closest one is sent so a
# hosted model does not fall back to a square output for a wide crop.
_ASPECT_RATIOS: tuple[tuple[int, int], ...] = (
    (1, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4),
    (9, 16), (16, 9), (21, 9), (1, 4), (4, 1), (8, 1), (1, 8),
)


@dataclass(frozen=True)
class DetailRegion:
    """One area to repaint.

    ``crop_region`` is ``(x1, y1, x2, y2)`` in ORIGINAL image coordinates.
    ``crop`` is ``[h, w, 3]`` float 0-1 and ``mask_crop`` is ``[h, w]`` float
    0-1 (white = repaint), both at ``crop_region`` resolution.
    """

    crop_region: tuple[int, int, int, int]
    crop: np.ndarray
    mask_crop: np.ndarray

    @property
    def width(self) -> int:
        return int(self.crop_region[2] - self.crop_region[0])

    @property
    def height(self) -> int:
        return int(self.crop_region[3] - self.crop_region[1])


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _as_numpy(value: Any) -> np.ndarray:
    """Accept a numpy array, a torch tensor or anything exposing ``numpy()``."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def to_float01(array: Any) -> np.ndarray:
    """Coerce an image-like array to float32 0-1 (uint8 input is divided by 255)."""
    values = np.asarray(array)
    if values.dtype == np.uint8:
        return values.astype(np.float32) / 255.0
    values = values.astype(np.float32)
    if values.size and float(np.nanmax(values)) > 1.0:
        values = values / 255.0
    return np.clip(np.nan_to_num(values), 0.0, 1.0)


def to_uint8(array: Any) -> np.ndarray:
    values = np.asarray(array)
    if values.dtype == np.uint8:
        return values
    return np.clip(np.asarray(to_float01(values)) * 255.0, 0, 255).round().astype(np.uint8)


def frames_of(image: Any) -> list[np.ndarray]:
    """Split an IMAGE-like value into ``[H, W, 3]`` float 0-1 frames."""
    array = _as_numpy(image)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError(
            f"IMAGE must be [H,W,C] or [B,H,W,C]; got shape {np.shape(image)}."
        )
    frames: list[np.ndarray] = []
    for index in range(int(array.shape[0])):
        frame = array[index]
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        frames.append(np.clip(frame[..., :3], 0.0, 1.0).astype(np.float32))
    return frames


def masks_of(mask: Any) -> list[np.ndarray]:
    """Split a MASK-like value into ``[H, W]`` float 0-1 frames."""
    array = _as_numpy(mask)
    if array.ndim == 2:
        return [np.clip(np.nan_to_num(array), 0.0, 1.0)]
    if array.ndim == 4:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(
            f"MASK must be [H,W] or [B,H,W]; got shape {np.shape(mask)}."
        )
    return [
        np.clip(np.nan_to_num(array[index]), 0.0, 1.0)
        for index in range(int(array.shape[0]))
    ]


def mask_to_2d(mask: Any) -> np.ndarray:
    """Normalize a ComfyUI MASK-like value to ``[H, W]`` float 0-1."""
    array = _as_numpy(mask)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3:
        if array.shape[-1] == 1:
            array = array[..., 0]
        else:
            array = array[0]
    if array.ndim != 2:
        raise ValueError(
            f"MASK must be [H,W] or [B,H,W]; got shape {np.asarray(mask).shape}."
        )
    return np.clip(np.nan_to_num(array), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Coordinate helpers (same math as Impact Pack's make_crop_region)
# ---------------------------------------------------------------------------
def normalize_region(limit: int, startp: int, size: float) -> tuple[int, int]:
    if startp < 0:
        return 0, min(limit, int(size))
    if startp + size > limit:
        return max(0, limit - int(size)), int(limit)
    return int(startp), min(limit, int(startp + size))


def make_crop_region(
    width: int,
    height: int,
    bbox: tuple[int, int, int, int],
    crop_factor: float,
) -> tuple[int, int, int, int]:
    """Expand ``bbox`` by ``crop_factor`` around its centre, clamped to the image."""
    x1, y1, x2, y2 = (int(value) for value in bbox)
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    crop_w = max(1.0, bbox_w * float(crop_factor))
    crop_h = max(1.0, bbox_h * float(crop_factor))
    centre_x = x1 + bbox_w / 2.0
    centre_y = y1 + bbox_h / 2.0
    new_x1, new_x2 = normalize_region(width, int(centre_x - crop_w / 2.0), crop_w)
    new_y1, new_y2 = normalize_region(height, int(centre_y - crop_h / 2.0), crop_h)
    return (int(new_x1), int(new_y1), int(new_x2), int(new_y2))


# ---------------------------------------------------------------------------
# Binary morphology + connected components (numpy only, no scipy)
# ---------------------------------------------------------------------------
def dilate_binary(binary: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean mask by ``radius`` pixels (square structuring element)."""
    radius = min(int(radius), 128)
    if radius <= 0:
        return np.array(binary, dtype=bool, copy=True)
    source = np.asarray(binary, dtype=bool)
    height, width = source.shape
    padded = np.zeros((height + 2 * radius, width + 2 * radius), dtype=np.int32)
    padded[radius : radius + height, radius : radius + width] = source.astype(np.int32)
    integral = padded.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    grid = np.zeros((height + 2 * radius + 1, width + 2 * radius + 1), dtype=np.int32)
    grid[1:, 1:] = integral
    k = 2 * radius + 1
    window = (
        grid[k:, k:] - grid[:-k, k:] - grid[k:, :-k] + grid[:-k, :-k]
    )
    return window > 0


def _row_runs(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = binary.shape
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = binary.astype(np.int8)
    edges = np.diff(padded, axis=1)
    starts = np.argwhere(edges == 1)
    ends = np.argwhere(edges == -1)
    return starts, ends


def connected_components(binary: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Label 8-connected components of a boolean mask.

    Returns ``[(x1, y1, x2, y2, area)]`` in a stable (top-left first) order.  The
    implementation is a run-length union-find: it touches one Python object per
    horizontal run instead of per pixel, so a full-frame mask stays fast without
    pulling in scipy.
    """
    source = np.asarray(binary, dtype=bool)
    if source.ndim != 2:
        raise ValueError(f"binary mask must be 2-D, got {source.shape}.")
    if not source.any():
        return []

    starts, ends = _row_runs(source)
    if len(starts) == 0 or len(starts) != len(ends):
        return []

    rows = starts[:, 0]
    run_start = starts[:, 1]
    run_end = ends[:, 1]
    count = len(run_start)

    parent = list(range(count))

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    previous: list[tuple[int, int, int]] = []
    current: list[tuple[int, int, int]] = []
    current_row = None
    for index in range(count):
        row = int(rows[index])
        start = int(run_start[index])
        end = int(run_end[index])
        if row != current_row:
            previous = current
            current = []
            current_row = row
        for prev_start, prev_end, prev_index in previous:
            if prev_start <= end and start <= prev_end:
                union(index, prev_index)
        current.append((start, end, index))

    boxes: dict[int, list[int]] = {}
    for index in range(count):
        root = find(index)
        row = int(rows[index])
        start = int(run_start[index])
        end = int(run_end[index])
        entry = boxes.get(root)
        if entry is None:
            boxes[root] = [start, row, end, row + 1, end - start]
        else:
            entry[0] = min(entry[0], start)
            entry[1] = min(entry[1], row)
            entry[2] = max(entry[2], end)
            entry[3] = max(entry[3], row + 1)
            entry[4] += end - start

    ordered = sorted(boxes.items(), key=lambda item: (item[1][1], item[1][0], item[0]))
    return [(box[0], box[1], box[2], box[3], box[4]) for _, box in ordered]


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------
def regions_from_mask(
    image: Any,
    mask: Any,
    *,
    crop_factor: float = 3.0,
    drop_size: int = 10,
    mask_grow: int = 0,
    threshold: float = MASK_THRESHOLD,
) -> list[DetailRegion]:
    """Split a mask into regions and crop ``image`` around each one.

    ``image`` is ``[H, W, 3]`` (float 0-1 or uint8).  ``mask_grow`` dilates the
    mask before splitting, so the paste area can overshoot a tight selection —
    hosted models never respect the exact boundary.
    """
    array = to_float01(image)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"image must be [H,W,3]; got {np.shape(image)}.")
    array = array[..., :3]
    height, width = int(array.shape[0]), int(array.shape[1])

    mask2d = mask_to_2d(mask)
    if mask2d.shape != (height, width):
        raise ValueError(
            f"MASK size {mask2d.shape} does not match the image {(height, width)}."
        )

    binary = mask2d > float(threshold)
    if int(mask_grow) > 0:
        grown = dilate_binary(binary, int(mask_grow))
        mask2d = np.maximum(mask2d, grown.astype(np.float32))
        binary = grown

    regions: list[DetailRegion] = []
    for x1, y1, x2, y2, area in connected_components(binary):
        if area < int(drop_size):
            continue
        crop_region = make_crop_region(width, height, (x1, y1, x2, y2), crop_factor)
        cx1, cy1, cx2, cy2 = crop_region
        if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
            continue
        regions.append(
            DetailRegion(
                crop_region=crop_region,
                crop=np.ascontiguousarray(array[cy1:cy2, cx1:cx2]),
                mask_crop=np.ascontiguousarray(mask2d[cy1:cy2, cx1:cx2]),
            )
        )
    return regions


# ---------------------------------------------------------------------------
# Resize / encode
# ---------------------------------------------------------------------------
def resize_region(
    crop: np.ndarray, mask: np.ndarray, target: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resize ``(crop, mask)`` so the longest side equals ``target``.

    ``target <= 0`` leaves both untouched.  Unlike Impact Pack's detailer this
    also shrinks (hosted endpoints bill and cap by pixels), and the crop is
    resampled with LANCZOS while the mask stays smooth (BILINEAR).
    """
    crop_u8 = to_uint8(crop)
    mask01 = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    height, width = int(crop_u8.shape[0]), int(crop_u8.shape[1])
    longest = max(height, width)
    target = int(target or 0)
    if target <= 0 or longest == target:
        return crop_u8, mask01
    target = min(target, MAX_DETAIL_RESOLUTION)
    scale = target / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized_crop = np.asarray(
        Image.fromarray(crop_u8, mode="RGB").resize((new_w, new_h), Image.LANCZOS)
    )
    resized_mask = np.asarray(
        Image.fromarray(to_uint8(mask01), mode="L").resize((new_w, new_h), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    return resized_crop, resized_mask


def png_bytes(crop_u8: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(to_uint8(crop_u8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def native_mask_png(mask: np.ndarray) -> bytes:
    """RGBA PNG for endpoints with a real mask field (transparent = editable)."""
    mask01 = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    height, width = int(mask01.shape[0]), int(mask01.shape[1])
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 3] = ((1.0 - mask01) * 255.0).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def paint_guide(
    crop_u8: np.ndarray, mask: np.ndarray, guide: str, pad: int = 0
) -> np.ndarray:
    """Return the crop with the mask made visible for a mask-less edit model.

    ``pad`` pushes the outline / halo away from the mask edge.  A guide drawn
    inside the paste alpha band would be blended back into the picture — that is
    exactly the "yellow ring" artifact, so callers pass at least ``feather + 3``.
    """
    base = to_float01(crop_u8)
    inside = np.asarray(mask, dtype=np.float32) > MASK_THRESHOLD
    if guide == GUIDE_OUTLINE:
        pad = max(0, int(pad))
        inner = dilate_binary(inside, pad) if pad else inside
        ring = dilate_binary(inside, pad + 2) & ~inner
        halo = dilate_binary(inside, pad + 4) & ~dilate_binary(inside, pad + 2)
        base = base.copy()
        base[halo] *= 0.30
        base[ring] = np.array([1.0, 0.85, 0.0], dtype=np.float32)
    elif guide == GUIDE_DIM:
        base = base.copy()
        base[~inside] *= 0.30
    elif guide == GUIDE_TINT:
        base = base.copy()
        base[inside] = base[inside] * 0.55 + np.array([0.10, 0.35, 1.0], dtype=np.float32) * 0.45
    return to_uint8(base)


def aspect_ratio_for(width: int, height: int) -> str:
    """Snap an arbitrary crop aspect to the closest ratio a hosted model accepts."""
    if width <= 0 or height <= 0:
        return "1:1"
    current = math.log(float(width) / float(height))
    best = min(
        _ASPECT_RATIOS,
        key=lambda pair: abs(math.log(pair[0] / pair[1]) - current),
    )
    return f"{best[0]}:{best[1]}"


def guide_pad(feather: int) -> int:
    """Push a guide marker far enough out that the paste alpha is ~0 there."""
    return max(3, int(feather or 0) + 3)


def encode_region(
    region: DetailRegion,
    guide: str,
    target: int,
    feather: int = 0,
) -> tuple[bytes, bytes | None]:
    """Prepare the ``(image_png, mask_png)`` pair for one region.

    With ``GUIDE_NONE`` the crop goes out untouched (no marks of any kind): the
    whole crop is sent and only the prompt guides the edit.
    """
    crop_u8, mask = resize_region(region.crop, region.mask_crop, target)
    if guide == GUIDE_NATIVE:
        return png_bytes(crop_u8), native_mask_png(mask)
    return png_bytes(paint_guide(crop_u8, mask, guide, guide_pad(feather))), None


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------
def feathered_alpha(mask: np.ndarray, feather: int) -> np.ndarray:
    """Gaussian-blurred mask, used as the paste alpha (both sides of the edge)."""
    mask01 = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    radius = int(feather or 0)
    if radius <= 0:
        return mask01
    blurred = Image.fromarray(to_uint8(mask01), mode="L").filter(
        ImageFilter.GaussianBlur(radius=float(radius))
    )
    return np.asarray(blurred, dtype=np.float32) / 255.0


def rect_edge_alpha(
    region: DetailRegion,
    image_shape: Sequence[int],
    feather: int,
) -> np.ndarray:
    """Alpha that fades out at the CROP BORDER (not at the mask outline).

    The ramp lives entirely inside the crop: alpha is 0 on the border, reaches 1
    ``feather`` pixels in, and uses a smoothstep so neither end shows a step.
    Sides that sit on the image border get no ramp (there is nothing to blend
    against there).
    """
    x1, y1, x2, y2 = (int(value) for value in region.crop_region)
    height, width = y2 - y1, x2 - x1
    radius = int(feather or 0)
    if radius <= 0:
        return np.ones((height, width), dtype=np.float32)

    full_h, full_w = int(image_shape[0]), int(image_shape[1])
    ys = np.arange(height, dtype=np.float32)[:, None] + 0.5
    xs = np.arange(width, dtype=np.float32)[None, :] + 0.5
    distance = np.full((height, width), 1e9, dtype=np.float32)
    if x1 > 0:
        distance = np.minimum(distance, xs)
    if y1 > 0:
        distance = np.minimum(distance, ys)
    if x2 < full_w:
        distance = np.minimum(distance, float(width) - xs)
    if y2 < full_h:
        distance = np.minimum(distance, float(height) - ys)
    t = np.clip(distance / float(radius), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def composite(
    composed: np.ndarray,
    region: DetailRegion,
    edited: np.ndarray,
    feather: int,
    paste: str = PASTE_MASK,
) -> None:
    """Blend the model's answer for ``region`` into ``composed`` (in place).

    ``paste=PASTE_CROP`` replaces the whole crop with the model's answer and
    fades the border; ``paste=PASTE_MASK`` keeps only the mask (the surroundings
    stay original) and fades the mask outline.
    """
    x1, y1, x2, y2 = region.crop_region
    target_w, target_h = x2 - x1, y2 - y1

    edited_rgb = to_float01(edited)
    if edited_rgb.ndim == 2:
        edited_rgb = np.repeat(edited_rgb[..., None], 3, axis=-1)
    edited_rgb = edited_rgb[..., :3]
    if edited_rgb.shape[0] != target_h or edited_rgb.shape[1] != target_w:
        edited_rgb = np.asarray(
            Image.fromarray(to_uint8(edited_rgb), mode="RGB").resize(
                (target_w, target_h), Image.LANCZOS
            )
        )
        edited_rgb = to_float01(edited_rgb)

    if paste == PASTE_CROP:
        alpha = rect_edge_alpha(region, composed.shape, feather)[..., None]
    else:
        alpha = feathered_alpha(region.mask_crop, feather)[..., None]
    alpha = alpha.astype(np.float32)
    region_view = composed[y1:y2, x1:x2].astype(np.float32)
    blended = region_view * (1.0 - alpha) + edited_rgb * alpha
    composed[y1:y2, x1:x2] = np.clip(blended, 0.0, 1.0).astype(composed.dtype)


def region_report(region: DetailRegion, index: int) -> dict[str, Any]:
    return {
        "index": int(index),
        "crop_region": [int(value) for value in region.crop_region],
        "crop_size": [region.height, region.width],
    }
