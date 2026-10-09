"""Widget spec helpers with the "do not send" sentinel.

The provider node owns the model, so a call node cannot know at
``INPUT_TYPES()`` time which model-specific fields are valid.  Every optional
field therefore carries an explicit omit value and is dropped from the wire
body unless the user picks something:

* combo  -> an extra first option ``不发送``
* int    -> ``-1`` (the widget ``min`` is clamped to ``-1``)
* float  -> ``-1.0``
* string -> ``""``
"""

from __future__ import annotations

from typing import Any, Sequence

OMIT = "不发送"
OMIT_ALIASES = frozenset({OMIT, "omit", "none", "(omit)", "<omit>"})


def combo(options: Sequence[str], default: str | None = None, **opts: Any) -> tuple:
    """A combo whose first option is the omit sentinel."""
    values = (OMIT,) + tuple(str(option) for option in options if str(option) != OMIT)
    spec: dict[str, Any] = {}
    spec.update(opts)
    spec.setdefault("default", default if default in values else OMIT)
    return (values, spec)


def plain_combo(options: Sequence[str], default: str | None = None, **opts: Any) -> tuple:
    values = tuple(str(option) for option in options)
    spec: dict[str, Any] = {}
    spec.update(opts)
    spec.setdefault("default", default if default in values else values[0])
    return (values, spec)


def text(default: str = "", *, multiline: bool = True, **opts: Any) -> tuple:
    spec = {"default": default, "multiline": multiline}
    spec.update(opts)
    return ("STRING", spec)


def optional_string(default: str = "", *, multiline: bool = True, **opts: Any) -> tuple:
    return text(default, multiline=multiline, **opts)


def boolean(default: bool = False, **opts: Any) -> tuple:
    spec = {"default": bool(default)}
    spec.update(opts)
    return ("BOOLEAN", spec)


def optional_int(
    default: int = -1,
    *,
    minimum: int = -1,
    maximum: int = 2147483647,
    step: int = 1,
    **opts: Any,
) -> tuple:
    spec = {"default": int(default), "min": int(minimum), "max": int(maximum), "step": int(step)}
    spec.update(opts)
    return ("INT", spec)


def optional_float(
    default: float = -1.0,
    *,
    minimum: float = -1.0,
    maximum: float = 2.0,
    step: float = 0.01,
    **opts: Any,
) -> tuple:
    spec = {
        "default": float(default),
        "min": float(minimum),
        "max": float(maximum),
        "step": float(step),
    }
    spec.update(opts)
    return ("FLOAT", spec)


def is_omitted(value: Any) -> bool:
    """True when a widget value means "do not send this field"."""
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return float(value) < 0
    if isinstance(value, str):
        return value.strip() == "" or value.strip() in OMIT_ALIASES
    return False


def clean(value: Any) -> Any:
    """Return the value, or ``None`` when it should be omitted."""
    return None if is_omitted(value) else value


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip()
    return text_value or None


def split_csv(value: Any) -> list[str] | None:
    if is_omitted(value):
        return None
    parts = [part.strip() for part in str(value).split(",")]
    parts = [part for part in parts if part]
    return parts or None
