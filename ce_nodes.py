"""ComfyUI nodes for the Custom Endpoint API package.

Design rules
------------
* The provider node owns the endpoint, the key and the model.  Call nodes own
  the request parameters only.
* Call nodes call the provider directly (no "request object" hand-off).
* ``INPUT_IS_LIST = True`` on every call node: the whole input batch arrives at
  once and is dispatched concurrently, so ``prompts = [p1, p2, p3]`` produces
  three parallel API calls and list outputs.
* A failed call never raises: it yields a 1x1 image (and the error text) so the
  queue keeps going.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any

import numpy as np

if __package__:
    from .custom_endpoint import api_gemini, api_grok, api_openai, api_qwen, detailer
    from .custom_endpoint import widgets as W
    from .custom_endpoint.batch import resolve_batch, run_concurrent
    from .custom_endpoint.errors import EndpointError, error_json, report_json
    from .custom_endpoint.images import (
        FIT_CONTAIN,
        FIT_MODES,
        MAX_INPUT_IMAGES,
        PAD_COLOR_NAMES,
        POS_CENTER,
        POSITIONS,
        SIZE_MODE_MAX,
        SIZE_MODES,
        frame_count,
        image_to_pngs,
        iter_frames,
        mask_to_rgba_png,
        one_pixel_image,
        png_bytes_to_tensor,
        stack_images,
    )
    from .custom_endpoint.logs import request_log_json, response_log_json
    from .custom_endpoint.provider import (
        API_PROTOCOLS,
        PROTOCOL_GEMINI,
        PROTOCOL_GROK,
        PROTOCOL_OPENAI_CHAT,
        PROTOCOL_OPENAI_IMAGES,
        PROTOCOL_OPENAI_RESPONSES,
        PROTOCOL_QWEN,
        PROTOCOL_QWEN_IMAGES,
        ProviderConfig,
        build_provider,
        require_protocols,
    )
    from .custom_endpoint.transport import HttpClient
else:  # loaded as a top-level module (tests / non-package import)
    from custom_endpoint import api_gemini, api_grok, api_openai, api_qwen, detailer
    from custom_endpoint import widgets as W
    from custom_endpoint.batch import resolve_batch, run_concurrent
    from custom_endpoint.errors import EndpointError, error_json, report_json
    from custom_endpoint.images import (
        FIT_CONTAIN,
        FIT_MODES,
        MAX_INPUT_IMAGES,
        PAD_COLOR_NAMES,
        POS_CENTER,
        POSITIONS,
        SIZE_MODE_MAX,
        SIZE_MODES,
        frame_count,
        image_to_pngs,
        iter_frames,
        mask_to_rgba_png,
        one_pixel_image,
        png_bytes_to_tensor,
        stack_images,
    )
    from custom_endpoint.logs import request_log_json, response_log_json
    from custom_endpoint.provider import (
        API_PROTOCOLS,
        PROTOCOL_GEMINI,
        PROTOCOL_GROK,
        PROTOCOL_OPENAI_CHAT,
        PROTOCOL_OPENAI_IMAGES,
        PROTOCOL_OPENAI_RESPONSES,
        PROTOCOL_QWEN,
        PROTOCOL_QWEN_IMAGES,
        ProviderConfig,
        build_provider,
        require_protocols,
    )
    from custom_endpoint.transport import HttpClient

try:
    import folder_paths
except Exception:  # pragma: no cover - standalone import
    folder_paths = None

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
CATEGORY = "自定义端点 API"
CATEGORY_AUX = "自定义端点 API · 附属"
PROVIDER_TYPE = "CUSTOM_ENDPOINT_PROVIDER"
FILES_TYPE = "CUSTOM_ENDPOINT_FILES"
REFERENCE_SLOTS = tuple(f"image_{index}" for index in range(1, 7))
MAX_MULTIPART_IMAGES = 16
DEFAULT_CONCURRENCY = 4

GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of format, intent, or abstraction—as literal visual "
    "directives for image composition.\nIf a prompt is conversational or lacks specific visual "
    "details, you must creatively invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or "
    "conversational requests."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _first(value: Any) -> Any:
    items = _as_list(value)
    return items[0] if items else None


def _clean_params(job: dict[str, Any], names) -> dict[str, Any]:
    """Read the widget values of one job, dropping every omit sentinel."""
    params: dict[str, Any] = {}
    for name in names:
        value = _first(job.get(name))
        cleaned = W.clean(value)
        if cleaned is not None:
            params[name] = cleaned
    return params


def _collect_images(job: dict[str, Any], names=REFERENCE_SLOTS) -> list[bytes]:
    """Gather the reference image slots in order (gap-free, batch aware)."""
    values = [job.get(name) for name in names]
    connected = [index for index, value in enumerate(values) if value is not None]
    if not connected:
        return []
    last = connected[-1]
    for index in range(last + 1):
        if values[index] is None:
            raise EndpointError(
                f"Reference image inputs must be connected without gaps: connect "
                f"{names[index]} before {names[last]}.",
                error_type="validation",
                error_code="image_gap",
                phase="validate",
            )
    pngs: list[bytes] = []
    for index in range(last + 1):
        pngs.extend(image_to_pngs(values[index], max_frames=MAX_INPUT_IMAGES))
    if len(pngs) > MAX_MULTIPART_IMAGES:
        raise EndpointError(
            f"At most {MAX_MULTIPART_IMAGES} reference images can be sent, got {len(pngs)}.",
            error_type="validation",
            error_code="too_many_images",
            phase="validate",
        )
    return pngs


def _mask_png(job: dict[str, Any], images: list[bytes]) -> bytes | None:
    mask = _first(job.get("mask"))
    if mask is None:
        return None
    if not images:
        raise EndpointError(
            "A mask requires exactly one reference image.",
            error_type="validation",
            error_code="mask_without_image",
            phase="validate",
        )
    if len(images) != 1:
        raise EndpointError(
            "A mask can only be used with exactly one reference image.",
            error_type="validation",
            error_code="mask_multiple_images",
            phase="validate",
        )
    tensor = mask[0] if getattr(mask, "dim", lambda: 0)() == 3 else mask
    height = int(tensor.shape[-2])
    width = int(tensor.shape[-1])
    return mask_to_rgba_png(tensor, height, width)


def _files_parts(job: dict[str, Any], family: str) -> list[dict[str, Any]]:
    payload = _first(job.get("files"))
    if not isinstance(payload, dict):
        return []
    payload_family = payload.get("family")
    if payload_family and payload_family != family:
        raise EndpointError(
            f"The connected file loader is for the '{payload_family}' family but this node "
            f"needs '{family}'. Use the matching file loader node.",
            error_type="validation",
            error_code="file_family_mismatch",
            phase="validate",
        )
    parts = payload.get("parts") or []
    return [part for part in parts if isinstance(part, dict)]


def _new_client() -> HttpClient:
    """HTTP client factory — tests monkeypatch this to inject a fake opener."""
    return HttpClient()


class _ProviderResolver:
    """Resolve each provider's session handle exactly once per queue item."""
    def __init__(self) -> None:
        self._cache: dict[int, ProviderConfig] = {}

    def resolve(self, provider: Any) -> ProviderConfig:
        if isinstance(provider, ProviderConfig):
            cached = self._cache.get(id(provider))
            if cached is not None:
                return cached
            if provider.config_error:
                self._cache[id(provider)] = provider
                return provider
            try:
                resolved = replace(provider, api_key=provider.credential())
            except Exception as error:  # noqa: BLE001
                resolved = replace(provider, api_key="", config_error=str(error))
            self._cache[id(provider)] = resolved
            return resolved
        reason = (
            "No provider connected." if provider is None else f"Unsupported provider value: {type(provider).__name__}."
        )
        return ProviderConfig(
            base_url="",
            model="",
            api_key="",
            config_error=reason,
        )


# ---------------------------------------------------------------------------
# Provider node
# ---------------------------------------------------------------------------
class CustomEndpointProvider:
    """Endpoint + API key + model, shared by every call node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "API 基础地址，例如 https://api.example.com/v1 。"
                        "未写版本号时自动补 /v1（gemini 协议补 /v1beta）。",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "API 密钥。加密导出开启时不会写进工作流文件；"
                        "图片元数据在任何情况下都不含密钥。",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "调用的模型 id，例如 gpt-image-2.5-flare / gemini-3.1-flash-image / "
                        "grok-imagine-image-2.0 。",
                    },
                ),
                "api_protocol": (
                    tuple(str(item) for item in API_PROTOCOLS),
                    {"default": "auto", "tooltip": "auto 会按模型名推断协议；不匹配时报错在调用节点的日志里。"},
                ),
            },
            "optional": {
                "encrypt_export": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "加密导出（抹除端点与密钥）",
                        "tooltip": "开启：导出工作流时把本节点的端点与密钥清空（重载后需重新填写）；"
                        "关闭：导出保留端点与密钥。图片元数据始终不含端点与密钥。",
                    },
                ),
                "timeout_sec": (
                    "INT",
                    {"default": 300, "min": 5, "max": 7200, "step": 1, "tooltip": "单次请求超时（秒）。"},
                ),
                "retries": (
                    "INT",
                    {"default": 1, "min": 0, "max": 5, "step": 1, "tooltip": "网络/限流失败时的重试次数。"},
                ),
                "extra_headers": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": '额外请求头，JSON 对象，例如 {"X-Foo":"bar"} 。',
                    },
                ),
                "provider_name": ("STRING", {"default": "provider", "multiline": False}),
            },
        }

    RETURN_TYPES = (PROVIDER_TYPE,)
    RETURN_NAMES = ("provider",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        endpoint,
        api_key,
        model,
        api_protocol="auto",
        encrypt_export=True,
        timeout_sec=300,
        retries=1,
        extra_headers="",
        provider_name="provider",
    ):
        provider = build_provider(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            api_protocol=str(api_protocol),
            encrypt_export=bool(encrypt_export),
            timeout_sec=int(timeout_sec),
            retries=int(retries),
            extra_headers=extra_headers,
            provider_name=provider_name,
        )
        return (provider,)


# ---------------------------------------------------------------------------
# Call node base classes
# ---------------------------------------------------------------------------
class _BaseCallNode:
    CATEGORY = CATEGORY
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    LABEL = "call node"
    ALLOWED_PROTOCOLS: tuple[str, ...] = ()
    RESULT_KEYS: tuple[str, ...] = ("images", "texts", "request_log", "response_log")
    RETURN_NAMES = RESULT_KEYS
    OUTPUT_IS_LIST = tuple([True] * len(RESULT_KEYS))

    # -- subclass contract ------------------------------------------------
    def _build_request(self, provider: ProviderConfig, job: dict[str, Any]):
        raise NotImplementedError

    def _parse(self, provider: ProviderConfig, response: Any, client: HttpClient):
        raise NotImplementedError

    def _request_alternative(
        self, provider: ProviderConfig, job: dict[str, Any], request: Any, response: Any
    ):
        """Return a replacement request when the endpoint rejected this surface.

        Used by protocols that are served in more than one wire shape (a relay
        answering 404 'Invalid URL' for the native Qwen path).  Returning
        ``None`` keeps the original response authoritative.
        """
        return None

    @property
    def _wants_images(self) -> bool:
        return "images" in self.RESULT_KEYS

    # -- execution --------------------------------------------------------
    def execute(self, **kwargs):
        length, aligned = resolve_batch(dict(kwargs))
        jobs = []
        for index in range(length):
            job = {
                name: (values[index] if index < len(values) else values[-1])
                for name, values in aligned.items()
            }
            job["_index"] = index
            jobs.append(job)

        resolver = _ProviderResolver()
        concurrency = int(_first(aligned.get("max_concurrency")) or DEFAULT_CONCURRENCY)

        def worker(job):
            return self._run_one(job, resolver)

        results = run_concurrent(worker, jobs, concurrency)
        normalized = [
            result if isinstance(result, dict) else self._failure(job, results[index], resolver)
            for index, (result, job) in enumerate(zip(results, jobs))
        ]
        return tuple([result[key] for result in normalized] for key in self.RESULT_KEYS)

    def _run_one(self, job: dict[str, Any], resolver: _ProviderResolver) -> dict[str, Any]:
        provider = resolver.resolve(job.get("provider"))
        request = None
        secrets = (provider.credential(),) if provider.api_key else ()
        try:
            require_protocols(provider, self.ALLOWED_PROTOCOLS, node_label=self.LABEL)
            request = self._build_request(provider, job)
            request_log = request_log_json(
                provider=provider,
                method=request.method,
                url=request.url,
                headers=request.headers,
                body=request.effective_log_body(),
                image_summary=request.image_summary,
                operation=request.operation,
                item_index=job["_index"],
            )
        except Exception as error:  # noqa: BLE001 - per-item failure is data
            return self._failure(job, error, resolver, provider=provider, request=request)

        try:
            client = _new_client()
            response = client.request(
                request.method,
                request.url,
                headers=request.headers,
                json_body=request.json_body,
                body=request.body,
                content_type=request.content_type,
                timeout=provider.timeout_sec,
                retries=provider.retries,
                phase=request.operation or "request",
            )
        except Exception as error:  # noqa: BLE001
            return self._failure(
                job, error, resolver, provider=provider, request=request, request_log=request_log
            )

        fallback = None
        if not response.ok:
            try:
                alternative = self._request_alternative(provider, job, request, response)
            except Exception:  # noqa: BLE001 - the original response stays authoritative
                alternative = None
            if alternative is not None:
                fallback = {
                    "reason": "the endpoint does not serve the request's route",
                    "first_status": response.status,
                    "first_url": request.url,
                    "first_error": response.excerpt(300),
                    "url": alternative.url,
                }
                try:
                    request = alternative
                    request_log = request_log_json(
                        provider=provider,
                        method=request.method,
                        url=request.url,
                        headers=request.headers,
                        body=request.effective_log_body(),
                        image_summary=request.image_summary,
                        operation=request.operation,
                        item_index=job["_index"],
                        extra={"fallback": fallback},
                    )
                    response = client.request(
                        request.method,
                        request.url,
                        headers=request.headers,
                        json_body=request.json_body,
                        body=request.body,
                        content_type=request.content_type,
                        timeout=provider.timeout_sec,
                        retries=provider.retries,
                        phase=request.operation or "request",
                    )
                except Exception as error:  # noqa: BLE001
                    return self._failure(
                        job, error, resolver, provider=provider, request=request,
                        request_log=request_log,
                    )

        try:
            parsed = self._parse(provider, response, client)
        except Exception as error:  # noqa: BLE001
            return self._failure(
                job,
                error,
                resolver,
                provider=provider,
                request=request,
                request_log=request_log,
                response=response,
            )

        images = self._tensors(parsed.images)
        thought = self._tensors(parsed.thought_images)
        texts = parsed.text or (parsed.revised_prompts[0] if parsed.revised_prompts else "")
        parsed_report = {
            "image_count": len(parsed.images),
            "thought_image_count": len(parsed.thought_images),
            "text": parsed.text[:4000] if parsed.text else "",
            "revised_prompts": parsed.revised_prompts[:4],
            "image_urls": parsed.image_urls[:4],
        }
        parsed_report.update(parsed.details)
        response_log = response_log_json(
            provider=provider,
            response=response,
            parsed=parsed_report,
            item_index=job["_index"],
            extra={"fallback": fallback} if fallback else None,
        )
        return {
            "images": images,
            "thought_images": thought,
            "texts": texts,
            "request_log": request_log,
            "response_log": response_log,
        }

    def _tensors(self, payloads: list[bytes]):
        if not payloads:
            return one_pixel_image()
        return stack_images([png_bytes_to_tensor(payload) for payload in payloads])

    def _failure(
        self,
        job: dict[str, Any],
        error: Any,
        resolver: _ProviderResolver,
        *,
        provider: ProviderConfig | None = None,
        request: Any = None,
        request_log: str | None = None,
        response: Any = None,
    ) -> dict[str, Any]:
        if not isinstance(error, BaseException):
            error = EndpointError(f"Unexpected worker result: {error!r}", phase="worker")
        if provider is None:
            provider = resolver.resolve(job.get("provider"))
        secrets = (provider.credential(),) if provider.api_key else ()
        message = getattr(error, "message", None) or str(error) or error.__class__.__name__

        if request_log is None:
            if request is not None:
                try:
                    request_log = request_log_json(
                        provider=provider,
                        method=request.method,
                        url=request.url,
                        headers=request.headers,
                        body=request.effective_log_body(),
                        image_summary=request.image_summary,
                        operation=request.operation,
                        item_index=job.get("_index"),
                        extra={"error": message},
                    )
                except Exception:  # noqa: BLE001
                    request_log = error_json(error, known_secrets=secrets, phase="request")
            else:
                request_log = error_json(
                    error,
                    known_secrets=secrets,
                    phase=getattr(error, "phase", None) or "request",
                    protocol=getattr(provider, "api_protocol", None),
                )

        try:
            response_log = response_log_json(
                provider=provider,
                response=response,
                parsed={"error": message},
                item_index=job.get("_index"),
                ok=False,
                extra={"error_detail": getattr(error, "report", lambda **_: {})()},
            )
        except Exception:  # noqa: BLE001
            response_log = error_json(error, known_secrets=secrets, phase="response")

        return {
            "images": one_pixel_image(),
            "thought_images": one_pixel_image(),
            "texts": message,
            "request_log": request_log,
            "response_log": response_log,
        }

    # -- shared widget blocks --------------------------------------------
    @staticmethod
    def _common_widgets():
        return {
            "max_concurrency": (
                "INT",
                {
                    "default": DEFAULT_CONCURRENCY,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "批次并发数（输入为列表时并行调用）。",
                },
            ),
        }


class _ImageCallNode(_BaseCallNode):
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")

    @staticmethod
    def _image_inputs(max_slots: int = 6):
        inputs: dict[str, Any] = {
            name: ("IMAGE", {"tooltip": "参考图；连上后会自动出现下一个槽位。批量输入按帧展开。"})
            for name in REFERENCE_SLOTS[:max_slots]
        }
        return inputs


# ---------------------------------------------------------------------------
# GPT image
# ---------------------------------------------------------------------------
class CustomEndpointGPTImage(_ImageCallNode):
    LABEL = "自定义端点 · GPT 图像"
    ALLOWED_PROTOCOLS = (PROTOCOL_OPENAI_IMAGES,)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "provider": (PROVIDER_TYPE,),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "图像提示词。"}),
            "size": (
                ("auto", "1024x1024", "1024x1536", "1536x1024", "2048x2048", "2048x1152",
                 "1152x2048", "3840x2160", "2160x3840", "Custom"),
                {"default": "auto", "tooltip": "输出尺寸；Custom 时使用下面两个自定义尺寸。"},
            ),
            "custom_width": ("INT", {"default": 1024, "min": 480, "max": 3840, "step": 16}),
            "custom_height": ("INT", {"default": 1024, "min": 480, "max": 3840, "step": 16}),
            "background": W.combo(("auto", "opaque", "transparent")),
            "quality": W.combo(("low", "medium", "high", "xhigh", "max"), "low"),
            "output_format": W.combo(("png", "jpeg", "webp")),
            "output_compression": W.optional_int(-1, minimum=-1, maximum=100),
            "moderation": W.combo(("auto", "low")),
            "response_format": W.combo(("url", "b64_json")),
            "n": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "生成几张图。"}),
            "seed": W.optional_int(-1, minimum=-1, maximum=2147483647, tooltip="部分中转支持；默认不发送。"),
        }
        required.update(cls._common_widgets())
        optional = cls._image_inputs()
        optional["mask"] = ("MASK", {"tooltip": "局部重绘遮罩（仅 GPT 图像的原生 mask 字段）。"})
        return {"required": required, "optional": optional}

    def _build_request(self, provider, job):
        images = _collect_images(job)
        mask = _mask_png(job, images)
        params = _clean_params(
            job,
            ("size", "custom_width", "custom_height", "background", "quality", "output_format",
             "output_compression", "moderation", "response_format", "n", "seed"),
        )
        if params.get("size") == "Custom":
            params["custom_width"] = int(_first(job.get("custom_width")) or 1024)
            params["custom_height"] = int(_first(job.get("custom_height")) or 1024)
        operation = "edit" if images or mask is not None else "generate"
        summary = {"reference_images": [len(png) for png in images]} if images else {}
        if mask is not None:
            summary["mask"] = len(mask)
        return api_openai.prepare_image_request(
            provider,
            operation=operation,
            prompt=str(_first(job.get("prompt")) or ""),
            images=images,
            mask=mask,
            params=params,
            image_summary=summary,
        )

    def _parse(self, provider, response, client):
        return api_openai.parse_image_response(provider, response, client, timeout=provider.timeout_sec)


# ---------------------------------------------------------------------------
# Gemini image
# ---------------------------------------------------------------------------
class CustomEndpointGeminiImage(_BaseCallNode):
    LABEL = "自定义端点 · Gemini 图像"
    ALLOWED_PROTOCOLS = (PROTOCOL_GEMINI,)
    RESULT_KEYS = ("images", "thought_images", "texts", "request_log", "response_log")
    RETURN_NAMES = RESULT_KEYS
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    OUTPUT_IS_LIST = (True, True, True, True, True)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "provider": (PROVIDER_TYPE,),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "图像提示词。"}),
            "aspect_ratio": W.plain_combo(
                ("auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9",
                 "21:9", "1:4", "4:1", "8:1", "1:8"),
                "auto",
            ),
            "resolution": W.combo(("1K", "2K", "4K"), "1K"),
            "thinking_level": W.combo(("MINIMAL", "HIGH")),
            "response_modalities": W.combo(("IMAGE", "IMAGE+TEXT"), "IMAGE"),
            "mime_type": W.combo(("image/png", "image/jpeg", "image/webp")),
            "compression_quality": W.optional_int(-1, minimum=-1, maximum=100),
            "temperature": W.optional_float(-1.0, minimum=-1.0, maximum=2.0),
            "top_p": W.optional_float(-1.0, minimum=-1.0, maximum=1.0),
            "seed": W.optional_int(-1, minimum=-1, maximum=2147483647),
            "system_prompt": (
                "STRING",
                {"default": GEMINI_IMAGE_SYS_PROMPT, "multiline": True, "tooltip": "系统指令。"},
            ),
        }
        required.update(cls._common_widgets())
        optional = {
            name: ("IMAGE", {"tooltip": "参考图；连上后会自动出现下一个槽位。"})
            for name in REFERENCE_SLOTS
        }
        optional["files"] = (FILES_TYPE, {"tooltip": "来自「Gemini 文件加载」节点。"})
        return {"required": required, "optional": optional}

    def _build_request(self, provider, job):
        images = _collect_images(job)
        params = _clean_params(
            job,
            ("aspect_ratio", "resolution", "thinking_level", "response_modalities", "mime_type",
             "compression_quality", "temperature", "top_p", "seed", "system_prompt"),
        )
        files = _files_parts(job, "gemini")
        return api_gemini.prepare_image_request(
            provider,
            operation="generate",
            prompt=str(_first(job.get("prompt")) or ""),
            images=images,
            params={**params, "files": files},
            image_summary={"reference_images": [len(png) for png in images]} if images else {},
        )

    def _parse(self, provider, response, client):
        return api_gemini.parse_response(
            provider, response, client, timeout=provider.timeout_sec, want_images=True
        )


# ---------------------------------------------------------------------------
# Grok image
# ---------------------------------------------------------------------------
class CustomEndpointGrokImage(_ImageCallNode):
    LABEL = "自定义端点 · Grok 图像"
    ALLOWED_PROTOCOLS = (PROTOCOL_GROK,)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "provider": (PROVIDER_TYPE,),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "图像提示词。"}),
            "aspect_ratio": W.combo(
                ("auto", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "9:19.5",
                 "19.5:9", "9:20", "20:9", "1:2", "2:1"),
                "1:1",
            ),
            "resolution": W.combo(("1K", "2K"), "1K"),
            "quality": W.combo(("medium", "low"), "medium"),
            "response_format": W.combo(("url", "b64_json")),
            "number_of_images": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            "seed": W.optional_int(-1, minimum=-1, maximum=2147483647),
        }
        required.update(cls._common_widgets())
        return {"required": required, "optional": cls._image_inputs()}

    def _build_request(self, provider, job):
        images = _collect_images(job)
        params = _clean_params(
            job,
            ("aspect_ratio", "resolution", "quality", "response_format", "number_of_images", "seed"),
        )
        params["n"] = params.pop("number_of_images", 1)
        operation = "edit" if images else "generate"
        return api_grok.prepare_image_request(
            provider,
            operation=operation,
            prompt=str(_first(job.get("prompt")) or ""),
            images=images,
            params=params,
            image_summary={"reference_images": [len(png) for png in images]} if images else {},
        )

    def _parse(self, provider, response, client):
        return api_grok.parse_image_response(provider, response, client, timeout=provider.timeout_sec)


# ---------------------------------------------------------------------------
# Qwen image (DashScope multimodal-generation)
# ---------------------------------------------------------------------------
QWEN_SIZE_MODES = ("match input", "auto", "custom")
QWEN_PROTOCOLS = (PROTOCOL_QWEN, PROTOCOL_QWEN_IMAGES)
QWEN_SIZE_TOOLTIP = (
    "输出分辨率：match input 沿用第一张参考图的尺寸（会自动收进允许的像素范围）；"
    "auto 由服务端按输入比例选择；custom 使用下面的宽高。"
)


class _QwenImageBase(_ImageCallNode):
    """Shared wiring for the two Qwen surfaces (native / OpenAI-compatible)."""

    ALLOWED_PROTOCOLS = QWEN_PROTOCOLS
    OPERATION = "generate"

    def _build_request(self, provider, job, dialect=None):
        """``dialect`` is only passed when retrying on another wire shape."""
        raise NotImplementedError

    def _request_alternative(self, provider, job, request, response):
        """A relay 404s the DashScope path; retry once on ``/images/*``."""
        if provider.api_protocol != PROTOCOL_QWEN or not api_qwen.is_missing_route(response):
            return None
        return self._build_request(provider, job, api_qwen.DIALECT_OPENAI_IMAGES)

    def _parse(self, provider, response, client):
        return api_qwen.parse_image_response(provider, response, client, timeout=provider.timeout_sec)


class CustomEndpointQwenImage(_QwenImageBase):
    """Text to image, mirroring the partner node `Qwen Image 3 Text to Image`."""

    LABEL = "自定义端点 · Qwen 图像"
    OPERATION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "provider": (PROVIDER_TYPE,),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "图像提示词（中英文均可）。"}),
            "negative_prompt": (
                "STRING",
                {"default": "", "multiline": True, "tooltip": "反向提示词；留空则不发送该字段。"},
            ),
            "width": (
                "INT",
                {
                    "default": 1024,
                    "min": 256,
                    "max": 2560,
                    "step": 16,
                    "tooltip": "输出宽度。总像素需在 512×512 与 2560×2560 之间（中转上游常见上限 2048×2048），"
                    "宽高比 1:8–8:1。",
                },
            ),
            "height": (
                "INT",
                {
                    "default": 1024,
                    "min": 256,
                    "max": 2560,
                    "step": 16,
                    "tooltip": "输出高度。总像素需在 512×512 与 2560×2560 之间（中转上游常见上限 2048×2048），"
                    "宽高比 1:8–8:1。",
                },
            ),
            "n": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1, "tooltip": "生成张数，一次请求返回一个批次。"}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "step": 1, "tooltip": "随机种子。"}),
            "prompt_extend": ("BOOLEAN", {"default": True, "tooltip": "是否让服务端 AI 扩写提示词。"}),
            "watermark": ("BOOLEAN", {"default": False, "tooltip": "是否加 AI 生成水印。"}),
        }
        required.update(cls._common_widgets())
        return {"required": required}

    def _build_request(self, provider, job, dialect=None):
        params = _clean_params(
            job,
            ("negative_prompt", "width", "height", "n", "seed", "prompt_extend", "watermark"),
        )
        return api_qwen.prepare_image_request(
            provider,
            operation="generate",
            prompt=str(_first(job.get("prompt")) or ""),
            params=params,
            dialect=dialect,
        )


class CustomEndpointQwenImageEdit(_QwenImageBase):
    """Prompt-driven edit of 1-3 reference images, mirroring `Qwen Image 3 Edit`."""

    LABEL = "自定义端点 · Qwen 图像编辑"
    OPERATION = "edit"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "provider": (PROVIDER_TYPE,),
            "prompt": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "tooltip": "编辑指令（中英文均可）；用 @Image1 / @Image2 引用参考图，"
                    "按连线顺序从 1 开始编号（批量输入按帧计数）。",
                },
            ),
            "negative_prompt": (
                "STRING",
                {"default": "", "multiline": True, "tooltip": "反向提示词；留空则不发送该字段。"},
            ),
            "size_mode": W.plain_combo(QWEN_SIZE_MODES, "match input", tooltip=QWEN_SIZE_TOOLTIP),
            "width": (
                "INT",
                {"default": 1024, "min": 256, "max": 2560, "step": 16, "tooltip": "size_mode=custom 时的输出宽度。"},
            ),
            "height": (
                "INT",
                {"default": 1024, "min": 256, "max": 2560, "step": 16, "tooltip": "size_mode=custom 时的输出高度。"},
            ),
            "n": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1, "tooltip": "生成张数，一次请求返回一个批次。"}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "step": 1, "tooltip": "随机种子。"}),
            "prompt_extend": ("BOOLEAN", {"default": True, "tooltip": "是否让服务端 AI 扩写提示词。"}),
            "watermark": ("BOOLEAN", {"default": False, "tooltip": "是否加 AI 生成水印。"}),
        }
        required.update(cls._common_widgets())
        optional = cls._image_inputs(api_qwen.MAX_EDIT_IMAGES)
        return {"required": required, "optional": optional}

    def _build_request(self, provider, job, dialect=None):
        images = _collect_images(job, REFERENCE_SLOTS[: api_qwen.MAX_EDIT_IMAGES])
        params = _clean_params(
            job,
            ("negative_prompt", "width", "height", "n", "seed", "prompt_extend", "watermark"),
        )
        params["size_mode"] = str(_first(job.get("size_mode")) or "match input")
        return api_qwen.prepare_image_request(
            provider,
            operation="edit",
            prompt=str(_first(job.get("prompt")) or ""),
            images=images,
            params=params,
            image_summary={"reference_images": [len(png) for png in images]} if images else {},
            dialect=dialect,
        )


# ---------------------------------------------------------------------------
# API detailer: mask -> crop -> hosted edit -> feathered paste
# ---------------------------------------------------------------------------
API_DETAILER_PROTOCOLS = (
    PROTOCOL_GEMINI,
    PROTOCOL_OPENAI_IMAGES,
    PROTOCOL_GROK,
    PROTOCOL_QWEN,
    PROTOCOL_QWEN_IMAGES,
)


def _placeholder(tensor) -> bool:
    """True when a per-region result is the 1x1 failure placeholder."""
    shape = tuple(int(value) for value in tensor.shape)
    return len(shape) == 4 and shape[1] == 1 and shape[2] == 1


def _stack_frames(frames):
    """Compose a list of float 0-1 HWC frames into one IMAGE tensor."""
    import torch

    array = np.stack([np.clip(frame, 0.0, 1.0).astype(np.float32) for frame in frames])
    return torch.from_numpy(array)


def _frame_tensor(frame):
    """Wrap one float 0-1 HWC frame into a single-image IMAGE tensor."""
    import torch

    return torch.from_numpy(np.clip(np.asarray(frame, np.float32), 0.0, 1.0))[None]


def _detailer_prompt(prompt: Any, guide: str) -> str:
    """User prompt plus the sentence that explains the visual mask guide."""
    text = str(prompt or "").strip()
    hint = detailer.GUIDE_HINTS.get(guide, "")
    if not hint:
        return text
    return f"{text}\n\n{hint}" if text else hint


class CustomEndpointAPIDetailer(_BaseCallNode):
    """Mask-driven local repaint performed by a hosted image-edit model.

    The pipeline is Impact Pack's ``MaskDetailer`` with the diffusion sampler
    replaced by one API call per region:

    ``mask -> connected regions -> crop (bbox x crop_factor) -> resize ->
    edit -> resize back -> feathered paste``

    With ``mask_guide=none`` (default) the crop goes out **untouched** — no mask,
    no markers — and only the prompt guides the edit.  ``paste_area`` decides
    where the answer is blended back: the whole crop with a ramp on the crop
    border (default), or the mask only.
    """

    LABEL = "自定义端点 · API 局部重绘"
    ALLOWED_PROTOCOLS = API_DETAILER_PROTOCOLS
    RESULT_KEYS = ("images", "texts", "request_log", "response_log", "cropped")
    RETURN_NAMES = RESULT_KEYS
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "IMAGE")
    OUTPUT_IS_LIST = (True, True, True, True, True)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "provider": (PROVIDER_TYPE,),
            "image": ("IMAGE", {"tooltip": "待修复的原图；批量时逐帧处理。"}),
            "mask": (
                "MASK",
                {"tooltip": "白色 = 要重绘的区域。按连通域切成若干块，每块发一次编辑请求。"},
            ),
            "prompt": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "tooltip": "只描述遮罩里应该出现什么（+风格），不需要写全图：周边像素本身就是上下文。",
                },
            ),
            "crop_factor": (
                "FLOAT",
                {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "裁剪框 = 遮罩外接框 × 该倍数；越大上下文越多、边缘越容易接上，但每像素细节更少。",
                },
            ),
            "target_resolution": (
                "INT",
                {
                    "default": 1024,
                    "min": 0,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "发给模型的裁剪块长边像素（会放大也会缩小）；0 = 保持原始裁剪尺寸。",
                },
            ),
            "feather": (
                "INT",
                {
                    "default": 5,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "贴回时混合带的宽度（原图像素），作用位置由 paste_area 决定："
                    "crop box → 矩形框四边；mask → 遮罩边缘（对称羽化，向内也会侵蚀）。0 = 硬切。",
                },
            ),
            "drop_size": (
                "INT",
                {
                    "default": 10,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "面积小于该像素数的连通域直接忽略（过滤噪点小斑）。",
                },
            ),
            "mask_grow": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": "先把遮罩向外膨胀这么多像素再切块：模型不会严格遵守边界，留点余量更稳。",
                },
            ),
            "paste_area": W.plain_combo(
                detailer.PASTE_AREAS,
                detailer.PASTE_CROP,
                tooltip=(
                    "模型结果贴回哪里：crop box = 整块裁剪贴回，羽化落在矩形框四边"
                    "（接缝离开主体轮廓，默认）；mask = 只贴遮罩内，羽化落在遮罩边缘。"
                ),
            ),
            "mask_guide": W.plain_combo(
                detailer.GUIDES,
                detailer.GUIDE_NONE,
                tooltip=(
                    "如何把遮罩告诉模型：none = 原样发送、只靠提示词（默认）；"
                    "outline = 黄框描边；dim outside = 圈外压暗；tint inside = 圈内染色；"
                    "native mask = 原生 mask 字段（仅 GPT 图像）。"
                    "paste_area=crop box 时这些标记会被忽略（否则会被贴进画面）。"
                ),
            ),
        }
        required.update(cls._common_widgets())
        return {"required": required}

    # -- request / parse ---------------------------------------------------
    @staticmethod
    def _region_params(provider: ProviderConfig, region) -> dict[str, Any]:
        # Only Gemini exposes an aspect-ratio knob; Grok rejects a custom ratio
        # for a single source image and the others derive it from the input.
        if provider.api_protocol == PROTOCOL_GEMINI:
            return {
                "aspect_ratio": detailer.aspect_ratio_for(region.width, region.height),
                "mime_type": "image/png",
            }
        return {}

    def _build_request(self, provider, job):
        region = job["_region"]
        guide = str(job.get("_guide") or detailer.GUIDE_NONE)
        target = int(job.get("_target") or 0)
        image_png, mask_png = detailer.encode_region(
            region, guide, target, int(job.get("_feather") or 0)
        )
        if mask_png is not None and provider.api_protocol != PROTOCOL_OPENAI_IMAGES:
            raise EndpointError(
                "mask_guide='native mask' 只有 GPT 图像协议（/images/edits）支持；"
                "Gemini / Grok / Qwen 请改用 outline、dim outside、tint inside 或 none。",
                error_type="validation",
                error_code="mask_guide_unsupported",
                phase="validate",
            )
        prompt = _detailer_prompt(job.get("_prompt"), guide)
        params = self._region_params(provider, region)
        summary = {
            "reference_images": [len(image_png)],
            "region": [int(value) for value in region.crop_region],
            "guide": guide,
        }
        return _prepare_edit_request(provider, prompt, image_png, mask_png, params, summary)

    def _parse(self, provider, response, client):
        if provider.api_protocol == PROTOCOL_GEMINI:
            return api_gemini.parse_response(
                provider, response, client, timeout=provider.timeout_sec, want_images=True
            )
        if provider.api_protocol == PROTOCOL_OPENAI_IMAGES:
            return api_openai.parse_image_response(
                provider, response, client, timeout=provider.timeout_sec
            )
        if provider.api_protocol == PROTOCOL_GROK:
            return api_grok.parse_image_response(
                provider, response, client, timeout=provider.timeout_sec
            )
        return api_qwen.parse_image_response(
            provider, response, client, timeout=provider.timeout_sec
        )

    # -- execution ---------------------------------------------------------
    def execute(self, **kwargs):
        length, aligned = resolve_batch(dict(kwargs))
        jobs = []
        for index in range(length):
            job = {
                name: (values[index] if index < len(values) else values[-1])
                for name, values in aligned.items()
            }
            job["_index"] = index
            jobs.append(job)

        resolver = _ProviderResolver()
        concurrency = int(_first(aligned.get("max_concurrency")) or DEFAULT_CONCURRENCY)
        results = run_concurrent(lambda job: self._run_frame(job, resolver), jobs, concurrency)
        normalized = [
            result if isinstance(result, dict) else self._failure(jobs[index], result, resolver)
            for index, result in enumerate(results)
        ]
        cropped: list[Any] = []
        for result in normalized:
            cropped.extend(result.get("cropped") or [])
        if not cropped:
            cropped = [one_pixel_image()]
        return (
            [result["images"] for result in normalized],
            [result["texts"] for result in normalized],
            [result["request_log"] for result in normalized],
            [result["response_log"] for result in normalized],
            cropped,
        )

    def _run_frame(self, job: dict[str, Any], resolver: _ProviderResolver) -> dict[str, Any]:
        try:
            image_value = _first(job.get("image"))
            mask_value = _first(job.get("mask"))
            if image_value is None:
                raise EndpointError(
                    "No image connected.", error_type="validation",
                    error_code="image_missing", phase="validate",
                )
            if mask_value is None:
                raise EndpointError(
                    "No mask connected.", error_type="validation",
                    error_code="mask_missing", phase="validate",
                )
            frames = detailer.frames_of(image_value)
            masks = detailer.masks_of(mask_value)
        except Exception as error:  # noqa: BLE001 - per-item failure is data
            return self._failure(job, error, resolver)

        if not frames:
            return self._failure(
                job,
                EndpointError(
                    "The image input is empty.", error_type="validation",
                    error_code="image_missing", phase="validate",
                ),
                resolver,
            )
        if len(masks) == 1 and len(frames) > 1:
            masks = masks * len(frames)
        if len(masks) != len(frames):
            return self._failure(
                job,
                EndpointError(
                    f"Mask frames ({len(masks)}) must match image frames ({len(frames)}), "
                    "or be a single frame that broadcasts.",
                    error_type="validation",
                    error_code="mask_frames_mismatch",
                    phase="validate",
                ),
                resolver,
            )

        crop_factor = float(_first(job.get("crop_factor")) or 3.0)
        target = int(_first(job.get("target_resolution")) or 0)
        feather = int(_first(job.get("feather")) or 0)
        drop_size = int(_first(job.get("drop_size")) or 10)
        mask_grow = int(_first(job.get("mask_grow")) or 0)
        guide = str(_first(job.get("mask_guide")) or detailer.GUIDE_NONE)
        paste_area = str(_first(job.get("paste_area")) or detailer.PASTE_CROP)
        prompt = str(_first(job.get("prompt")) or "")
        provider_value = job.get("provider")

        guide_note = ""
        if paste_area == detailer.PASTE_CROP and guide != detailer.GUIDE_NONE:
            # A marker inside the crop would be pasted back at full strength.
            guide_note = (
                f"mask_guide={guide} 已忽略：paste_area=crop box 会把整块裁剪贴回，"
                "标记会被一起贴进画面。"
            )
            guide = detailer.GUIDE_NONE

        composed_frames = []
        frame_reports = []
        cropped_frames: list[Any] = []
        request_logs: list[str] = []
        response_logs: list[str] = []
        seen_regions = 0
        painted_regions = 0
        last_error = ""

        for frame_index, (frame, mask) in enumerate(zip(frames, masks)):
            composed = np.array(frame, dtype=np.float32, copy=True)
            try:
                regions = detailer.regions_from_mask(
                    frame,
                    mask,
                    crop_factor=crop_factor,
                    drop_size=drop_size,
                    mask_grow=mask_grow,
                )
            except Exception as error:  # noqa: BLE001
                return self._failure(job, error, resolver)

            frame_report: dict[str, Any] = {
                "frame": frame_index,
                "regions": len(regions),
                "paste_area": paste_area,
                "items": [],
            }
            if guide_note:
                frame_report["note"] = guide_note
            seen_regions += len(regions)

            for region_index, region in enumerate(regions):
                region_job = {
                    "provider": provider_value,
                    "prompt": prompt,
                    "_prompt": prompt,
                    "_guide": guide,
                    "_target": target,
                    "_feather": feather,
                    "_region": region,
                    "_index": len(request_logs),
                }
                result = self._run_one(region_job, resolver)
                request_logs.append(str(result.get("request_log") or ""))
                response_logs.append(str(result.get("response_log") or ""))

                item = detailer.region_report(region, region_index)
                tensor = result.get("images")
                if tensor is None or _placeholder(tensor):
                    item["ok"] = False
                    item["error"] = str(result.get("texts") or "region failed")[:600]
                    last_error = item["error"]
                else:
                    edited = detailer.frames_of(tensor)[0]
                    cropped_frames.append(_frame_tensor(edited))
                    detailer.composite(composed, region, edited, feather, paste_area)
                    item["ok"] = True
                    item["edited_size"] = [int(edited.shape[0]), int(edited.shape[1])]
                    painted_regions += 1
                frame_report["items"].append(item)

            composed_frames.append(composed)
            frame_reports.append(frame_report)

        if seen_regions > 0 and painted_regions == 0:
            # Every region failed: keep the package convention (1x1 + error text)
            # but leave the per-region logs in place for diagnosis.
            response_logs.append(
                json.dumps({"ok": False, "error": last_error or "every region failed"}, ensure_ascii=False)
            )
            return {
                "images": one_pixel_image(),
                "texts": last_error or "Every region failed.",
                "request_log": self._merge_logs(request_logs),
                "response_log": self._merge_logs(response_logs),
                "cropped": cropped_frames,
            }

        report = {
            "ok": True,
            "frames": len(frames),
            "regions": seen_regions,
            "painted": painted_regions,
            "paste_area": paste_area,
            "detail": frame_reports,
        }
        if guide_note:
            report["note"] = guide_note
        return {
            "images": _stack_frames(composed_frames),
            "texts": json.dumps(report, ensure_ascii=False),
            "request_log": self._merge_logs(request_logs),
            "response_log": self._merge_logs(response_logs),
            "cropped": cropped_frames,
        }

    @staticmethod
    def _merge_logs(entries: list[str]) -> str:
        """One JSON array per frame instead of one JSON object per region."""
        merged: list[Any] = []
        for entry in entries:
            text = str(entry or "").strip()
            if not text:
                continue
            try:
                merged.append(json.loads(text))
            except Exception:  # noqa: BLE001 - non-JSON log stays readable
                merged.append({"log": text})
        return json.dumps(merged, ensure_ascii=False)


def _prepare_edit_request(
    provider: ProviderConfig,
    prompt: str,
    image_png: bytes,
    mask_png: bytes | None,
    params: dict[str, Any],
    image_summary: dict[str, Any],
):
    """Dispatch one masked edit to the family that owns the provider's protocol."""
    protocol = provider.api_protocol
    if protocol == PROTOCOL_GEMINI:
        return api_gemini.prepare_image_request(
            provider,
            operation="edit",
            prompt=prompt,
            images=[image_png],
            mask=mask_png,
            params=params,
            image_summary=image_summary,
        )
    if protocol == PROTOCOL_OPENAI_IMAGES:
        return api_openai.prepare_image_request(
            provider,
            operation="edit",
            prompt=prompt,
            images=[image_png],
            mask=mask_png,
            params=params,
            image_summary=image_summary,
        )
    if protocol == PROTOCOL_GROK:
        return api_grok.prepare_image_request(
            provider,
            operation="edit",
            prompt=prompt,
            images=[image_png],
            mask=mask_png,
            params=params,
            image_summary=image_summary,
        )
    return api_qwen.prepare_image_request(
        provider,
        operation="edit",
        prompt=prompt,
        images=[image_png],
        mask=mask_png,
        params=params,
        image_summary=image_summary,
    )


# ---------------------------------------------------------------------------
# Text (LLM) nodes
# ---------------------------------------------------------------------------
class _ChatCallNode(_BaseCallNode):
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RESULT_KEYS = ("texts", "request_log", "response_log")
    RETURN_NAMES = RESULT_KEYS
    OUTPUT_IS_LIST = (True, True, True)

    @classmethod
    def _text_inputs(cls, family_widgets: dict[str, Any]):
        optional = {
            name: ("IMAGE", {"tooltip": "作为上下文的图像；连上后会自动出现下一个槽位。"})
            for name in REFERENCE_SLOTS[:4]
        }
        optional["files"] = (FILES_TYPE, {"tooltip": "来自文件加载节点。"})
        return optional


class CustomEndpointGPTChat(_ChatCallNode):
    LABEL = "自定义端点 · GPT 对话"
    ALLOWED_PROTOCOLS = (PROTOCOL_OPENAI_CHAT, PROTOCOL_OPENAI_RESPONSES)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "provider": (PROVIDER_TYPE,),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "instructions": ("STRING", {"default": "", "multiline": True, "tooltip": "系统/开发者指令。"}),
            "truncation": W.combo(("auto", "disabled"), "auto"),
            "max_output_tokens": W.optional_int(-1, minimum=-1, maximum=131072),
            "reasoning_effort": W.combo(("none", "minimal", "low", "medium", "high", "xhigh", "max")),
            "temperature": W.optional_float(-1.0, minimum=-1.0, maximum=2.0, tooltip="/responses 协议会忽略此参数。"),
            "top_p": W.optional_float(-1.0, minimum=-1.0, maximum=1.0, tooltip="/responses 协议会忽略此参数。"),
            "store": ("BOOLEAN", {"default": False, "tooltip": "是否让服务端保存该响应（默认否）。"}),
        }
        required.update(cls._common_widgets())
        return {"required": required, "optional": cls._text_inputs(None)}

    def _build_request(self, provider, job):
        images = _collect_images(job, REFERENCE_SLOTS[:4])
        params = _clean_params(
            job,
            ("instructions", "truncation", "max_output_tokens", "reasoning_effort", "temperature",
             "top_p", "store"),
        )
        params["store"] = bool(_first(job.get("store")))
        files = _files_parts(job, "openai")
        return api_openai.prepare_chat_request(
            provider,
            prompt=str(_first(job.get("prompt")) or ""),
            images=images,
            files=files,
            params=params,
            image_summary={"images": [len(png) for png in images]} if images else {},
        )

    def _parse(self, provider, response, client):
        return api_openai.parse_chat_response(provider, response)


class CustomEndpointGeminiChat(_ChatCallNode):
    LABEL = "自定义端点 · Gemini 对话"
    ALLOWED_PROTOCOLS = (PROTOCOL_GEMINI,)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "provider": (PROVIDER_TYPE,),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "system_prompt": ("STRING", {"default": "", "multiline": True}),
            "thinking_level": W.combo(("MINIMAL", "LOW", "MEDIUM", "HIGH"), "MEDIUM"),
            "temperature": W.optional_float(-1.0, minimum=-1.0, maximum=2.0),
            "top_p": W.optional_float(-1.0, minimum=-1.0, maximum=1.0),
            "top_k": W.optional_int(-1, minimum=-1, maximum=1000),
            "max_output_tokens": W.optional_int(-1, minimum=-1, maximum=65536),
            "stop_sequences": ("STRING", {"default": "", "multiline": False, "tooltip": "逗号分隔，留空不发送。"}),
            "seed": W.optional_int(-1, minimum=-1, maximum=2147483647),
        }
        required.update(cls._common_widgets())
        return {"required": required, "optional": cls._text_inputs(None)}

    def _build_request(self, provider, job):
        images = _collect_images(job, REFERENCE_SLOTS[:4])
        params = _clean_params(
            job,
            ("system_prompt", "thinking_level", "temperature", "top_p", "top_k",
             "max_output_tokens", "seed"),
        )
        stops = W.split_csv(_first(job.get("stop_sequences")))
        if stops:
            params["stop_sequences"] = stops
        files = _files_parts(job, "gemini")
        return api_gemini.prepare_chat_request(
            provider,
            prompt=str(_first(job.get("prompt")) or ""),
            images=images,
            files=files,
            params=params,
            image_summary={"images": [len(png) for png in images]} if images else {},
        )

    def _parse(self, provider, response, client):
        return api_gemini.parse_response(
            provider, response, client, timeout=provider.timeout_sec, want_images=False
        )


class CustomEndpointGrokChat(_ChatCallNode):
    LABEL = "自定义端点 · Grok 对话"
    ALLOWED_PROTOCOLS = (PROTOCOL_GROK,)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "provider": (PROVIDER_TYPE,),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "system_prompt": ("STRING", {"default": "", "multiline": True}),
            "temperature": W.optional_float(-1.0, minimum=-1.0, maximum=2.0),
            "top_p": W.optional_float(-1.0, minimum=-1.0, maximum=1.0),
            "max_tokens": W.optional_int(-1, minimum=-1, maximum=131072),
            "reasoning_effort": W.combo(("low", "medium", "high")),
        }
        required.update(cls._common_widgets())
        return {"required": required, "optional": cls._text_inputs(None)}

    def _build_request(self, provider, job):
        images = _collect_images(job, REFERENCE_SLOTS[:4])
        params = _clean_params(
            job, ("system_prompt", "temperature", "top_p", "max_tokens", "reasoning_effort")
        )
        files = _files_parts(job, "grok")
        return api_grok.prepare_chat_request(
            provider,
            prompt=str(_first(job.get("prompt")) or ""),
            images=images,
            files=files,
            params=params,
            image_summary={"images": [len(png) for png in images]} if images else {},
        )

    def _parse(self, provider, response, client):
        return api_grok.parse_chat_response(provider, response)


# ---------------------------------------------------------------------------
# Auxiliary nodes
# ---------------------------------------------------------------------------
def _input_directory() -> str:
    if folder_paths is not None:
        try:
            return folder_paths.get_input_directory()
        except Exception:  # pragma: no cover
            pass
    fallback = os.path.join(PACKAGE_ROOT, "input")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _list_input_files(extensions: tuple[str, ...], max_bytes: int) -> list[str]:
    directory = _input_directory()
    try:
        entries = [entry for entry in os.scandir(directory) if entry.is_file()]
    except OSError:
        return []
    names = [
        entry.name
        for entry in entries
        if entry.name.lower().endswith(extensions) and entry.stat().st_size < max_bytes
    ]
    return sorted(names)


def _resolve_input_path(name: str) -> str:
    if folder_paths is not None:
        try:
            return folder_paths.get_annotated_filepath(name)
        except Exception:  # pragma: no cover
            pass
    return os.path.join(_input_directory(), name)


class _FileLoader:
    FAMILY = "openai"
    EXTENSIONS = (".txt", ".pdf")
    MAX_BYTES = 32 * 1024 * 1024
    LABEL = "文件加载"

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_input_files(cls.EXTENSIONS, cls.MAX_BYTES)
        return {
            "required": {
                "file": (
                    tuple(files) if files else ("",),
                    {"default": files[0] if files else "", "tooltip": "input 目录下的文件。"},
                )
            },
            "optional": {
                "files": (FILES_TYPE, {"tooltip": "可选：串联上一个文件加载节点的结果。"}),
            },
        }

    RETURN_TYPES = (FILES_TYPE,)
    RETURN_NAMES = ("files",)
    FUNCTION = "load"
    CATEGORY = CATEGORY_AUX

    def load(self, file, files=None):
        if not str(file or "").strip():
            raise ValueError("请选择一个文件（ComfyUI 的 input 目录）。")
        path = _resolve_input_path(file)
        if os.path.getsize(path) >= self.MAX_BYTES:
            raise ValueError(f"文件过大：{file}")
        with open(path, "rb") as handle:
            content = handle.read()
        part = self._make_part(os.path.basename(path), content)
        parts: list[dict[str, Any]] = []
        if isinstance(files, dict):
            parts.extend(files.get("parts") or [])
        parts.append(part)
        return ({"family": self.FAMILY, "parts": parts},)

    def _make_part(self, filename: str, content: bytes) -> dict[str, Any]:
        import base64

        mime = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
        return {
            "type": "input_file",
            "filename": filename,
            "file_data": f"data:{mime};base64," + base64.b64encode(content).decode("ascii"),
        }


class CustomEndpointGPTFiles(_FileLoader):
    FAMILY = "openai"
    LABEL = "GPT 文件加载"


class CustomEndpointGeminiFiles(_FileLoader):
    FAMILY = "gemini"
    LABEL = "Gemini 文件加载"

    def _make_part(self, filename: str, content: bytes) -> dict[str, Any]:
        import base64

        mime = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
        return {
            "inlineData": {"mimeType": mime, "data": base64.b64encode(content).decode("ascii")},
            "_filename": filename,
        }


class CustomEndpointBatchImages:
    """Merge several IMAGE inputs into one batch, and hand back the raw list.

    ``images`` is a single ``[N,H,W,3]`` tensor.  A tensor batch has exactly one
    canvas, so frames of other sizes are fitted into it **with their aspect
    ratio intact** (``fit``): the old implementation resized every frame to the
    first frame's width/height, which squashed squeezed 16:9 into 1:1.

    ``image_list`` is the plain python list of the same frames at their native
    sizes - ComfyUI carries those fine (each list item is its own tensor, and a
    downstream node simply runs once per item).  Use it when the exact pixels
    matter; use ``images`` when one call has to carry several references.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE", {"tooltip": "第 1 张。"}),
                "size_mode": W.plain_combo(
                    SIZE_MODES,
                    SIZE_MODE_MAX,
                    tooltip=(
                        "画布尺寸：max size = 各图最大宽×最大高，谁都不会被缩小（默认）；"
                        "first image = 第一张（旧行为）；manual = 用下面的 width/height（0 = 该边取最大）。"
                    ),
                ),
                "fit": W.plain_combo(
                    FIT_MODES,
                    FIT_CONTAIN,
                    tooltip=(
                        "尺寸不一致时怎么放进画布：contain = 等比缩放 + 留边（默认，不畸变、不裁切）；"
                        "cover = 等比放大铺满 + 裁掉溢出；stretch = 直接拉伸（旧行为，会把图拉畸形）。"
                    ),
                ),
                "position": W.plain_combo(
                    POSITIONS, POS_CENTER, tooltip="留边 / 裁切时画面贴在画布的哪一边。"
                ),
                "pad_color": W.plain_combo(
                    PAD_COLOR_NAMES, "black", tooltip="contain 留边的填充色。"
                ),
                "size_multiple": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "把画布宽高向上取整到该倍数（8 / 16 / 64 常用于尺寸对齐）。",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "固定画布宽（0 = 自动）。",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "固定画布高（0 = 自动）。",
                    },
                ),
            },
            "optional": {
                f"image_{index}": ("IMAGE", {"tooltip": "可选，按顺序拼在后面。"})
                for index in range(2, 5)
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "image_list")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "merge"
    CATEGORY = CATEGORY_AUX

    def merge(
        self,
        image_1,
        image_2=None,
        image_3=None,
        image_4=None,
        size_mode=SIZE_MODE_MAX,
        fit=FIT_CONTAIN,
        position=POS_CENTER,
        pad_color="black",
        size_multiple=1,
        width=0,
        height=0,
    ):
        frames: list[Any] = []
        for value in (image_1, image_2, image_3, image_4):
            frames.extend(iter_frames(value))
        if not frames:
            return (one_pixel_image(), [one_pixel_image()])
        merged = stack_images(
            frames,
            size_mode=size_mode,
            width=width,
            height=height,
            fit=fit,
            position=position,
            pad_color=pad_color,
            multiple=size_multiple,
        )
        return (merged, frames)


class CustomEndpointPromptList:
    """Split a multi-line text into a STRING list to drive a concurrent batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "每行一条提示词。"}),
                "skip_empty": ("BOOLEAN", {"default": True}),
                "trim": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split"
    CATEGORY = CATEGORY_AUX

    def split(self, text, skip_empty=True, trim=True):
        lines = str(text or "").splitlines()
        if trim:
            lines = [line.strip() for line in lines]
        if skip_empty:
            lines = [line for line in lines if line]
        return (lines,)


# ---------------------------------------------------------------------------
# Registrations (single source of truth: __init__.py imports these)
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "CustomEndpointProvider": CustomEndpointProvider,
    "CustomEndpointGPTImage": CustomEndpointGPTImage,
    "CustomEndpointGeminiImage": CustomEndpointGeminiImage,
    "CustomEndpointGrokImage": CustomEndpointGrokImage,
    "CustomEndpointQwenImage": CustomEndpointQwenImage,
    "CustomEndpointQwenImageEdit": CustomEndpointQwenImageEdit,
    "CustomEndpointAPIDetailer": CustomEndpointAPIDetailer,
    "CustomEndpointGPTChat": CustomEndpointGPTChat,
    "CustomEndpointGeminiChat": CustomEndpointGeminiChat,
    "CustomEndpointGrokChat": CustomEndpointGrokChat,
    "CustomEndpointGPTFiles": CustomEndpointGPTFiles,
    "CustomEndpointGeminiFiles": CustomEndpointGeminiFiles,
    "CustomEndpointBatchImages": CustomEndpointBatchImages,
    "CustomEndpointPromptList": CustomEndpointPromptList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomEndpointProvider": "自定义端点 · API 提供商",
    "CustomEndpointGPTImage": "自定义端点 · GPT 图像",
    "CustomEndpointGeminiImage": "自定义端点 · Gemini 图像",
    "CustomEndpointGrokImage": "自定义端点 · Grok 图像",
    "CustomEndpointQwenImage": "自定义端点 · Qwen 图像",
    "CustomEndpointQwenImageEdit": "自定义端点 · Qwen 图像编辑",
    "CustomEndpointAPIDetailer": "自定义端点 · API 局部重绘",
    "CustomEndpointGPTChat": "自定义端点 · GPT 对话",
    "CustomEndpointGeminiChat": "自定义端点 · Gemini 对话",
    "CustomEndpointGrokChat": "自定义端点 · Grok 对话",
    "CustomEndpointGPTFiles": "自定义端点 · GPT 文件加载",
    "CustomEndpointGeminiFiles": "自定义端点 · Gemini 文件加载",
    "CustomEndpointBatchImages": "自定义端点 · 图像批量拼接",
    "CustomEndpointPromptList": "自定义端点 · 提示词列表",
}
