# 维护手册 · ComfyUI-Custom-Endpoint-Nodes

> **这份文档给谁看**：接手这个包的维护者（人或 AI 助手）。目标是不通读全部代码也能
> ① 掌握节点包的结构与不变式，② 把 ComfyUI 任意一个**原生合作伙伴节点**快速、正确地
> 搬进本包（Qwen 节点就是按第 7 章的配方加进来的，可作范本）。
>
> 阅读顺序建议：第 1、2 章（事实与分层）→ 第 4、5、6 章（契约）→ 第 7 章（配方）→
> 第 8 章（提交前清单）。第 11、12 章是 Qwen 的现场记录与排障表，迁移别的协议时对照。

---

## 1. 快速事实

| 项 | 值 |
|---|---|
| 包根（ComfyUI 里就是仓库根） | `<ComfyUI>/custom_nodes/ComfyUI-Custom-Endpoint-Nodes` |
| ComfyUI 根 | `<ComfyUI>` |
| 测试用 python（带 torch/PIL/numpy） | ComfyUI 环境的 python（aki 整合包在 `<ComfyUI>/../python/python.exe`，官方版一般就是 `python`） |
| 跑测试 | `cd <包根>; <python> -m pytest tests/ -q` |
| 依赖 | **零第三方依赖**：只用标准库 + ComfyUI 自带的 torch/numpy/Pillow |
| 节点数 | 14（1 提供商 + 9 调用 + 2 文件加载 + 2 工具） |
| 测试数 | 230 项，全部离线（假 opener 重放响应） |
| 版本 | `pyproject.toml` 的 `version`（Registry 每次发布必须升版） |
| 前端扩展 | `custom_endpoint/web/*.js`（3 个文件，`WEB_DIRECTORY` 指向该目录） |

三条最容易踩的硬约束：

1. **顶层节点模块必须叫 `ce_nodes.py`，不能叫 `nodes.py`** —— ComfyUI 自己有
   `ComfyUI/nodes.py`，而 pytest 会把 ComfyUI 根塞进 `sys.path`，`import nodes` 会静默
   拿到 ComfyUI 的那份。改名要同时改 `__init__.py`、`tests/conftest.py` 以及所有
   `import ce_nodes` 的测试文件。
2. 包**必须能被当作普通模块导入**（pytest 走的是 `__init__.py` 的非包分支），
   所以任何新模块都得在 `ce_nodes.py` 的 `if __package__: / else:` 两个分支里都写 import。
3. 改完代码必须跑通全套 pytest；改节点集合/参数面还要同步 `tests/test_registration.py`。

---

## 2. 分层架构与数据流

```
[提供商节点]  endpoint / api_key / model / api_protocol
      │  build() → ProviderConfig（不可变；不暴露密钥，public_dict() 只给诊断信息）
      ▼
[调用节点]  INPUT_IS_LIST=True：整批输入一次到达
      │  _BaseCallNode.execute(): batch.resolve_batch() 做「长度 1 广播 / 长度 N 对齐」
      │  → run_concurrent(worker, jobs, max_concurrency)
      ▼  每项 job 一条线程
  _run_one(job):
      require_protocols(provider, ALLOWED_PROTOCOLS)      # 协议不匹配就报错，不崩队列
      _build_request(provider, job) → PreparedRequest     # api_<family>.prepare_*()
      → 记 request_log（json 里图片变摘要、凭据打码）
      HttpClient().request(...)  →  HttpResponse          # transport：urllib + 重试
      _parse(provider, response, client) → ParsedResponse # api_<family>.parse_*()
      → 记 response_log（状态码/耗时/正文 + 解析结果）
      → dict(images=…, thought_images=…, texts=…, request_log=…, response_log=…)
      ▼
   任何异常 → _failure()：1×1 图片 + 报错串（队列继续跑）
      ▼
  execute() 按 RESULT_KEYS 把每项结果转成「每输出一路一个列表」，配合 OUTPUT_IS_LIST 输出
```

凭据链路（**不要破坏这三层**）：

```
前端 POST /custom-endpoint/session-credential ──► secrets.issue_session_credential() 内存句柄
   （routes.guard_prompt_payload 也会把 prompt 里的明文 endpoint/api_key 就地换成句柄）

queued prompt / history / PNG 的 prompt+workflow ──► 只含 custom-endpoint-session: 句柄
ProviderConfig.credential() ──► 执行时把句柄换回明文（api_key 一次性消费；endpoint 可重用）
导出 .json ──► web/export_scrub.js：「加密导出」开 → 清空端点与密钥
```

| 层 | 文件 | 职责 | 什么时候动 |
|---|---|---|---|
| 节点层 | `ce_nodes.py` | 节点类、`INPUT_TYPES`、`NODE_*_MAPPINGS`、批处理流水线 | 加节点 / 改参数面 |
| 协议层 | `api_openai.py` `api_gemini.py` `api_grok.py` `api_qwen.py` | 一家厂商的请求构造 + 响应解析（纯函数，无网络） | 接新厂商 / 改字段 |
| 路由层 | `provider.py` | 协议枚举、`auto` 推断、模型别名、端点归一化、`auth_headers` | 加协议 / 加模型别名 |
| 传输层 | `transport.py` | urllib 客户端、重试、multipart、`auth_headers` | 极少（加鉴权方案时才改） |
| 表达层 | `api_common.py` | `PreparedRequest` / `ParsedResponse` / `check_response` / `download_bytes` | 加公共能力 |
| 图像层 | `images.py` | tensor↔PNG、掩码→RGBA、1×1 兜底图、`stack_images` + 画布几何（`resolve_canvas` / `fit_to_canvas`） | 加图像处理 / 改批次几何 |
| 日志层 | `logs.py` `redact.py` `errors.py` | 两路日志、脱敏、`EndpointError` | 日志字段变化 |
| 并发层 | `batch.py` | 批次对齐 + 线程池 | 极少 |
| 控件层 | `widgets.py` | 「不发送」哨兵与控件规格 helper | 加控件写法 |
| 凭据层 | `secrets.py` `routes.py` | 句柄、守卫、`session-credential` 路由 | 安全模型变化 |
| 前端 | `custom_endpoint/web/*.js` | 密码框、导出抹除、图像端口按需增长 | 交互变化 |

---

## 3. 文件地图（改什么去哪里）

| 文件 | 关键符号 | 说明 |
|---|---|---|
| `__init__.py` | `NODE_CLASS_MAPPINGS`、`WEB_DIRECTORY`、`register_routes()` | ComfyUI 入口；`if __package__` 双分支 |
| `ce_nodes.py` | `CustomEndpointProvider`、`_BaseCallNode`、`_ImageCallNode`、`_ChatCallNode`、14 个节点类、`_collect_images/_clean_params/_first/_files_parts/_mask_png`、`_new_client`、`_prepare_edit_request` | 节点层全部 |
| `custom_endpoint/provider.py` | `PROTOCOL_*`、`API_PROTOCOLS`、`PROTOCOL_LABELS`、`FAMILY_*`、`_PROTOCOL_FAMILY`、`*_MODEL_IDS`、`normalize_model_id`、`infer_protocol`、`normalize_base_url`、`join_url`、`ProviderConfig`、`build_provider`、`require_protocols` | 协议路由 |
| `custom_endpoint/api_common.py` | `PreparedRequest`、`ParsedResponse`、`check_response`、`extract_error_message`、`download_bytes`、`decode_b64_image` | 协议层公共面 |
| `custom_endpoint/api_openai.py` | `prepare_image_request` / `parse_image_response` / `prepare_chat_request` / `parse_chat_response`、`resolve_size` | GPT 图像 + GPT 对话（chat + responses） |
| `custom_endpoint/api_gemini.py` | `generate_endpoint`、`prepare_image_request`、`parse_response(..., want_images=)` | Gemini（图像与文本共用 generateContent） |
| `custom_endpoint/api_grok.py` | `prepare_image_request`、`parse_image_response`、`prepare_chat_request` | Grok（OpenAI 兼容 images + chat） |
| `custom_endpoint/api_qwen.py` | `generation_endpoint`、`resolve_size`、`resolve_image_refs`、`prepare_image_request`、`parse_image_response`、`_upload_payload` | Qwen 图像 / 图像编辑 |
| `custom_endpoint/images.py` | `MAX_INPUT_IMAGES=9`、`iter_frames`、`image_to_pngs`、`mask_to_rgba_png`、`png_bytes_to_tensor`、`one_pixel_image`、`describe_bytes`、批次几何 `SIZE_MODES`/`FIT_MODES`/`POSITIONS`/`PAD_COLORS`、`resolve_canvas`、`fit_to_canvas`、`round_up`、`stack_images` | tensor↔字节、统一画布 |
| `custom_endpoint/detailer.py` | `DetailRegion`、`make_crop_region`、`connected_components`、`dilate_binary`、`regions_from_mask`、`resize_region`、`paint_guide`、`guide_pad`、`encode_region`、`feathered_alpha`、`rect_edge_alpha`、`composite`、`aspect_ratio_for`、`GUIDES`、`GUIDE_HINTS`、`PASTE_AREAS` | 局部重绘几何（纯 numpy+PIL，无 scipy） |
| `custom_endpoint/logs.py` | `request_log_json`、`response_log_json`、`_B64_FIELD_NAMES`、`summarize_binary` | 两路日志 |
| `custom_endpoint/errors.py` | `EndpointError`（message/status/error_type/error_code/phase/…）、`error_json`、`report_json` | 结构化错误 |
| `custom_endpoint/redact.py` | `redact_headers`、`redact_url`、`redact_text`、`sanitize` | 脱敏 |
| `custom_endpoint/batch.py` | `resolve_batch`、`broadcast`、`run_concurrent`、`MAX_PARALLEL_WORKERS=16` | 并发批次 |
| `custom_endpoint/widgets.py` | `OMIT`（=「不发送」）、`combo`、`plain_combo`、`text`、`boolean`、`optional_int`、`optional_float`、`clean`、`clean_text`、`split_csv` | 控件写法 |
| `custom_endpoint/secrets.py` | `HANDLE_PREFIX="custom-endpoint-session:"`、TTL 30 min、`issue/resolve/consume/clear_session_credential` | 内存句柄 |
| `custom_endpoint/routes.py` | `ROUTE_PATH`、`guard_prompt_payload`、`scrub_provider_nodes`、`register_routes` | 守卫与路由 |
| `custom_endpoint/web/provider_widget.js` | `NODE_TYPE="CustomEndpointProvider"`、`WIDGET_LABELS` | 密钥密码框 + 中文标签 |
| `custom_endpoint/web/export_scrub.js` | 序列化钩子 | 「加密导出」抹除 |
| `custom_endpoint/web/grow_image_inputs.js` | `declaredMax(nodeType, nodeData)` | 图像端口按需增长（读 `nodeData.input.optional` 里的 `image_N`） |
| `example_workflows/*.json` | 3 份真实导出的工作流 | 手工维护（不再有生成器）；`tests/test_example_workflows.py` 只校验本包节点、widgets 与连线，第三方节点允许存在 |
| `tests/*` | 见第 9 章 | 离线测试 |

---

## 4. 节点注册契约（`ce_nodes.py`）

### 4.1 现有节点

| 节点 id | 显示名 | 协议 | 输出 |
|---|---|---|---|
| `CustomEndpointProvider` | 自定义端点 · API 提供商 | — | `CUSTOM_ENDPOINT_PROVIDER` |
| `CustomEndpointGPTImage` | 自定义端点 · GPT 图像 | `openai_images` | images/texts/request_log/response_log |
| `CustomEndpointGeminiImage` | 自定义端点 · Gemini 图像 | `gemini` | + `thought_images`（5 路） |
| `CustomEndpointGrokImage` | 自定义端点 · Grok 图像 | `grok` | 4 路 |
| `CustomEndpointQwenImage` | 自定义端点 · Qwen 图像 | `qwen` | 4 路 |
| `CustomEndpointQwenImageEdit` | 自定义端点 · Qwen 图像编辑 | `qwen` | 4 路 |
| `CustomEndpointAPIDetailer` | 自定义端点 · API 局部重绘 | `gemini`/`openai_images`/`grok`/`qwen`/`qwen_images` | 5 路（每帧一条日志数组 + `cropped`） |
| `CustomEndpointGPTChat` | 自定义端点 · GPT 对话 | `openai_chat` / `openai_responses` | texts/request_log/response_log |
| `CustomEndpointGeminiChat` | 自定义端点 · Gemini 对话 | `gemini` | 3 路 |
| `CustomEndpointGrokChat` | 自定义端点 · Grok 对话 | `grok` | 3 路 |
| `CustomEndpointGPTFiles` / `CustomEndpointGeminiFiles` | 文件加载 | — | `CUSTOM_ENDPOINT_FILES`（`{"family":…, "parts":[…]}`） |
| `CustomEndpointBatchImages` | 图像批量拼接 | — | `IMAGE` ×2：`images`（一个张量批次）/ `image_list`（list，原尺寸） |
| `CustomEndpointPromptList` | 提示词列表 | — | `STRING`（`OUTPUT_IS_LIST`） |

注册是**单一事实源**：`ce_nodes.py` 末尾的 `NODE_CLASS_MAPPINGS` /
`NODE_DISPLAY_NAME_MAPPINGS`，`__init__.py` 只做 re-export。两个 mapping 的键必须完全一致
（`tests/test_registration.py::test_class_mappings_are_exact` 断言集合相等），显示名必须含
「自定义端点」（`test_display_names_are_non_empty_and_chinese`）。

### 4.2 调用节点的子类契约

`_BaseCallNode` 负责 `execute()`、批次、并发、失败兜底、日志。子类只提供：

```python
class CustomEndpointXxx(_ImageCallNode):      # 或 _ChatCallNode / _BaseCallNode
    LABEL = "自定义端点 · Xxx"                  # 出现在 protocol_mismatch 报错里
    ALLOWED_PROTOCOLS = (PROTOCOL_XXX,)       # 必须是元组
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")   # 与 RESULT_KEYS 一一对应

    @classmethod
    def INPUT_TYPES(cls):
        required = {…}                         # 组合框必须是 tuple
        required.update(cls._common_widgets())  # max_concurrency
        return {"required": required, "optional": cls._image_inputs(3)}

    def _build_request(self, provider, job):   # job 里每个值都是「单项」（已按批对齐）
        …→ api_xxx.prepare_...(provider, …)    # 返回 PreparedRequest

    def _parse(self, provider, response, client):
        …→ api_xxx.parse_...(…, timeout=provider.timeout_sec)   # 返回 ParsedResponse

    # 可选：同一模型有多条线路时（见 6.1 节），把"路由不存在"转成另一次尝试
    def _request_alternative(self, provider, job, request, response):
        if not api_xxx.is_missing_route(response):
            return None
        return self._build_request(provider, job, api_xxx.DIALECT_OTHER)
```

子类若重写 `_build_request` 以支持多线路，签名写成
`def _build_request(self, provider, job, dialect=None)`——基类只按 `(provider, job)` 调用它，
默认 `dialect=None` 由协议决定形状。

固定不变的部分（**不要复制粘贴重写**）：

- `RESULT_KEYS` 决定输出语义：`("images","texts","request_log","response_log")` 是默认；
  文本节点用 `("texts","request_log","response_log")`；Gemini 图像多一路 `thought_images`。
  改 `RESULT_KEYS` 必须同步 `RETURN_TYPES` / `RETURN_NAMES` / `OUTPUT_IS_LIST`（后者长度
  必须等于 `RETURN_TYPES`，测试会查）。
- `_run_one` 永远返回**含全部键的 dict**（失败时 `images/thought_images` 是 1×1 图、
  `texts` 是报错串），所以任何 `RESULT_KEYS` 都能取到值。
- `_failure()` 会尽量把「已构造的请求」也写进 request_log，方便定位是哪一步失败。

### 4.3 `job` 里的值是什么

`INPUT_IS_LIST=True` 的真实语义（已核对 `execution.py::get_input_data`）：

- 节点函数**只被调用一次**；
- widget 值以 **1 元素列表**到达（`[1024]`、`["不发送"]`）；
- link（连线）值保持上游返回值原样 —— 可能 `[tensor, tensor]`（列表），也可能是标量；
- 缺失的可选输入（没连线）**根本不出现在 kwargs 里**，于是 `job.get(name)` 得到 `None`；
- 输出侧 `OUTPUT_IS_LIST` 会把返回的列表 `extend` 出去。

因此节点内部一律走这三个 helper：

| helper | 作用 |
|---|---|
| `_first(value)` | 取单项（列表取第一个），单值直通 |
| `_clean_params(job, names)` | 抽 widget 值并丢掉「不发送」哨兵（`W.clean`），返回 `{name: value}` |
| `_collect_images(job, names=REFERENCE_SLOTS)` | 按端口顺序收集参考图 PNG 字节（批量输入按帧展开，跳号报 `image_gap`，>16 张报 `too_many_images`） |

批次对齐失败（例如 prompt 3 条、图片 2 张）会在 `resolve_batch` 抛 `ValueError`，
**这是在派发之前抛的**，所以它是"整条 prompt 失败"而不是"某一项失败"——这是已知的、
有意的取舍（配对比配错好）。

### 4.4 批次有两种容器：张量 vs list（改几何前必读）

ComfyUI 里"一批图"有两种完全不同的容器，语义不同、能力也不同：

| 容器 | 形态 | 能否混尺寸 | 下游怎么消费 | 谁在用 |
|---|---|---|---|---|
| 张量批次 | 一个 `[N,H,W,3]` 张量 | **不能**：`H/W` 是共享维度，torch 没有参差维度 | `IMAGE` 端口收到整个张量，节点自己按帧循环 | 采样器/VAE、`ImageBatch`、本包调用节点的参考图槽 |
| list | python `list`，每项各自一个 `[1,H,W,3]` | **能**：每项独立张量，尺寸互不影响 | 非 list 节点**按项被调用 N 次**；`INPUT_IS_LIST` 节点收到整表 | `OUTPUT_IS_LIST` 输出（本包 9 个调用节点、Impact `Make Image List`） |

机制在 `execution.py`：`merge_result_data` 对 `OUTPUT_IS_LIST=True` 的输出做
`extend`，否则收成 `[每次调用的结果]`；`_async_map_node_over_list` 对非 list 节点
按 `slice_dict` 一次一项地调用。**所以"不同尺寸的图走一条链路"在这里是合法的**，
只有"塞进同一个张量"才不合法。

推论（本包的两个硬约束）：

1. **想保原尺寸就别压张量**：`CustomEndpointBatchImages.image_list` 就是直通的
   list；把 list 接到 `SaveImage`、采样器这类非 list 节点，会逐张按原生尺寸处理。
2. **"一次请求带多张参考图"只能用张量批次**：调用节点的 `resolve_batch` 把任何
   list 当**批次维度**（长度 1 广播、长度 N 对齐），所以 list 接进去 = N 个独立
   job、每 job 一张图，而不是一 job 多张参考图。要一 job 多图就必须先统一画布
   （`images` 输出），而统一画布时**不许拉伸**（见 `fit`：contain 留边 / cover 裁剪）。

画布几何全在 `images.py`：`resolve_canvas()`（`max size` / `first image` /
`manual` + `width`/`height` + `multiple` 向上取整）与 `fit_to_canvas()`
（`contain` / `cover` / `stretch`）。`stack_images()` 默认 `max size` + `contain`，
即"谁都不缩小、只补边"，`stretch` 只作为显式选项保留（历史行为，会畸变）。

---

## 5. 「不发送」哨兵（`widgets.py`）

提供商节点独占 model，调用节点在 `INPUT_TYPES()` 时不知道模型 ⇒ 只能展示**该家族参数的超集**，
无关字段必须能被显式省略，否则严格中转会因未知字段 400。

| 类型 | 省略值 | 构造 helper |
|---|---|---|
| combo | 首项 `不发送`（`OMIT`） | `W.combo(("low","high"))` |
| int | `-1`（`min` 自动放宽到 -1） | `W.optional_int(-1, minimum=-1, maximum=…)` |
| float | `-1.0` | `W.optional_float(-1.0, …)` |
| string | 空串 | 普通 `("STRING", {"default": ""})` + `W.clean_text` |
| combo（必须有值） | 无哨兵 | `W.plain_combo(("a","b"), "a")` |

- `W.clean(value)` → 命中哨兵返回 `None`（→ 不上线）；`W.is_omitted` 对 `bool` 恒 False，
  所以 `False` 是**有效值**、不会被吞掉。
- 陷阱：把 helper 返回值再包一层圆括号 `(W.combo(...),)` 会得到 1 元组 →
  `IndexError: tuple index out of range`，且只在 `INPUT_TYPES()` 被调用时暴露。
  这种跨行包裹要用**括号配平**排查，正则匹配不到。
- 测试会断言「`不发送` 不能出现在 request_log 里」（等于断言它真没被发送）。

---

## 6. 协议层（`provider.py`）——新增一个协议要改的 5 处

1. 常量：`PROTOCOL_<FAMILY> = "<name>"`，并加进 `API_PROTOCOLS`（顺序 = 节点下拉顺序）。
2. `PROTOCOL_LABELS[PROTOCOL_<FAMILY>] = "<name> (<路径梗概>)"`（前端只用于展示）。
3. `FAMILY_<FAMILY> = "<name>"` + `_PROTOCOL_FAMILY[PROTOCOL_<FAMILY>] = FAMILY_<FAMILY>`
   （`ProviderConfig.family` 由它推导，文件加载器用它校验 family）。
4. `infer_protocol()` 加推断分支（`auto` 用）。**注意顺序**：`gpt-image`/`dall-e` 先判，
   然后 `grok`，然后 `gemini`，然后 `qwen-image`，最后回落 `openai_chat`。`auto` 永远不会
   路由到"模型专属协议"以外的东西，所以模型节点必须 `ALLOWED_PROTOCOLS` 只写自己的协议。
5. `<FAMILY>_MODEL_IDS` + `normalize_model_id()` 分支：把原生节点的友好标签
   （"Qwen Image 3 Pro"）映射成真实 API id（`qwen-image-3.0-pro`），大小写/空格不敏感、
   幂等（已是 id 就原样返回）。

端点归一化（`normalize_base_url`）：**只在用户 URL 完全没有版本段时**补版本
（gemini→`/v1beta`，其余→`/v1`）。这是"静默前缀陷阱"的补丁：中转挂在 `/v1beta` 下时自动补
`/v1` 会让每个请求 404。若协议可能被粘贴整条 URL，则在 `api_*.py` 里做"已含路径就原样用"的
短路（见 `api_gemini.generate_endpoint`、`api_qwen.generation_endpoint`）。

鉴权（`transport.auth_headers`）：`gemini` 同时发 `x-goog-api-key` 与 `Authorization: Bearer`；
其余只发 Bearer；最后叠加 `provider.extra_headers`。**下载结果图时不要带鉴权头**
（`api_common.download_bytes` 传 `headers={}`），签名 URL 也不需要。

### 6.1 同一模型的多条线路（dialect）

一个原生模型可能有两种可用的接口形状，只能靠**端点主机**或**用户选择**区分。Qwen 就是
现成的例子（2026-11 实测）：

| 协议 | 形状 | 端点 | 谁用 |
|---|---|---|---|
| `qwen` | DashScope 原生 | `POST /v1/services/aigc/multimodal-generation/generation` | 官方 `*.aliyuncs.com` |
| `qwen_images` | OpenAI 兼容 | `POST /v1/images/edits`、`/v1/images/generations`，图片放顶层 `image` | 中转（one-api / new-api / LiteLLM 系） |

实现要点（照抄 `api_qwen.py`）：

1. `api_<family>.py` 内用 `DIALECT_*` 常量 + `dialect_for(provider, dialect=None)` 选形状；
   `prepare_image_request(..., dialect=None)` 多一个可选入参，**默认由协议决定**。
2. 各线路**分别校验自己的限制**（Qwen：原生允许 2560²，中转上游实测只到 2048²，超过时上游用
   `HTTP 500 invalid_request` 拒掉）。校验失败要在消息里写清"哪条线路、上限多少"。
3. `auto` 推断可以看主机：`infer_protocol(model, base_url)`（`provider.py` 的
   `_DASHSCOPE_HOST_MARKERS`），无主机信息时保持旧默认（原生）。
4. **线路不匹配要能自愈**：`api_<family>.is_missing_route(response)` 只认
   `404 + "invalid url"` 这类"网关没有这条路由"的签名，其它 404（模型不存在等）不动，
   否则会把真正的错误掩盖掉。
5. 节点通过 `_request_alternative(provider, job, request, response)` 返回替代请求；
   `_BaseCallNode._run_one` 会重发一次，把 `{"fallback": {...}}` 同时写进 `request_log.extra`
   与 `response_log.extra`（`first_status` / `first_url` / `first_error` / 新 `url`）。
   日志里**必须**能看到这次切换，禁止静默改线路。
6. 响应解析要**同时容忍两种形状**（`api_qwen._urls_and_texts`：`data[]` / `images[]` /
   `output.choices[].message.content[]`；`url` / `b64_json` / 裸字符串）。


---

## 7. 迁移配方：把下一个原生合作伙伴节点搬进本包

### 第 0 步 · 侦察原生实现（一次性读完，别猜）

原生节点在 `ComfyUI/comfy_api_nodes/`：

- `nodes_<name>.py` —— 节点定义（`IO.Schema`：inputs/outputs/tooltip/price badge/hidden）
- `apis/<name>.py` —— pydantic 请求/响应模型（= 字段名的唯一权威）
- `util.py` —— `ApiEndpoint`、`sync_op`、`download_url_to_image_tensor`、`tensor_to_base64_string`、`validate_string` 等

从这三处要提取 **6 件事**：

1. HTTP 方法与路径（例如 `/proxy/qwen/api/v1/services/aigc/multimodal-generation/generation`
   → 去掉 `/proxy/<vendor>/api` 前缀后剩下的才是"我们拼在 base_url 后面"的部分）。
2. 鉴权方式（Bearer？`x-goog-api-key`？自定义头？→ 决定是否要动 `auth_headers`）。
3. 请求体结构（逐字段抄，尤其是嵌套对象：Gemini `systemInstruction`/`imageConfig` 这类
   "扁控件 → 嵌套对象"的转换就是 400 的高发区）。
4. 响应体里取图/取文本的路径，以及"没有图/没有文本"时原生抛什么错。
5. 参数面：哪些是模型族共享的（→ `required` 普通控件），哪些是模型专属的
   （→ `W.combo`/`optional_int` 哨兵），哪些是 `DynamicCombo` 因为不同模型参数不同。
6. 校验规则：面积/宽高比/张数/引用编号/文件大小上限 —— **原样移植**，因为它们是服务端的
   真实限制，不是 UI 装饰。

同时记下**哪些东西要丢**：`IO.Hidden.auth_token_comfy_org` / `api_key_comfy_org`
（凭据改由提供商节点提供）、`PriceBadge`（自定义端点不计价）、`is_api_node=True`、
`/proxy/...` 前缀（那是 ComfyUI 官方代理）。

### 第 1 步 · 决定拆几个节点

一个原生节点 → 一个本包节点，**除非**参数语义在生成/编辑之间真的不同。

判据：把两侧参数面写成并集后，是否能用"哨兵 + 参考图端口"表达且不产生歧义？

- 能 → 一个节点。例：GPT 图像（有参考图即 edit，`mask` 可选）；Grok 图像。
- 不能 → 拆开。例：Qwen 的 `QwenImageTextToImageApi`（size 恒为 `width*height`）与
  `QwenImageEditApi`（size 有 `match input`/`auto`/`custom` 三态）语义不同，
  合起来会让 `size` 一个控件有两种含义，所以本包做成 `QwenImage` + `QwenImageEdit`。

### 第 2 步 · `provider.py` 加协议（第 6 章的 5 处）

新模型 id 一定要进 `<FAMILY>_MODEL_IDS`，否则用户复制原生节点里看到的名字会直接 400。

### 第 3 步 · 写 `custom_endpoint/api_<family>.py`

**签名固定**，照 `api_qwen.py` / `api_grok.py` 抄骨架：

```python
def prepare_image_request(provider, *, operation, prompt, images=None, mask=None,
                          params=None, image_summary=None) -> PreparedRequest
def parse_image_response(provider, response, client, *, timeout) -> ParsedResponse
# 文本族则是：
def prepare_chat_request(provider, *, prompt, images=None, files=None, params=None,
                         image_summary=None) -> PreparedRequest
def parse_chat_response(provider, response) -> ParsedResponse
```

必须遵守：

- **纯函数，不做网络 I/O**（解析阶段允许用传进来的 `client` 下载结果图，
  一律 `api_common.download_bytes(client, url, timeout=…, phase="download", protocol=…)`）。
- 参数校验失败抛 `EndpointError(message, error_type="validation", error_code="snake_case", phase="validate")`；
  HTTP 非 2xx 用 `check_response(...)`；响应结构不对抛 `error_type="protocol"` / `"empty_result"`。
- 省略字段用"不传 key"而不是"传 null"（除非原生就是这么传的）。
- 图片字节→data URL 用 `api_qwen._image_data_uri`（含 2048² 与 10 MB 上限、PNG→JPEG 回落）
  或 `images.image_to_pngs` + `base64`。

### 第 4 步 · `ce_nodes.py` 加节点类（第 4.2 节模板）

必须同时更新三个地方：`if __package__` **两个分支**的 import、两个 registration mapping。

### 第 5 步 · 参数面映射表（原生 → 本包）

| 原生写法 | 本包写法 |
|---|---|
| `IO.String.Input(multiline=True)` | `("STRING", {"default": "", "multiline": True, "tooltip": …})` |
| `IO.Int/Float/Boolean.Input(...)` | 同名类型 + `{"default","min","max","step","tooltip"}` |
| `IO.Combo.Input(options=[...])` | `W.plain_combo(tuple, default)`（必须有值）或 `W.combo(tuple, default)`（可省略） |
| `IO.DynamicCombo.Input`（模型族参数差异） | 参数超集 + 哨兵；语义冲突时拆节点 |
| `IO.Autogrow.Input(names=[image_1..image_N])` | 在 `optional` 里**声明全部** `image_1..image_N`（`_image_inputs(N)`），前端 JS 只做显隐（ComfyUI 的旧式 `input_category` 校验只遍历已声明输入，未声明的会被静默忽略） |
| `IO.Hidden.auth_token_comfy_org` 等 | 删除；凭据由提供商节点提供 |
| `PriceBadge` / `is_api_node` | 删除 |
| `ApiEndpoint(path=...)` | `api_x.<family>_endpoint(provider)`（含"整条 URL 已给"短路） |
| `sync_op(...)` | `transport.HttpClient` + `check_response`（同步、带重试） |
| `download_url_to_image_tensor` | `api_common.download_bytes` → `images.png_bytes_to_tensor` |
| `tensor_to_base64_string` | `images.image_to_pngs` / `api_qwen._image_data_uri` |
| `validate_string(..., min_length=1)` | `EndpointError(..., error_code="empty_prompt")` |

### 第 6 步 · 测试（4 类必测，见第 9 章）

1. 线格式：URL、方法、鉴权头、body 逐字段（含"省略字段真的不在 body 里"）。
2. 校验：每条服务端限制各一条 `EndpointError` 断言（`error_code` 精确匹配）。
3. 解析：正常响应（图片 + 文本 + request_id）、HTTP 错误体、结构正确但无结果。
4. 节点级：真执行一次（假 opener），断言张量形状、日志 `operation`、失败兜底为 1×1。

**别忘了**：把新节点 id 加进 `tests/test_registration.py` 的 `EXPECTED_NODES` 与 `CALL_NODES`，
否则 `test_class_mappings_are_exact` 立刻红。协议名加进 `test_every_protocol_is_offered_by_the_provider`。

### 第 7 步 · 跑测试 + 真机烟测（第 10 章）

### 第 8 步 · 更新文档

`README.md`：节点清单表、已知限制、目录树、测试数；`pyproject.toml`：`description` 与
`version`（Registry 已发布版本不可改，必须升版）。

---

## 7.5 API 局部重绘（`CustomEndpointAPIDetailer`）——与其它调用节点的偏差

它不是"一次输入 = 一次请求"，而是"一帧 = N 个区域 = N 次请求"，所以有几处**有意**偏离通用模板。
改它之前先读完这一节，不要按「统一成一次请求」去重构。

1. **`execute()` 被重写，`_run_one()` 被复用。**
   `execute` 只做三件事：批次对齐（`resolve_batch`）、把每帧的遮罩拆成 region job、并发跑帧。
   每个 region 仍然走基类的 `_run_one(job, resolver)`——协议校验、请求、重试/换线、解析、
   两路日志、失败兜底全部不变；`job` 里多带 `_region` / `_guide` / `_target` / `_feather` /
   `_prompt`，由 `_build_request` 消费。**不要**为 detailer 另写一套请求/解析流程。
2. **日志是 JSON 数组，不是单个对象。** 一帧可能有 3 个区域 ⇒ 3 条 request/response 日志，
   `_merge_logs()` 把它们合成一个数组字符串。`test_registration` 只要求节点有
   `request_log`/`response_log` 两路输出，不要求是对象。
3. **失败语义按帧裁剪。** 部分区域失败 → 成功的照常贴回，失败项写进 `texts` 的报告；
   整帧所有区域都失败 → 才退回「1×1 + 报错串」（与其它调用节点一致）；遮罩为空 → 原样直通
   且**不发任何请求**（避免"忘了连遮罩"时白花钱，但也意味着此时提供商填错不会报错）。
4. **`paste_area` 决定羽化落在哪**（2026-11 新增，默认 `crop box`）：
   - `crop box`：整块裁剪贴回，alpha 由 `rect_edge_alpha` 在矩形框四边做 smoothstep 渐出
     （sides 贴图像边界时不渐出，那里没有可混合的对象）；
   - `mask`：`feathered_alpha` 贴遮罩（MaskDetailer 行为）。
   两者都只影响**贴回**，不影响送给模型的内容。默认改成 `crop box` 的原因是：遮罩通常正好
   压住主体轮廓，`mask` 模式会把"模型的柔边"与"原图的硬线"混成一条接缝环。
5. **引导标记必须落在 alpha 带之外**（`guide_pad(feather) = max(3, feather + 3)`）：黄环原本画在
   遮罩外 0–2 px，那里 alpha ≈ 0.32–0.46，会被以三成强度贴回成一圈黄线（2026-11 修复）。
   `paste_area=crop box` 时引导被强制关闭并在报告里写 `note`，否则整块贴回会把标记一起贴进画面。
6. **`cropped` 输出**（`RESULT_KEYS` 第 5 路）：每块模型返回的**原图**，扁平铺开（不是每帧一项）。
   诊断"是模型画歪了还是贴回贴歪了"只看这一路即可；`execute` 因此不能再用
   `[result[key] for result in normalized]` 的通用写法。

区域几何、引导渲染、贴回合成全部在 `custom_endpoint/detailer.py`，纯 numpy+PIL：
连通域用**行程编码 + 并查集**（`connected_components`），膨胀用**积分图**（`dilate_binary`），
因此不需要 scipy（本包零第三方依赖的不变式）。改几何时同步 `tests/test_detailer.py` 的
`test_make_crop_region_*` / `test_connected_components_*` / `test_composite_*` /
`test_rect_edge_alpha_*` / `test_guide_marker_*`。

---

## 8. 提交前的不变式清单

- [ ] `INPUT_TYPES()` / `NODE_CLASS_MAPPINGS` 里**没有**网络 I/O、文件读写、模型加载。
- [ ] 调用节点失败只产出「1×1 图 + 报错串」，不抛异常（`_run_one` 的 try 块覆盖
      `_build_request`、`client.request`、`_parse` 三段）。
- [ ] `OUTPUT_IS_LIST` 长度 == `RETURN_TYPES` 长度，且全为 `True`；`RETURN_NAMES` 与
      `RESULT_KEYS` 一致。**唯一例外**：`CustomEndpointBatchImages` 是
      `(False, True)` —— 它同时给"统一画布的张量批次"和"原尺寸 list"两路（见 4.4）。
- [ ] 合并不同尺寸的图**不许拉伸**：默认 `fit=contain`（留边）；要裁剪用 `cover`，
      `stretch` 只能由用户显式选择。
- [ ] 组合框值一律**元组**；「不发送」不会出现在 request_log / body 里。
- [ ] 新增字段不泄漏凭据；`request_log` 的 headers 仍走 `redact_headers`；结果图下载不带鉴权头。
- [ ] 若加了「换线路重试」：只在路由缺失的签名上触发，且 `request_log.extra.fallback` /
      `response_log.extra.fallback` 里能看到这次切换（禁止静默换线）。
- [ ] 只用标准库 + torch/numpy/Pillow，**不新增依赖**（`pyproject.toml` 的
      `dependencies` 必须保持空，否则 Manager 可能把用户的 CUDA torch 换成 CPU 版）。
- [ ] 顶层模块没有叫 `nodes.py`；新模块在两个 import 分支里都注册了。
- [ ] 全套 `pytest -q` 绿；节点集合/参数面变化时 `test_registration.py` 已同步。
- [ ] `README.md` / `pyproject.toml` 与代码一致。

---

## 9. 测试套地图

| 文件 | 覆盖点 | 加新节点时要动吗 |
|---|---|---|
| `tests/conftest.py` | 假 opener（`QueueOpener`/`HandlerOpener`/`RawResponse`/`json_response`）、`png_bytes`、`image_tensor`、`install_client` fixture（monkeypatch `ce_nodes._new_client`）、`package` fixture（按 ComfyUI 方式加载 `__init__.py`） | 新 API 模块要加进它的 import |
| `test_registration.py` | 节点集合精确相等、显示名含中文、`WEB_DIRECTORY`、提供商控件顺序（`endpoint`/`api_key` 必须是 0/1 —— `routes.py` 按索引抹除）、每个调用节点都是并发批次节点、参数超集、哨兵、协议下拉 | **要**（`EXPECTED_NODES` / `CALL_NODES`） |
| `test_api_requests.py` | 四家的线格式、校验、解析（本文件的 `_provider(family)` 支持 `openai_images/openai_chat/openai_responses/gemini/grok/qwen`） | 要（加一节） |
| `test_nodes_calls.py` | 节点执行：批处理并发、单条失败、网络失败、缺 provider、协议不匹配、句柄解析、Qwen 生成/编辑 | 建议加 |
| `test_provider_and_transport.py` | URL 归一化、协议推断、模型别名、multipart、重试、鉴权头 | 建议加（推断/别名） |
| `test_redact_and_logs.py` | 脱敏、日志去图、错误 JSON | 一般不用 |
| `test_images_and_batch.py` | CHW→HWC 编码、掩码、堆叠缩放、画布几何（contain/cover/stretch、画布来源、`size_multiple`、填充色/位置）、1×1 兜底、批对齐 | 一般不用（改批次几何要加用例） |
| `test_metadata_end_to_end.py` | 真实 ComfyUI 路径：`/prompt` → 守卫 → 历史 → PNG 元数据不含端点/密钥 | 一般不用 |
| `test_frontend_extensions.py` | 用真 node.js 跑 `export_scrub.js`；三个扩展都自带 `registerExtension`；图像槽位最大数来自 `nodeData` | 若加前端扩展要改 |
| `test_example_workflows.py` | 示例工作流与本包节点定义一致（widgets 个数含 seed 控制值、串联输入的 `widget` 标记、连线类型与节点存在性、无凭据/绝对路径、调用节点是否连了 provider）；示例是真实导出的图，**允许第三方节点** | 加示例才要改 |
| `test_routes_guard.py` | 句柄路由、守卫抹除 | 一般不用 |

两个 pytest 专属坑：

1. pytest 会把「带 `__init__.py` 的包根」当 Package 节点导入 → 走的正是 `__init__.py`
   的非包分支，所以那个分支必须能独立跑通。
2. `conftest.load_package()` 会造出**第二棵模块树**（`ComfyUI-Custom-Endpoint-Nodes.xxx`），
   只用于注册元数据；行为测试一律用顶层 `ce_nodes`，否则 `isinstance` 会跨树失败。

---

## 10. 真机烟测配方（离线测试绿了也要跑）

用户主实例通常在 8188，**别去动它**。另起一个 CPU 实例（换端口 + 独立 DB，否则第二个实例
会因 `ComfyUI/user/comfyui.db` 被锁而报 "Database is locked"，并给主实例制造同库锁竞争）：

```powershell
cd "<ComfyUI>"
& "<python>" main.py --port 8199 --cpu --disable-auto-launch `
  --database-url "sqlite:///<临时目录>/ce_smoke.db" `
  --output-directory "<临时目录>/ce_smoke_output" `
  --temp-directory  "<临时目录>/ce_smoke_temp"
# 约 1–2 分钟：轮询 GET http://127.0.0.1:8199/system_stats 直到 200
```

然后：

1. `GET /object_info` —— 必须列出全部 14 个节点（少一个就是 registration 写错）。
2. `POST /prompt`：把提供商 `endpoint` 指向死端口 `http://127.0.0.1:1/v1`，
   链路「提供商 → 目标节点 → SaveImage」。断言：
   - POST 返回 200 且 `node_errors` 为空；
   - `/history/<id>` 的 status 是 success / completed=True（= 失败没有炸队列）；
   - 输出 PNG 尺寸是 `(1,1)`（= 1×1 兜底图）；
   - PNG 的 `prompt` / `workflow` 里**没有**明文 key/endpoint，但有
     `custom-endpoint-session:` 句柄；
   - 把 `response_log` 接 `PreviewAny` 还能在 history 里看到真实报错串。
3. 清理：`Stop-Process` 只杀 `CommandLine` 匹配 `port 8199` 的那个 python
   （**别用通配命令**），并删掉 temp 目录。被强杀的后台任务会以 exit code
   `4294967295`（0xFFFFFFFF）汇报，那不是脚本失败——去读日志确认
   `Prompt executed in N seconds`。

---

## 11. 现场记录：Qwen 图像节点（2026-11 加入）

**来历**：原生 `comfy_api_nodes/nodes_qwen.py`（`QwenImageTextToImageApi` /
`QwenImageEditApi`，模型 `qwen-image-3.0` / `qwen-image-3.0-pro`）+ `apis/qwen.py`。

**协议要点**

- 端点：`POST {base}/services/aigc/multimodal-generation/generation`。原生走
  `/proxy/qwen/api/v1/services/...`，去掉 `/proxy/qwen/api` 后剩下的就是我们要拼的部分，
  所以 `endpoint=https://dashscope.aliyuncs.com/api` → 归一化成 `/api/v1` → 最终
  `/api/v1/services/aigc/multimodal-generation/generation`。中转若挂在
  `/proxy/qwen/api/v1` 下，填到 `/v1` 即可（`generation_endpoint` 会识别已含路径的整条 URL）。
- 鉴权：`Authorization: Bearer <key>`（DashScope 原生）。中转只接受别的头时用提供商节点的
  `extra_headers` 覆盖。
- 请求体：`{"model", "input": {"messages": [{"role": "user", "content": [{"image": dataURL}…, {"text"}]}]},
  "parameters": {"size": "W*H", "n", "seed", "prompt_extend", "watermark", "negative_prompt"}}`。
  `size` 在 edit 的 `auto` 模式下**不发送**；`negative_prompt` 为空时不发送。
- 响应体：`output.choices[].message.content[]`，元素带 `image`（签名 URL）或 `text`；另有
  `request_id` / `code` / `message`。图片 URL **不带鉴权头**下载。
- 零依赖实现：图片用 PIL 压到 ≤2048×2048（等价原生 `tensor_to_base64_string(total_pixels=…)`），
  PNG 超过 10 MB 转 JPEG q92（等价原生的 JPEG 回落）。

**与原生节点的差异（有意为之）**

| 项 | 原生 | 本包 |
|---|---|---|
| 输出 | 仅 `IMAGE` | `images` + `texts` + `request_log` + `response_log`（失败时 1×1 图 + 报错串） |
| 批量 | 一张图一次 | `INPUT_IS_LIST`：提示词列表并发，输出顺序一致 |
| 凭据 | ComfyUI 官方代理 | 提供商节点自带 endpoint/key，三层防泄漏 |
| `dynamic combo` | 模型下拉切参数 | 参数超集 + 哨兵（`W.combo`）；`size` 三态用 `size_mode` |
| 参考图数量 | Autogrow 1–3 | 声明 `image_1..image_3`（前端按需增长，最多 3） |
| 计费徽章 | 有 | 无（自定义端点不计价） |

**保留的原生行为**（不要"优化"掉，否则与官方结果不可比）：

- 像素面积 512×512…2560×2560、宽高比 1:8–8:1 的校验与数值报错；
- `match input` 取自**第一张参考图**并用 `_fit_to_size` 收进允许窗口；
- `@Image1` 风格引用重写成 `Image 1`（词边界判定，`user@image1.com` 不受影响），
  越界引用直接报 `invalid_image_reference`；
- `n ∈ [1,6]`、`seed ∈ [0, 2147483647]`；`prompt_extend` 默认 True、`watermark` 默认 False。

### 11.1 中转线路 `qwen_images` 的实测记录（2026-11，某 one-api / new-api 系中转站）

原生线路接到中转上会得到 `404 {"error":{"message":"Invalid URL (POST /v1/services/aigc/multimodal-generation/generation)","type":"invalid_request_error"}}`
——这是 one-api / new-api 网关"没有这条路由"的签名（同一网关的其它错误带 `"type":"new_api_error"`）。
逐条实测（`/v1/images/edits`，全部 HTTP 200）得到的对接事实：

| 变量 | 结论 |
|---|---|
| `image` | 顶层字段，**接受 data URL 字符串**；2–3 张时传字符串数组也能用（第二张确实被模型引用） |
| `size` | 接受 `"W*H"`（DashScope 写法，原样回传交付尺寸），也接受 `"WxH"`；`1141*765` 原样兑现 |
| `n` | 返回 `data` 数组里的 n 个条目 |
| `seed` / `negative_prompt` / `watermark` / `prompt_extend` | 都接受，不报未知字段 |
| 缺 `size` | 按输入比例自动选（实测 1141×765 输入 → 2496×1664，1.9–4.2 MP 区间） |
| 像素上限 | **4,194,304（2048×2048）**，超出时 **HTTP 500 + `code:"invalid_request"`**（不是 400！） |
| 响应形状 | `{"data":[{"url":"https://…oss…png?Expires=…&Signature=…"}],"created":…}`，只有 URL、无 `usage` |
| 结果图下载 | 阿里云 OSS 预签名 URL，**不能带 Authorization 头**（`download_bytes` 已经不带） |
| 限流 | 连续快速请求会 429 `request rate limit has been reached`（transport 会重试） |

由此产生的两个固定行为：

1. `qwen_images` 用 2048² 作为面积上限（`api_qwen.MAX_AREA_RELAY`），`match input` 按它收敛；
   超出时报 `invalid_size` 并提示"要 2560×2560 请直连官方端点"。
2. `transport.looks_like_validation_failure()` 让 `500 + invalid_request` **不再重试**——
   这类错误的成因是参数，重试只会白等。


---

## 12. 排障表

| 症状 | 大概率原因 | 处理 |
|---|---|---|
| 节点不在 ComfyUI 里 | `ce_nodes.py` 语法错误 / registration 漏项 / 缓存 | 看启动日志的 import 报错；删 `__pycache__` 重启 |
| `response_log` 里 `protocol_mismatch` | 提供商 `api_protocol` 与节点不匹配（`auto` 猜错也算） | 在提供商节点显式选协议 |
| `endpoint_missing` / `model_missing` / `provider_invalid` | 提供商节点没填全 | 补齐后重排 |
| "temporary API credential expired" | 从历史重新入队：密钥句柄一次性消费 | 用界面 Queue（前端每次重发原始值），或重新填 key |
| 1×1 图片 + 一段报错串 | 该条目调用失败（**这是设计行为**） | 读 `response_log` 的 `parsed.error` / `details` |
| 请求 400 unknown field | 上游不认识某个字段 | 用「不发送」关掉该字段；或按中转文档改 `extra_headers` |
| Qwen 报 `404 Invalid URL (POST /v1/services/aigc/...)` | 中转只开了 OpenAI 兼容路由 | 已在 0.3.0 自动降级到 `/images/edits`（看 `request_log.extra.fallback`）；想省掉那次 404 就把协议改成 `qwen_images` 或 `auto` |
| Qwen 报 `500 … invalid_request … total pixels must be in [262144, 4194304]` | 中转上游只到 2048² | 把尺寸降到 2048×2048 以内（`qwen_images` 路由已在本地拦下并提示） |
| 结果图下载失败 | 签名 URL 过期 / 网关拦 | 看 `phase="download"` 的日志；缩短排队时间 |
| 图像端口不增长 | `INPUT_TYPES` 里的端口名不是 `image_N`，或 `input.optional` 没声明 | 端口名必须 `image_1..image_N` 且在 `optional` 里 |
| 拼接后图片被拉变形 | 用了 `fit=stretch`、或走了原生 `ImageBatch` / Impact `MakeImageBatch`（它们按第一张硬缩） | 用本节点默认 `contain`；要**一个像素都不动**就接 `image_list`（见 4.4） |
| 拼接后画面出现边框 | `fit=contain` 在补齐画布（正常行为） | 换 `cover` 裁切，或把 `pad_color` 调成与画面相衬的颜色；`size_mode` 选 `first image` 可保持旧画布 |
| 导出的工作流里还有 key | 「加密导出」关着 | 打开开关再导出（图片元数据始终干净） |
| Qwen 报 `invalid_size` | 面积/比例越界 | 按报错里的数值调 `width/height`；或 edit 用 `auto` |
| Qwen 报 `invalid_image_reference` | 提示词引用 `@ImageN` 超过连上的张数 | 编号按连线顺序从 1 开始；批量输入按帧计数 |

---

## 13. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-13 | **0.5.2：示例工作流改用 ASCII 文件名**（`basic_gpt_image.json` / `batch_gpt_gemini.json` / `gemini_inpaint.json`）。原因：`comfy node publish` 用 `git ls-files` + 本地 locale 解码打包，中文文件名在 GBK 系统上要么被跳过（默认 `core.quotepath=true`），要么让打包直接 `UnicodeDecodeError`（设成 `false` 时），两种情况都会让 Registry 包里缺文件；改用 ASCII 后与 locale 无关。同时 `.gitignore` 忽略打包产物 `node.zip` |
| 2026-11-13 | **公开发布准备**：示例工作流换成三份真实在用的图（`基础示例工作流（GPT）`、`批量API绘图（gemini-GPT二采）`、`gemini局部修复`），删除生成器 `tools/make_examples.py`，`test_example_workflows.py` 改为按真实导出格式校验（seed 控制值、widget 输入标记、连线类型、第三方节点放行、调用节点必须连 provider）；README 重写为面向读者的版本（徽章 / 目录 / mermaid / FAQ）；LICENSE 署名 `let-the-name-be-x1`；`pyproject.toml` 填仓库地址与 PublisherId，新增 `.gitattributes`；测试 227 → 230 项；版本仍 0.5.1 |
| 2026-11 | **批量拼接不再畸变**：`CustomEndpointBatchImages` 改成两路输出 —— `images`（统一画布的张量批次：默认 `max size` + `contain` 等比留边，可切 `cover`/`stretch`，另有 `position`/`pad_color`/`size_multiple`/手动 `width`/`height`）与 `image_list`（`OUTPUT_IS_LIST`，原尺寸直通）；`images.py` 新增画布几何层（`resolve_canvas` / `fit_to_canvas`），`stack_images` 默认不再拉伸（provider 返回多图同源问题一并修掉）；`merge` 改走 `iter_frames`（补上通道/量纲归一化）；测试 215 → 227 项；版本 0.5.0 → 0.5.1 |
| 2026-11 | **局部重绘定版**：`paste_area` 新增并默认 `crop box`（整块裁剪贴回、羽化落在矩形框四边，替代"只贴遮罩"导致的接缝环）；`mask_guide` 默认改 `none`；引导标记外推到 `feather+3` px 之外（修掉"黄环被贴回"）；新增第 5 路输出 `cropped`（每块模型原图）；测试 210 → 215 项；版本 0.4.0 → 0.5.0 |
| 2026-11 | **API 局部重绘**：新增 `CustomEndpointAPIDetailer`（对标 Impact Pack `MaskDetailer`），把"遮罩→连通域→外扩裁剪→缩放→远端编辑→羽化贴回"做成本包的节点；新增 `custom_endpoint/detailer.py`（纯 numpy+PIL：行程编码并查集做连通域、积分图做膨胀、PIL 高斯做羽化、把遮罩渲染成引导标记）；节点重写 `execute` 但复用 `_run_one`，日志按帧输出 JSON 数组；测试 185 → 210 项；版本 0.3.0 → 0.4.0 |
| 2026-11 | **Qwen 双线路**：新增协议 `qwen_images`（中转的 `/images/edits` + `/images/generations`，顶层 `image`）、`auto` 按主机区分两条线路、`_request_alternative` 自动降级（404 Invalid URL → 换线路重试一次并记入日志）、中转面积上限 2048²、`500 + invalid_request` 不再重试、响应解析同时兼容 OpenAI 与 DashScope 两种形状；测试 172 → 185 项；版本 0.2.0 → 0.3.0 |
| 2026-11 | 新增 Qwen 图像（`CustomEndpointQwenImage`）与 Qwen 图像编辑（`CustomEndpointQwenImageEdit`）；`provider.py` 新增 `qwen` 协议、`QWEN_MODEL_IDS`、`qwen-image*` 的 `auto` 推断；新增 `custom_endpoint/api_qwen.py`；测试 151 → 172 项；版本 0.1.0 → 0.2.0 |
| 更早 | 初版：GPT / Gemini / Grok 图像与文本、文件加载、图像批量拼接、提示词列表、三层凭据保护 |

---

## 14. 发布清单（GitHub / Comfy Registry）

### 14.1 首次推送

仓库名建议与目录同名——ComfyUI 是按目录名加载节点包的：

```bash
git remote add origin https://github.com/let-the-name-be-x1/ComfyUI-Custom-Endpoint-Nodes.git
git push -u origin master
```

- `.gitattributes` 已统一 LF（`* text=auto eol=lf`），Windows 上 `core.autocrlf=true` 也不会让工作树抖动。
- `.gitignore` 已排除 `__pycache__/`、`.pytest_cache/`、`input/`、`output/`；推送前 `git status` 应该是干净的。

### 14.2 发版前检查

- [ ] `pyproject.toml` 的 `version` 已升；Registry 已发布的版本不能覆盖，只能发新版本。
- [ ] `README.md` 的节点数、测试数、示例清单与代码一致（本文件第 1 章的快速事实同步）。
- [ ] 全套 `pytest -q` 绿（230 项）。
- [ ] 示例工作流里没有真实端点 / 密钥 / 本地绝对路径——`test_example_workflows.py` 会拦。
- [ ] 示例工作流文件名保持 **ASCII**：`comfy node publish` 用 `git ls-files` + 本地 locale 打包，
  中文名在 GBK 系统上会被跳过或让打包崩溃（详见第 13 章 0.5.2 那条）。
- [ ] `LICENSE` 的年份与署名正确。
- [ ] 代码与文档里没有调试 `print`、没有本机绝对路径。

### 14.3 发布到 Comfy Registry

```bash
pip install comfy-cli
comfy --skip-prompt login
comfy registry publish
```

`[tool.comfy]` 的 `PublisherId` 必须与 registry.comfy.org 上注册的 publisher id 一致；
`DisplayName` 是 Manager 里显示的名字。首个版本发布后，ComfyUI Manager 就能搜到本包。
