# ComfyUI-Custom-Endpoint-Nodes

让 ComfyUI 通过**你自己填写的 API 端点**（官方接口 / 中转站 / 私有代理）调用 GPT、Gemini、Grok 的图像与文本模型。

> 原生「合作伙伴节点」把端点写死并只走 ComfyUI 官方代理，无法指向自定义端点；本包把
> **端点、密钥、模型**抽到一个独立的**提供商节点**，调用节点只负责参数与调用，并附带
> 并发批次、双路日志与"导出即抹除密钥"的加密开关。

## 特性

- **提供商独立成节点**：`API 端点` + `API 密钥` + `模型` + 协议，一个提供商可被多个调用节点复用。
- **加密导出开关**：开启时导出的工作流文件里**清空端点与密钥**；关闭时保留。**图片元数据在任何情况下都不含端点与密钥**。
- **调用节点直连**：像原生节点一样一次到位，不需要"先出请求对象、再交给执行器"。
- **并发批次**：输入为列表（例如提示词按行切分）时并发调用，输出图片/文本/日志均为列表。
- **两路日志**：`request_log` = 实际发出的请求（**不含图片数据**，二进制字段变成尺寸摘要），`response_log` = 服务商返回（状态码/耗时/正文 + 解析结果/报错）。
- **失败不炸队列**：任何调用失败都返回 **1×1 图片** + 报错信息字符串，队列继续跑。
- 参数面与原生合作伙伴节点对齐（`size / quality / background / aspect_ratio / resolution / thinking_level / n / seed …`），
  模型专属字段用「不发送」缺省，避免严格中转因未知字段 400。

## 节点清单

| 节点 | 对应原生节点 | 说明 |
|---|---|---|
| 自定义端点 · API 提供商 | — | 端点 / 密钥 / 模型 / 协议 / 加密导出 / 超时 / 重试 / 额外请求头 |
| 自定义端点 · GPT 图像 | OpenAI GPT Image 2.5 | `gpt-image-2.5-flare`、`gpt-image-2`、`gpt-image-1.5`…，连参考图即走 `/images/edits`，支持原生 `mask` |
| 自定义端点 · Gemini 图像 | Nano Banana 2 | `gemini-3.1-flash-image` / `-lite`，输出 图像 + 思维图 + 文本 |
| 自定义端点 · Grok 图像 | Grok Image / Image Edit | `grok-imagine-image-2.0 / -quality / -pro / -image` |
| 自定义端点 · GPT 对话 | OpenAI ChatGPT | `/chat/completions`（默认，中转友好）或 `/responses` |
| 自定义端点 · Gemini 对话 | Google Gemini | `generateContent`，含系统指令、thinking level、采样参数 |
| 自定义端点 · Grok 对话 | （原生没有） | 按 xAI 兼容 `/chat/completions` 实现 |
| 自定义端点 · GPT 文件加载 | OpenAI ChatGPT Input Files | `.txt` / `.pdf`，可链式串联 |
| 自定义端点 · Gemini 文件加载 | Gemini Input Files | `.txt` / `.pdf` → `inlineData` |
| 自定义端点 · 图像批量拼接 | Batch Images | 多个 IMAGE 输入拼成一个批次 |
| 自定义端点 · 提示词列表 | Split Text 类 | 每行一条提示词 → 触发并发批次 |

## 安装

把仓库克隆/解压到 `ComfyUI/custom_nodes/ComfyUI-Custom-Endpoint-Nodes`，重启 ComfyUI。
无第三方依赖（只用标准库 + ComfyUI 自带的 torch/Pillow）。

## 快速开始

1. 新建 **自定义端点 · API 提供商**：
   - `endpoint`：例如 `https://api.example.com/v1`（不写版本号时自动补 `/v1`；Gemini 协议补 `/v1beta`）
   - `api_key`：点击密码框填写
   - `model`：例如 `gpt-image-2.5-flare` / `gemini-3.1-flash-image` / `grok-imagine-image-2.0`
   - `api_protocol`：不确定就用 `auto`（按模型名推断）；错了会在调用节点的 `response_log` 里给出明确提示
2. 接一个 **自定义端点 · GPT 图像**，把 `provider` 连上，填提示词。
3. `images` → `SaveImage`；`request_log` / `response_log` → 任意文本预览节点。

`example_workflows/` 下有三个可直接拖入的示例（GPT 图像 / Gemini 并发批次 / GPT 对话）。

## 并发批次

调用节点声明了 `INPUT_IS_LIST`：所有输入按"长度 1 广播、长度 N 对齐"处理，`max_concurrency`
决定线程数（1–16，默认 4）。

```
[自定义端点 · 提示词列表] ──prompts──► [自定义端点 · Gemini 图像] ──images──► SaveImage
                                        ├─ texts
                                        ├─ request_log   （每项一条）
                                        └─ response_log  （每项一条）
```
输出列表与输入列表**顺序一致**；某一项失败只影响它自己（那一项是 1×1 图片 + 报错串）。

## 密钥与元数据（三层保护）

| 层 | 机制 | 行为 |
|---|---|---|
| 队列/历史 | 服务端 `on_prompt_handler` | 队列里出现的明文端点/密钥会被换成一次性句柄；明文在整份请求中被就地抹除 |
| 图片元数据 | 上述同一个钩子 | `SaveImage` 写入 PNG 的 `prompt` 与 `workflow` 都只剩句柄；**端点与密钥永远不出现**（无法关闭） |
| 工作流文件 | 前端序列化钩子 | 「加密导出」开启 → 导出 `.json` 时清空该节点的端点与密钥（重载后需重新填写）；关闭 → 保留明文 |

- 句柄保存在**进程内存**（TTL 30 分钟），API Key 句柄**一次性消费**，端点是 TTL 内可重用的句柄。
- 因此从历史记录"重新入队"旧 prompt 会拿不到密钥——请用界面的 Queue（前端每次都会重新发送原始值）。
- 把带密钥的 `.json` 分享出去前，请确认该节点「加密导出」是开启状态。

## 日志格式

`request_log`（图片一律换成摘要，凭据字段打码）：

```json
{"ok":true,"phase":"request","method":"POST","url":"https://api.example.com/v1/images/edits",
 "headers":{"Authorization":"[REDACTED]","Content-Type":"multipart/form-data; boundary=..."},
 "operation":"edit","item_index":0,
 "body":{"model":"gpt-image-2.5-flare","prompt":"a cat","size":"auto","_files":["image","mask"]},
 "images":{"reference_images":["image/png 1024x1024, 512.3 KB"]},"mask":12345}
```

`response_log`：

```json
{"ok":false,"phase":"response","http_status":429,"latency_ms":810,
 "request_id":"req_123","parsed":{"error":"quota exceeded"},
 "body":{"error":{"message":"quota exceeded","code":"rate_limit"}}}
```

## 已知限制

- 只移植 **GPT / Gemini / Grok**；视频类节点（Grok Video、Gemini Omni）未包含。
- 参考图端口按需增长（初始 1 个，连上后出现下一个，最多 6 个）；必须**连续连接**，跳号会在 `response_log` 里报错。
- `mask` 只有 GPT 图像（原生 `/images/edits` 的 mask 字段）；Gemini 与 Grok 没有 mask 字段，传了会明确报错而不是静默丢弃。
- Gemini 图像的「思维图」在原生节点里是可选输出，本包为它单独开了一路 `thought_images`。
- 传输层是同步 urllib + 重试；不含异步任务轮询（部分中转的 `/async` 端点）。

## 测试

使用 ComfyUI 自带的 python（含 torch/Pillow）：

```bash
cd custom_nodes/ComfyUI-Custom-Endpoint-Nodes
"<ComfyUI>/python/python.exe" -m pytest tests/ -q
```

全部离线（假 opener 重放响应），覆盖：请求构造与响应解析（三家）、图片/掩码编码、批次对齐与并发、
每项失败兜底、日志去图与脱敏、提示词守卫（含 PNG 元数据实证）、示例工作流结构校验。

## 目录结构

```
__init__.py                 ComfyUI 入口（NODE_CLASS_MAPPINGS / WEB_DIRECTORY / register_routes）
ce_nodes.py                 节点类（提供商 / 6 个调用节点 / 4 个附属节点）
custom_endpoint/
├── provider.py             协议路由、端点归一化、模型别名
├── secrets.py              会话句柄（内存、TTL）
├── routes.py               session-credential 路由 + on_prompt 守卫
├── transport.py            urllib 客户端（重试/多部分/超时）
├── redact.py, errors.py    脱敏与结构化错误
├── logs.py                 两路日志构造
├── images.py               tensor↔PNG、掩码→RGBA、1×1 兜底图
├── batch.py                批次对齐 + 并发执行
├── widgets.py              「不发送」哨兵与控件规格
├── api_openai.py           GPT 图像 / GPT 对话（chat + responses）
├── api_gemini.py           Gemini 图像 / 文本
├── api_grok.py             Grok 图像 / 文本
└── web/                    前端扩展：密码框、导出抹除、端口按需增长
tools/make_examples.py      示例工作流生成器
tests/                      147 项离线测试
```

## License

MIT
