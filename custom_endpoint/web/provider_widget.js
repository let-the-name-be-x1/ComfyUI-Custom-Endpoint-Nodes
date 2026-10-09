import { app } from "../../scripts/app.js";

/**
 * Masked API-key widget plus Chinese labels for the provider node.
 *
 * The key stays a normal widget value in the *in-memory* workflow (so the user
 * can edit it), but:
 *   - the queue payload is cleaned server-side (guard_prompt_payload), and
 *   - the exported workflow is cleaned by export_scrub.js when 加密导出 is on.
 */

const NODE_TYPE = "CustomEndpointProvider";
const PASSWORD_MASK = "••••••••••••";
const WIDGET_LABELS = {
    endpoint: "API 端点",
    api_key: "API 密钥",
    model: "模型",
    api_protocol: "协议",
    encrypt_export: "加密导出（抹除端点与密钥）",
    timeout_sec: "超时(秒)",
    retries: "重试次数",
    extra_headers: "额外请求头(JSON)",
    provider_name: "提供商名称",
};

function widgetByName(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function markDirty(node) {
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function applyLabels(node) {
    for (const [name, label] of Object.entries(WIDGET_LABELS)) {
        const widget = widgetByName(node, name);
        if (widget) {
            widget.label = label;
        }
    }
    markDirty(node);
}

function openKeyEditor(node, widget) {
    if (widget.__customEndpointEditorOpen) {
        return;
    }
    widget.__customEndpointEditorOpen = true;

    const dialog = document.createElement("dialog");
    dialog.setAttribute("aria-label", "Custom Endpoint API Key");
    Object.assign(dialog.style, {
        border: "1px solid #555",
        borderRadius: "10px",
        padding: "18px",
        color: "#eee",
        background: "#262626",
        minWidth: "420px",
        maxWidth: "calc(100vw - 48px)",
        boxShadow: "0 16px 48px rgba(0, 0, 0, 0.5)",
    });

    const form = document.createElement("form");
    form.method = "dialog";
    Object.assign(form.style, { display: "grid", gap: "12px" });

    const label = document.createElement("label");
    label.textContent = "API 密钥";
    label.style.fontWeight = "600";

    const input = document.createElement("input");
    input.type = "password";
    input.value = String(widget.value ?? "");
    input.autocomplete = "off";
    input.spellcheck = false;
    input.placeholder = "粘贴 API Key";
    Object.assign(input.style, {
        boxSizing: "border-box",
        width: "100%",
        padding: "9px 11px",
        border: "1px solid #666",
        borderRadius: "6px",
        color: "#fff",
        background: "#181818",
        font: "inherit",
    });

    const hint = document.createElement("div");
    hint.textContent =
        "图片元数据永不含该密钥；开启「加密导出」时导出的工作流也会清空端点与密钥。";
    Object.assign(hint.style, { color: "#aaa", fontSize: "12px" });

    const actions = document.createElement("div");
    Object.assign(actions.style, {
        display: "flex",
        justifyContent: "flex-end",
        gap: "8px",
        marginTop: "4px",
    });

    const cancel = document.createElement("button");
    cancel.type = "submit";
    cancel.value = "cancel";
    cancel.textContent = "取消";

    const save = document.createElement("button");
    save.type = "submit";
    save.value = "save";
    save.textContent = "保存";

    actions.append(cancel, save);
    form.append(label, input, hint, actions);
    dialog.append(form);
    document.body.append(dialog);

    dialog.addEventListener(
        "close",
        () => {
            if (dialog.returnValue === "save") {
                const oldValue = widget.value;
                widget.value = input.value.trim();
                widget.callback?.(widget.value, app.canvas, node);
                node.onWidgetChanged?.(widget.name, widget.value, oldValue, widget);
            }
            widget.__customEndpointEditorOpen = false;
            dialog.remove();
            markDirty(node);
        },
        { once: true },
    );

    dialog.showModal();
    input.focus();
    input.select();
}

function secureKeyWidget(node) {
    const widget = widgetByName(node, "api_key");
    if (!widget || widget.__customEndpointSecured) {
        return;
    }
    widget.__customEndpointSecured = true;
    const originalDraw = widget.draw;
    widget.draw = function (ctx, _node, width, y, height) {
        const drawWidth = Number.isFinite(width) ? width : (_node?.size?.[0] ?? 200);
        const hasValue = String(this.value ?? "").length > 0;
        ctx.save();
        ctx.beginPath();
        ctx.roundRect(15, y, Math.max(1, drawWidth - 30), height, [height * 0.5]);
        ctx.fillStyle = "#222";
        ctx.strokeStyle = "#666";
        ctx.fill();
        ctx.stroke();
        ctx.font = "12px sans-serif";
        ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        ctx.fillStyle = "#aaa";
        ctx.fillText("api_key", 26, y + height / 2);
        ctx.textAlign = "right";
        ctx.fillStyle = hasValue ? "#ddd" : "#888";
        ctx.fillText(hasValue ? PASSWORD_MASK : "未设置", drawWidth - 26, y + height / 2);
        ctx.restore();
    };
    widget.onClick = () => openKeyEditor(node, widget);
    widget.mouse = (event) => {
        const type = String(event?.type ?? "");
        if (type === "pointerdown" || type === "mousedown") {
            openKeyEditor(node, widget);
            return true;
        }
        return false;
    };
    if (typeof originalDraw !== "function") {
        // Nothing else to do: the custom draw above is the whole presentation.
    }
}

function applyProviderPresentation(node) {
    if (node?.type !== NODE_TYPE && node?.comfyClass !== NODE_TYPE) {
        return;
    }
    secureKeyWidget(node);
    applyLabels(node);
}

app.registerExtension({
    name: "CustomEndpoint.ProviderWidget",
    nodeCreated(node) {
        applyProviderPresentation(node);
    },
    loadedGraphNode(node) {
        applyProviderPresentation(node);
    },
    afterConfigureGraph() {
        for (const node of app.graph?._nodes ?? []) {
            applyProviderPresentation(node);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) {
            return;
        }
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            applyProviderPresentation(this);
            return result;
        };
        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            applyProviderPresentation(this);
            return result;
        };
    },
});
