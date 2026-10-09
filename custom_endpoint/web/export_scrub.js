import { app } from "../../scripts/app.js";

/**
 * 加密导出：当提供商节点的「加密导出」开启时，从**导出的工作流副本**中抹除
 * API 端点与密钥。
 *
 * 实现要点：只清理 serialise 出来的副本，绝不动图上活动节点的 widget 值，
 * 否则当前会话会立刻失去凭据。图片元数据由服务端 guard 无条件清理，与本文件
 * 无关（关闭加密导出时元数据依然干净）。
 */

export const PROVIDER_TYPE = "CustomEndpointProvider";
export const SECRET_WIDGET_NAMES = ["endpoint", "api_key", "base_url"];
const URL_LIKE = /^https?:\/\//i;
const KEY_LIKE = /^(sk-|sess-|key-|xai-|AIza)/;

function isProvider(entry) {
    return entry?.type === PROVIDER_TYPE;
}

function liveNodeFor(graph, id) {
    return graph?.getNodeById?.(id) ?? (graph?._nodes ?? []).find((node) => node.id === id);
}

function widgetNamesOf(node) {
    return (node?.widgets ?? []).map((widget) => String(widget?.name ?? ""));
}

function encryptEnabled(node, names, values) {
    if (node) {
        const widget = (node.widgets ?? []).find((item) => item.name === "encrypt_export");
        if (widget) {
            return Boolean(widget.value);
        }
    }
    if (Array.isArray(values)) {
        const index = names.indexOf("encrypt_export");
        if (index >= 0 && index < values.length) {
            return Boolean(values[index]);
        }
    }
    return true; // the node default
}

function looksSecret(value) {
    if (typeof value !== "string" || !value) {
        return false;
    }
    return URL_LIKE.test(value) || KEY_LIKE.test(value);
}

function scrubValues(names, values) {
    if (Array.isArray(values)) {
        names.forEach((name, index) => {
            if (index >= values.length) {
                return;
            }
            if (SECRET_WIDGET_NAMES.includes(name)) {
                values[index] = "";
            } else if (looksSecret(values[index])) {
                values[index] = "";
            }
        });
        if (names.length !== values.length) {
            // Widget order shifted (a widget was converted to an input) or the
            // live node is unavailable: fall back to pattern matching.
            values.forEach((value, index) => {
                if (looksSecret(value)) {
                    values[index] = "";
                }
            });
        }
        return;
    }
    if (values && typeof values === "object") {
        for (const name of SECRET_WIDGET_NAMES) {
            if (name in values) {
                values[name] = "";
            }
        }
        for (const [key, value] of Object.entries(values)) {
            if (looksSecret(value)) {
                values[key] = "";
            }
        }
    }
}

/** Scrub provider secrets from an already-serialized workflow object. */
export function scrubSerializedWorkflow(data, liveGraph) {
    const nodes = data?.nodes;
    if (!Array.isArray(nodes)) {
        return data;
    }
    for (const entry of nodes) {
        if (!isProvider(entry)) {
            continue;
        }
        const live = liveNodeFor(liveGraph, entry.id);
        const names = widgetNamesOf(live);
        const values = entry.widgets_values;
        if (!encryptEnabled(live, names, values)) {
            continue;
        }
        scrubValues(names, values);
        const inputs = entry.inputs;
        if (inputs && typeof inputs === "object" && !Array.isArray(inputs)) {
            for (const name of SECRET_WIDGET_NAMES) {
                if (typeof inputs[name] === "string") {
                    inputs[name] = "";
                }
            }
        }
    }
    return data;
}

function patchGraphPrototype() {
    const LGraph = globalThis.LiteGraph?.LGraph;
    const prototype = LGraph?.prototype;
    if (!prototype || prototype.__customEndpointScrubPatched) {
        return Boolean(prototype);
    }
    const original = prototype.serialize;
    if (typeof original !== "function") {
        return false;
    }
    prototype.serialize = function (...args) {
        const data = original.apply(this, args);
        try {
            scrubSerializedWorkflow(data, this);
        } catch (error) {
            console.warn("[CustomEndpoint] 加密导出清理失败", error);
        }
        return data;
    };
    prototype.__customEndpointScrubPatched = true;
    return true;
}

function patchActiveGraph() {
    const graph = app.graph;
    if (!graph || graph.__customEndpointScrubPatched) {
        return;
    }
    const original = graph.serialize;
    if (typeof original !== "function") {
        return;
    }
    graph.serialize = function (...args) {
        const data = original.apply(this, args);
        try {
            scrubSerializedWorkflow(data, this);
        } catch (error) {
            console.warn("[CustomEndpoint] 加密导出清理失败", error);
        }
        return data;
    };
    graph.__customEndpointScrubPatched = true;
}

app.registerExtension({
    name: "CustomEndpoint.ExportScrub",
    setup() {
        patchGraphPrototype();
    },
    afterConfigureGraph() {
        patchActiveGraph();
    },
    nodeCreated() {
        patchGraphPrototype();
    },
});
