import { app } from "../../scripts/app.js";

/**
 * 图像端口按需增长：初始只显示 1 个图像输入；每接入一个，就自动出现下一个
 * （最多到该节点声明的槽位数）。所有槽位都在 INPUT_TYPES 里声明，因此服务端
 * 校验正常；这里只是让画布上的插座数量跟随使用情况。
 *
 * 只在「尾部未连接的插座」上做删除，所以永远不会破坏已有连线。
 */

const IMAGE_NAME = /^image_(\d+)$/;
const NORMALIZING = "__customEndpointGrowingInputs";

function imageEntries(node) {
    return (node.inputs ?? [])
        .map((input, index) => ({ input, index }))
        .filter(({ input }) => IMAGE_NAME.test(String(input?.name ?? "")));
}

function isConnected(input) {
    return input?.link !== null && input?.link !== undefined;
}

function declaredMax(nodeType, nodeData) {
    const names = Object.keys(nodeData?.input?.optional ?? {}).filter((name) =>
        IMAGE_NAME.test(name),
    );
    return names.length || 0;
}

export function normalizeImageInputs(node, max) {
    if (!node || node[NORMALIZING] || !max) {
        return;
    }
    node[NORMALIZING] = true;
    try {
        let entries = imageEntries(node);
        if (!entries.length) {
            return;
        }
        const connected = entries.filter(({ input }) => isConnected(input));
        const highestConnected = entries.reduce(
            (acc, { input }, index) => (isConnected(input) ? index : acc),
            -1,
        );
        let desired = Math.min(max, Math.max(1, connected.length + 1));
        desired = Math.max(desired, highestConnected + 1);

        while (entries.length > desired) {
            const last = entries[entries.length - 1];
            if (isConnected(last.input)) {
                break;
            }
            node.removeInput(last.index);
            entries = imageEntries(node);
            if (!entries.length) {
                return;
            }
        }
        while (entries.length < desired && entries.length < max) {
            node.addInput(`image_${entries.length + 1}`, "IMAGE");
            entries = imageEntries(node);
        }
        entries.forEach(({ input }, index) => {
            const name = `image_${index + 1}`;
            input.name = name;
            input.label = `图像 ${index + 1}`;
        });
        recomputeHeight(node);
    } finally {
        node[NORMALIZING] = false;
    }
}

function recomputeHeight(node) {
    if (Array.isArray(node.size) && typeof node.computeSize === "function") {
        const width = node.size[0];
        const computed = node.computeSize([...node.size]);
        if (Array.isArray(computed) && Number.isFinite(computed[1])) {
            node.size[0] = width;
            node.size[1] = computed[1];
        }
    }
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function schedule(node, max) {
    if (node.__customEndpointGrowScheduled) {
        return;
    }
    node.__customEndpointGrowScheduled = true;
    setTimeout(() => {
        node.__customEndpointGrowScheduled = false;
        normalizeImageInputs(node, max);
    }, 0);
}

app.registerExtension({
    name: "CustomEndpoint.GrowImageInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const max = declaredMax(nodeType, nodeData);
        if (!max) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            schedule(this, max);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            schedule(this, max);
            return result;
        };

        const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
            const result = originalOnConnectionsChange?.apply(this, arguments);
            if (type !== 1 && type !== 2) {
                return result;
            }
            const touched = this.inputs?.[index];
            const target = this.inputs?.[linkInfo?.target_slot];
            const isImage = (input) => IMAGE_NAME.test(String(input?.name ?? ""));
            if (isImage(touched) || isImage(target)) {
                schedule(this, max);
            }
            return result;
        };
    },
});
