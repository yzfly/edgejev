"""nanojev：Qwen3-0.6B 解码器，每个候选一片叶子，读叶子末尾 EOS 的隐状态。

打分头是 LayerNorm → Linear(h, 1)，每个候选出一个标量；choice 再过一层候选之间的
set attention 加一个残差修正，score 和 noul 不过。noul 只有「命题为真」一条路径，
logits 记作 [0, z]，softmax 后就是 sigmoid(z)。这些都随模型导进 ONNX。

上游每个候选独立跑一条完整路径，这里用 prefix_tree 把公共前缀合并成一次前向，
数值与上游逐路径的结果一致。上游的类型名是 boolean，对外仍然叫 noul。
"""
from ..core import BackendSpec

SPEC = BackendSpec(
    name="nanojev",
    layout="prefix_tree",
    attention="block_causal",
    readout="model_logits",
    runtime="onnx",
    layout_kw={"type_names": {"noul": "boolean"}},
    input_names=["input_ids", "position_ids", "attn_mask_4d", "leaf_idx", "leaf_mask", "qtype"],
    extras={"default_model": "C-Tianyu/NanoJev",
            "default_revision": "unified-games-v1",
            "max_len": 8192,
            # 标量头只有一层 Linear，和 kev 一样怕量化噪声。实测 int8 对上游 dev logits
            # 最大偏差 0.89（fp32 是 0.029），60 条里 4 条 argmax 翻转，所以默认不量化。
            "default_precision": "fp32"},
)
