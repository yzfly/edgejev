"""kev：Qwen + LoRA，多题打包一条序列，block-causal 隔离，PointerHead 读出。

选项 span 之间互不可见、共享同一段 position id，<decide> 落在最长 span 之后的
固定位置——于是选项顺序不影响结果。PointerHead 用 <decide> 的隐状态去打每个
</opt> 位置的分，这部分随模型一起导进 ONNX。
"""
from ..core import BackendSpec

SPEC = BackendSpec(
    name="kev",
    layout="packed_branches",
    attention="block_causal",
    readout="model_logits",
    runtime="onnx",
    input_names=["input_ids", "position_ids", "attn_mask_4d", "decide_idx", "opt_idx", "opt_mask"],
    extras={"default_model": "jaredpalmer/kev-0.5b",
            "special_tokens": ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>",
                               "<|box_end|>", "<|fim_suffix|>"],
            "max_state": 384, "max_branch": 1024, "option_isolation": True,
            # PointerHead 是两个 Linear 做点积，中间没有非线性吸收量化噪声，
            # 噪声直接作用在选项排序上。实测 int8 在 emotion(6 选项) 上从 44% 掉到 21%
            # （随机基线 16.7%），所以这个后端默认不量化。
            "default_precision": "fp32"},
)
