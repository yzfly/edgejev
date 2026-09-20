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
            "max_state": 384, "max_branch": 1024, "option_isolation": True},
)
