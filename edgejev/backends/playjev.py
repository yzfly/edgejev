"""playjev：Qwen3.5-0.8B 视觉模型，从画面读「选项字母」的概率。

state 是像素而不是文本；读出走词表的字母槽而不是专用打分头；置信度用
(p_max - 1/K) / (1 - 1/K)。跑在 torch 上——Qwen3.5 的文本塔是混合线性注意力，
`linear_attention` 层依赖 causal_conv1d / flash-linear-attention 的递归状态核，
没有对应的 ONNX 算子。
"""
from ..core import BackendSpec

# 与上游逐字一致。上游注释说明这段「冻结的 OpenJev 措辞」比
# "You are a System One decision model" 那类开头高 3–6 个点，不要自己加前缀。
SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
FRAME_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

SPEC = BackendSpec(
    name="playjev",
    layout="letter_prompt",
    attention="causal",
    readout="vocab_slots",
    runtime="torch-vlm",
    confidence={"choice": "margin", "score": "margin", "noul": "margin"},
    layout_kw={"system": SYSTEM_PROMPT, "frame_placeholder": FRAME_PLACEHOLDER,
               "instructions_default": "Which move should the player make next?",
               "answer_cue": "Answer:", "slot_prefix": " "},
    extras={"default_model": "OmniJev/PlayJev-0.8B", "template": "plain",
            "default_precision": "fp32"},
)
