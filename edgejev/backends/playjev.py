"""PlayJev 后端：Qwen3.5-0.8B 视觉模型，从游戏画面一次前向读「选项字母」的概率。

与 laya / kev 的区别，三点都很关键：

1. **输入是像素**，不是文本 state。图像经 Qwen2VL 风格的处理器切成 patch。
2. **读出方式是「字母槽」**：把选项渲染成 `A. name: desc` 的列表，取最后一个位置的
   全词表 logits，只在 " A" / " B" ... 这 K 个 token 上做 float32 softmax。
   上游特意用 float32：bf16 在 logit 量级 25–50 时量化步长到 0.125，会让选项打平。
3. **置信度用 Jev 的 Choice 公式** `(p_max - 1/K) / (1 - 1/K)`，不是 laya 的归一化熵。

⚠️ 这个后端**跑在 torch 上，不走 EdgeJev 的 ONNX + 量化主路径**。原因是 Qwen3.5 的文本塔是
混合线性注意力（config 里 `layer_types` 有 `linear_attention`），那些层依赖
`causal_conv1d` / flash-linear-attention 这类带递归状态的自定义核，没有对应的标准 ONNX 算子。
所以用这个后端时，EdgeJev 只统一了 API 和 `serve`，拿不到「不依赖 torch」和量化加速。
"""
from typing import Dict, List, Sequence

import numpy as np

name = "playjev"
runtime = "torch-vlm"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROMPT_VERSION = "playjev-letters-v1"
# 与上游逐字一致。上游注释说明：这段「冻结的 OpenJev 措辞」比
# "You are a System One decision model" 那种开头高 3–6 个点，所以不要自己加前缀。
SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
DEFAULT_INSTRUCTIONS = "Which move should the player make next?"
FRAME_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
PLAIN_ANSWER_CUE = "Answer:"


def options_block(options: Sequence[dict]) -> str:
    if len(options) > len(LETTERS):
        raise ValueError("%d 个选项超出 %d 个字母槽" % (len(options), len(LETTERS)))
    lines = []
    for letter, opt in zip(LETTERS, options):
        desc = (opt.get("description") or "").strip()
        lines.append("%s. %s: %s" % (letter, opt["name"], desc) if desc
                     else "%s. %s" % (letter, opt["name"]))
    return "\n".join(lines)


def render_suffix(options, instructions=DEFAULT_INSTRUCTIONS) -> str:
    letters = ", ".join(LETTERS[: len(options)])
    return ("Question: %s\n\nOptions:\n%s\n\nAnswer with one letter: %s."
            % (instructions, options_block(options), letters))


def render_state(n_placeholders: int) -> str:
    return "<state>\n" + "\n".join([FRAME_PLACEHOLDER] * n_placeholders) + "\n</state>"


def build_plain_prompt(options, instructions=DEFAULT_INSTRUCTIONS, n_placeholders=1) -> str:
    return "%s\n\n%s\n\n%s\n%s" % (SYSTEM_PROMPT, render_state(n_placeholders),
                                   render_suffix(options, instructions), PLAIN_ANSWER_CUE)


def choice_confidence(probs: Sequence[float]) -> float:
    """Jev 的 Choice 置信度：(p_max - 1/K) / (1 - 1/K)。K=1 时恒为 1。"""
    k = len(probs)
    if k == 1:
        return 1.0
    return (max(probs) - 1.0 / k) / (1.0 - 1.0 / k)


def slot_ids(tokenizer, template="plain") -> List[int]:
    """解析字母槽的 token id，并校验每个都是单 token 且能往返。"""
    ids = []
    for letter in LETTERS:
        text = (" %s" % letter) if template == "plain" else letter
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc) != 1 or tokenizer.decode(enc) != text:
            raise ValueError("字母槽 %r 在这个 tokenizer 下不是单个可往返的 token" % text)
        ids.append(enc[0])
    if len(set(ids)) != len(ids):
        raise ValueError("字母槽 token 冲突")
    return ids


def readout(last_logits: np.ndarray, slot_token_ids: Sequence[int], k: int):
    """全词表 logits（最后一个位置）-> (K 个选项的概率, allowed_mass)。

    allowed_mass 是全词表 softmax 落在这 K 个答案槽上的质量，低于 ~0.5 说明模型
    其实没在回答这道题（比如跑去输出别的 token），是很有用的健康指标。
    """
    z = last_logits.astype(np.float64)
    z = z - z.max()
    full = np.exp(z)
    full /= full.sum()
    sel = full[list(slot_token_ids[:k])]
    allowed_mass = float(sel.sum())
    probs = sel / max(sel.sum(), 1e-12)
    return probs.astype(np.float64), allowed_mass


def options_from_question(q: Dict) -> List[dict]:
    """把 EdgeJev 的带类型问题转成 PlayJev 的选项列表。"""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [{"name": kk, "description": vv if isinstance(vv, str) else None}
                for kk, vv in crit.items()]
    if t == "score":
        return [{"name": "level %d" % i, "description": c if isinstance(c, str) else None}
                for i, c in enumerate(crit)]
    crit = crit or {}
    return [{"name": "false", "description": crit.get("false") or "the statement does not hold"},
            {"name": "true", "description": crit.get("true") or "the statement holds"}]
