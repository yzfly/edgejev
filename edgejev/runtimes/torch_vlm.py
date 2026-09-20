"""torch 上的视觉-语言决策运行时（PlayJev / Qwen3.5-0.8B）。

只做一件事：把「画面 + 选项列表」喂进去，取最后一个位置的全词表 logits。
选项怎么渲染、概率怎么从字母槽读出来，都在 `backends/playjev.py` 里。
"""
import io
import os
from typing import Any, List


def _to_pil(state: Any):
    from PIL import Image

    if hasattr(state, "convert"):                      # 已经是 PIL.Image
        return state.convert("RGB")
    if isinstance(state, (bytes, bytearray)):
        return Image.open(io.BytesIO(state)).convert("RGB")
    if isinstance(state, str):
        if state.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(state, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        if not os.path.exists(state):
            raise FileNotFoundError("图片路径不存在：%s" % state)
        return Image.open(state).convert("RGB")
    try:
        import numpy as np
        if isinstance(state, np.ndarray):
            return Image.fromarray(state).convert("RGB")
    except ImportError:
        pass
    raise TypeError("playjev 后端的 state 必须是图片：路径 / URL / bytes / PIL.Image / ndarray，"
                    "收到 %s" % type(state).__name__)


class TorchVLM:
    def __init__(self, model_id, device=None, dtype=None, template="plain"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.template = template
        local = os.path.exists(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local)
        self.processor.tokenizer.padding_side = "left"   # 最后一个位置就是答案槽

        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cpu")
        if dtype is None:
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.device, self.torch, self.dtype = device, torch, dtype
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, local_files_only=local).to(device).eval()
        # 读出在 float32 上做：bf16 的 logit 在量级 25–50 时量化步长到 0.125，会让选项打平。
        # 权重与嵌入是 tied 的，所以单独留一份 float32 副本，而不是把整个头重新转型。
        self.head_w32 = self.model.get_output_embeddings().weight.detach().float()
        self.note = "torch-vlm（%s, %s）" % (device, str(dtype).replace("torch.", ""))

    def letter_slots(self, prefix=" "):
        """解析字母槽的 token id，校验每个都是单 token 且能往返。"""
        from ..core.layout import LETTERS

        tok = self.processor.tokenizer
        ids = []
        for letter in LETTERS:
            text = prefix + letter
            enc = tok.encode(text, add_special_tokens=False)
            if len(enc) != 1 or tok.decode(enc) != text:
                raise ValueError("字母槽 %r 在这个 tokenizer 下不是单个可往返的 token" % text)
            ids.append(enc[0])
        if len(set(ids)) != len(ids):
            raise ValueError("字母槽 token 冲突")
        return ids

    def last_logits(self, image, prompt) -> "Any":
        """返回最后一个位置的全词表 logits，float32 numpy。

        走内层 `model.model` 再手算 logits，而不是 `ForConditionalGeneration` 的 `out.logits`——
        后者在这个架构上出来的分布几乎是平的（全词表最大概率 1e-4，词表 25 万时均匀是 4e-6）。
        """
        enc = self.processor(text=[prompt], images=[image], return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items() if hasattr(v, "to")}
        self._ntok = int(enc["input_ids"].shape[-1])
        with self.torch.inference_mode():
            out = self.model.model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                pixel_values=enc["pixel_values"], image_grid_thw=enc["image_grid_thw"],
                mm_token_type_ids=enc.get("mm_token_type_ids"),
                use_cache=False, return_dict=True)
            h = out.last_hidden_state[:, -1, :].float()   # 左 padding，最后一个位置就是答案槽
            logits = h @ self.head_w32.T
        return logits[0].cpu().numpy()

    @property
    def n_input_tokens(self):
        return getattr(self, "_ntok", 0)
