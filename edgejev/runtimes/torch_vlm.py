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
        self.note = "torch-vlm（%s, %s）" % (device, str(dtype).replace("torch.", ""))

    def slot_ids(self, backend):
        return backend.slot_ids(self.processor.tokenizer, self.template)

    def last_logits(self, image, prompt) -> "Any":
        """返回最后一个位置的全词表 logits，float32 numpy。

        上游特意在 float32 上读：bf16 在 logit 量级 25–50 时量化步长 0.125，会让选项打平。
        """
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model(**inputs)
        return out.logits[0, -1].float().cpu().numpy()

    @property
    def n_input_tokens(self):
        return getattr(self, "_ntok", 0)
