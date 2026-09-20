"""后端 = 在共用原语上的一组声明，而不是一份上游代码的搬运。

一个 System One 后端只在四个维度上有区别：

    布局 layout          token 怎么排、读出位置在哪
    注意力 attention     双向 / block-causal / 因果
    读出 readout         模型直接出 logits，还是从词表的特定 token 上取
    置信度 confidence    归一化熵 / 边际 / 二值

其余（渲染带类型的问题、温度标定、softmax、组装答案）全部共用。
新增一个后端就是填这张表，不需要再写一遍序列构造。
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import confidence as conf_mod
from .layout import LAYOUTS, Encoded
from .render import QTYPES, option_keys


@dataclass
class BackendSpec:
    name: str
    layout: str                                  # LAYOUTS 里的键
    attention: str = "bidirectional"             # bidirectional | block_causal
    readout: str = "model_logits"                # model_logits | vocab_slots
    runtime: str = "onnx"                        # onnx | torch-vlm
    #: 每个原语用哪种置信度。默认跟官方 Jev 一致：choice/score 用熵，noul 用二值。
    confidence: Dict[str, str] = field(
        default_factory=lambda: {"choice": "entropy", "score": "entropy", "noul": "binary"})
    render_kw: Dict = field(default_factory=dict)
    layout_kw: Dict = field(default_factory=dict)
    #: ONNX 图的输入名，供 build 期导出时对齐
    input_names: List[str] = field(default_factory=list)
    extras: Dict = field(default_factory=dict)

    def prepare(self, enc, state, questions, cfg) -> Encoded:
        fn = LAYOUTS[self.layout]
        return fn(enc, state, questions, cfg,
                  render_kw=self.render_kw or None, **self.layout_kw)

    def confidence_of(self, q, p, k) -> float:
        fn = conf_mod.FUNCS[self.confidence.get(q["t"], "entropy")]
        return round(float(fn(p, k)), 4)

    def keys_of(self, q) -> List[str]:
        return option_keys(q)
