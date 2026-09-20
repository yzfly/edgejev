"""训练一个自己的 System One 决策模型，然后用同一条路径导出、量化、部署。

    edgejev-train data   --out data/          # 把带标注的样本转成训练格式
    edgejev-train fit    --data data/ --out runs/my-jev
    edgejev build --backend laya --model runs/my-jev --out ./my-jev-int8
    edgejev serve --model ./my-jev-int8

设计上刻意与推理侧共用同一个渲染器（`edgejev.backends.laya.build_sequence`），
训练看到的序列和线上请求看到的逐 token 相同——这是复现者们反复强调的一条：
训练/推理渲染不一致是最容易出、又最难查的一类 bug。
"""
from .losses import proper_scoring_loss, ece
from .data import Example, load_jsonl, write_jsonl

__all__ = ["proper_scoring_loss", "ece", "Example", "load_jsonl", "write_jsonl"]
