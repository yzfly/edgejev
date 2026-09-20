"""置信度。各家用的公式不同，作为后端的一个可选维度。"""
import math
from typing import Sequence

import numpy as np


def entropy(p: np.ndarray, k: int) -> float:
    """归一化香农熵：1 - H(p)/log(k)。laya / 官方 Jev 的 Choice 用这个。"""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def margin(p: Sequence[float], k: int = None) -> float:
    """(p_max - 1/K) / (1 - 1/K)。PlayJev / OpenJev 用这个。"""
    k = k or len(p)
    if k < 2:
        return 1.0
    return float((max(p[:k]) - 1.0 / k) / (1.0 - 1.0 / k))


def binary(p: np.ndarray, k: int = 2) -> float:
    """noul 专用：max(p_true, 1 - p_true)。"""
    v = float(p[1])
    return max(v, 1.0 - v)


FUNCS = {"entropy": entropy, "margin": margin, "binary": binary}
