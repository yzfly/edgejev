"""后处理：温度标定、读出归一化、按原语组装答案。各后端共用。"""
from typing import Dict, List, Sequence

import numpy as np

from .core.render import QTYPES, QTYPE_NAMES


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def vocab_slot_probs(last_logits: np.ndarray, slot_ids: Sequence[int], k: int):
    """词表读出：全词表 softmax 后取 K 个字母槽再归一化。

    同时返回 allowed_mass——全词表概率落在这 K 个槽上的份额。base 骨干上它通常在
    1e-4 量级，低不代表出错；真正有判别力的是槽之间的相对排序。
    """
    z = last_logits.astype(np.float64)
    full = np.exp(z - z.max())
    full /= full.sum()
    sel = full[list(slot_ids[:k])]
    mass = float(sel.sum())
    return (sel / max(sel.sum(), 1e-12)), mass


def format_answer(spec, q: Dict, p: np.ndarray, conf: float, act_prob=None) -> Dict:
    extra = {}
    if act_prob is not None:
        extra["action"] = {"act_probability": round(float(act_prob), 4)}
    t = q["t"]
    if t == "choice":
        keys = spec.keys_of(q)
        return dict(type="choice", choice=keys[int(p.argmax())],
                    probabilities={k: round(float(v), 4) for k, v in zip(keys, p)},
                    confidence=conf, **extra)
    if t == "score":
        return dict(type="score", score=round(float((np.arange(len(p)) * p).sum()), 4),
                    legend={str(i): c for i, c in enumerate(q["crit"])},
                    probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    confidence=conf, **extra)
    noul = float(p[1])
    return dict(type="noul", noul=round(noul, 4), confidence=conf, **extra)


def assemble(spec, questions: List[Dict], qids: List[str], logits, act, ks,
             temperature, temperature_by_options) -> Dict:
    answers = {}
    for r, (qid, q) in enumerate(zip(qids, questions)):
        k = ks[r]
        qt = QTYPES[q["t"]]
        scale = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])
        p = softmax(logits[r, :k] / max(1e-3, float(scale)))
        conf = spec.confidence_of(q, p, k)
        answers[qid] = format_answer(spec, q, p, conf, None if act is None else act[r, 0])
    return answers
