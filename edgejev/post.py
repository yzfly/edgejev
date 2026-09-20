"""后处理：温度标定、熵置信度、按原语组装答案。各后端共用。"""
import math
from typing import Dict, List

import numpy as np

QTYPE_NAMES = {0: "choice", 1: "score", 2: "noul"}
QTYPES = {v: k for k, v in QTYPE_NAMES.items()}


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """归一化香农熵置信度：1 - H(p)/log(k)。k<2 时恒为 1。"""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def format_answer(q: Dict, p: np.ndarray, conf: float, act_prob=None) -> Dict:
    """组装成与官方 /v1/systemone 同构的答案。"""
    out = {}
    if act_prob is not None:
        out["action"] = {"act_probability": round(float(act_prob), 4)}
    t = q["t"]
    if t == "choice":
        keys = list(q["crit"].keys())
        return dict(type="choice", choice=keys[int(p.argmax())],
                    probabilities={k: round(float(v), 4) for k, v in zip(keys, p)},
                    confidence=conf, **out)
    if t == "score":
        return dict(type="score", score=round(float((np.arange(len(p)) * p).sum()), 4),
                    legend={str(i): c for i, c in enumerate(q["crit"])},
                    probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    confidence=conf, **out)
    noul = float(p[1])
    return dict(type="noul", noul=round(noul, 4),
                confidence=round(max(noul, 1.0 - noul), 4), **out)


def assemble(questions: List[Dict], qids: List[str], logits, act, ks,
             temperature, temperature_by_options) -> Dict:
    answers = {}
    for r, (qid, q) in enumerate(zip(qids, questions)):
        k = ks[r]
        qt = QTYPES[q["t"]]
        scale = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])
        p = softmax(logits[r, :k] / max(1e-3, float(scale)))
        conf = round(confidence_from_probs(p, k), 4)
        answers[qid] = format_answer(q, p, conf, None if act is None else act[r, 0])
    return answers


def to_internal(qdef: Dict) -> Dict:
    import json
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False)
    return {"t": t, "ins": ins, "crit": crit}
