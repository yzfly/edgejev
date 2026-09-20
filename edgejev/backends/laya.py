"""laya 后端：mmBERT 编码器 + 两层 transformer head + marker 打分。

序列格式： [CLS] <type> question: 指令 [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
每个选项前的 [MASK] 位置记为 marker，scorer 读这些位置的隐状态各出一个 logit。
每个问题占 batch 的一行，所以 N 个问题 = encoder 跑 batch=N。

与上游 laya（PyTorch）逐位一致：fp32 模式下最大概率偏差 0.00000。
"""
import json
from typing import Dict, List, Union

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}

name = "laya"
input_names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
TOKEN_IDS = ["cls_id", "sep_id", "mask_id", "pad_id"]


def serialize_state(state: Union[str, dict, list]) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Dict) -> List[str]:
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
                for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    fc, tc = crit.get("false"), crit.get("true")
    return [
        "false: " + (render_criterion(fc) if fc not in (None, "") else "no, the statement does not hold"),
        "true: " + (render_criterion(tc) if tc not in (None, "") else "yes, the statement holds"),
    ]


def build_sequence(enc, state, q: Dict, max_len: int, head_max_len: int,
                   truncate_left: bool = False):
    mt = enc.mask_token
    opts = render_options(q)
    ins = str(q["ins"]).replace(mt, " ")
    head_ids = enc.ids("%s question: %s" % (q["t"], ins))

    opt_ids = [[enc.mask_id] + enc.ids(" " + o.replace(mt, " "))[:48] for o in opts]
    budget = head_max_len - sum(len(o) for o in opt_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, budget)]

    ids = [enc.cls_id] + head_ids + [enc.sep_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(enc.sep_id)

    room = max(0, max_len - len(ids) - 1)
    st = enc.ids(serialize_state(state).replace(mt, " "))
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [enc.sep_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def prepare(enc, state, questions: List[Dict], cfg: Dict) -> Dict:
    max_len = cfg.get("max_len", 1024)
    head_max_len = cfg.get("head_max_len", 256)
    items = []
    for i, q in enumerate(questions):
        seq, markers = build_sequence(enc, state, q, max_len, head_max_len)
        if len(markers) != len(render_options(q)):
            raise ValueError("第 %d 个问题的选项超出 head_max_len=%d" % (i, head_max_len))
        items.append((seq, markers, QTYPES[q["t"]]))

    n = len(items)
    L = max(len(s) for s, _, _ in items)
    K = max(len(m) for _, m, _ in items)
    ids = np.full((n, L), enc.pad_id, dtype=np.int64)
    att = np.zeros((n, L), dtype=np.int64)
    mpos = np.zeros((n, K), dtype=np.int64)
    mmask = np.zeros((n, K), dtype=bool)
    for i, (seq, markers, _) in enumerate(items):
        ids[i, :len(seq)] = seq
        att[i, :len(seq)] = 1
        mpos[i, :len(markers)] = markers
        mmask[i, :len(markers)] = True
    feed = {"input_ids": ids, "attention_mask": att, "marker_pos": mpos,
            "marker_mask": mmask,
            "qtype": np.array([t for _, _, t in items], dtype=np.int64)}
    return {"feed": feed, "k": [len(m) for _, m, _ in items],
            "input_tokens": int(att.sum())}


def read_logits(outputs, meta):
    logits, act = outputs[0], outputs[1]
    act = np.exp(act - act.max(-1, keepdims=True))
    return logits, act / act.sum(-1, keepdims=True)
