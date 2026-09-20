"""训练数据格式与加载。

一条样本 = 一份 state + 一个带类型的问题 + 一个目标分布。
目标可以是硬标签（`label`）也可以是软标签（`target`，来自多人标注的分歧或 teacher 的概率）——
软标签是拿到好校准的关键：人都会犹豫的样本，模型也该犹豫。

JSONL 每行：
  {"state": "...", "type": "choice",
   "instructions": "该转给哪个组",
   "criteria": {"billing": "支付扣款", "technical": "程序缺陷"},
   "label": "billing"}                       # 或 "target": [0.7, 0.3]
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Example:
    state: Union[str, dict, list]
    type: str                       # choice | score | noul
    instructions: str
    criteria: Any = None
    label: Optional[Union[str, int, bool]] = None
    target: Optional[List[float]] = None
    meta: Dict = field(default_factory=dict)

    def options(self) -> List[str]:
        from ..backends.laya import render_options
        return render_options({"t": self.type, "ins": self.instructions, "crit": self.criteria})

    def target_vector(self) -> List[float]:
        """把 label / target 统一成一个和为 1 的向量。"""
        k = len(self.options())
        if self.target is not None:
            if len(self.target) != k:
                raise ValueError("target 长度 %d 与选项数 %d 不符" % (len(self.target), k))
            s = float(sum(self.target))
            if s <= 0:
                raise ValueError("target 之和必须为正")
            return [v / s for v in self.target]
        if self.label is None:
            raise ValueError("label 和 target 至少要有一个")
        if self.type == "choice":
            keys = list(self.criteria.keys()) if isinstance(self.criteria, dict) else list(self.criteria)
            idx = keys.index(self.label)
        elif self.type == "score":
            idx = int(self.label)
        else:
            idx = 1 if self.label in (True, 1, "true", "True") else 0
        v = [0.0] * k
        v[idx] = 1.0
        return v


def load_jsonl(path) -> List[Example]:
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Example(**json.loads(line)))
            except Exception as e:
                raise ValueError("%s 第 %d 行解析失败：%s" % (path, ln, e))
    return out


def write_jsonl(path, examples: List[Example]):
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            d = {k: v for k, v in e.__dict__.items() if v not in (None, {}, [])}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def stats(examples: List[Example]) -> Dict:
    from collections import Counter
    c = Counter(e.type for e in examples)
    ks = Counter(len(e.options()) for e in examples)
    soft = sum(1 for e in examples if e.target is not None)
    return {"总数": len(examples), "按类型": dict(c), "按选项数": dict(sorted(ks.items())),
            "软标签占比": round(soft / max(1, len(examples)), 3)}
