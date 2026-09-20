"""后端注册表。新增一个后端＝加一个模块并在这里登记。"""
from . import laya

_REGISTRY = {"laya": laya}

try:
    from . import kev
    _REGISTRY["kev"] = kev
except Exception:      # kev 适配器是可选的
    pass


def get(name):
    if name not in _REGISTRY:
        raise ValueError("未知后端 %r，已注册：%s" % (name, ", ".join(sorted(_REGISTRY))))
    return _REGISTRY[name]


def names():
    return sorted(_REGISTRY)
