"""后端注册表。一个后端就是一份 BackendSpec 声明，不是一份上游代码的搬运。"""
from . import kev, laya, nanojev, playjev

_REGISTRY = {m.SPEC.name: m.SPEC for m in (laya, kev, nanojev, playjev)}


def get(name):
    if name not in _REGISTRY:
        raise ValueError("未知后端 %r，已注册：%s" % (name, ", ".join(sorted(_REGISTRY))))
    return _REGISTRY[name]


def names():
    return sorted(_REGISTRY)


def specs():
    return dict(_REGISTRY)
