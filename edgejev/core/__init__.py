"""EdgeJev 的共用原语：渲染、布局、掩码、读出、置信度。

各后端在 `edgejev/backends/` 里只声明「用哪种布局 / 注意力 / 读出 / 置信度」，
序列构造和掩码逻辑都在这里，一处修复全体受益。
"""
from .spec import BackendSpec
from .layout import Encoded

__all__ = ["BackendSpec", "Encoded"]
