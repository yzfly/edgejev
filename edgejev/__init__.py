"""EdgeJev —— 在自己的设备上跑 Jev 式的类型化决策：快、小、运行时不需要 torch。

    edgejev build --backend laya --out ./jev-int8
    from edgejev import Agent
    Agent("./jev-int8").system_one(state, questions)
"""
from .agent import Agent, load

__all__ = ["Agent", "load"]
__version__ = "0.4.0"
