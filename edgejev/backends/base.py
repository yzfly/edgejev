"""后端适配器协议。

不同的开源 Jev 复现，序列构造和打分头完全不同：

* laya   —— mmBERT 编码器，每题一个 batch 行，读 [MASK] 标记位的隐状态
* kev    —— Qwen 解码器 + LoRA，多题打包进一条序列，block-causal 掩码，
            PointerHead 用 <decide> 的隐状态去打各个 </opt> 位置的分

所以后端必须自己负责：把 (state, questions) 变成模型输入、说明 ONNX 的输入输出签名、
以及构建期怎么从上游 checkpoint 导出。后处理（温度、熵置信度、答案格式）是共用的。
"""
from typing import Any, Dict, List, Protocol


class Backend(Protocol):
    name: str
    #: ONNX 图的输入名，顺序无关
    input_names: List[str]

    def prepare(self, enc, state, questions: List[Dict], cfg: Dict) -> Dict[str, Any]:
        """返回 {"feed": {onnx输入名: ndarray}, "k": [每题的选项数]}。"""

    def read_logits(self, outputs, meta: Dict) -> Any:
        """从 ONNX 输出里取出 [问题数, 最大选项数] 的 logits 和可选的 act 概率。"""
