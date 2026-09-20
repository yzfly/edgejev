"""跨平台的 ONNX Runtime execution provider 选择。

Linux / Windows x86：CPU EP，靠 AVX512-VNNI 或 AVX2 跑 int8。
macOS Apple Silicon：优先 CoreML（可落到 ANE / GPU），失败回退 CPU。
ARM Linux：CPU EP，int8 走 SDOT。
"""
import os
import platform
import sys


def detect(prefer=None):
    """返回 (providers, 说明)。prefer 可以是 'cpu' / 'coreml' / 'auto'(默认)。"""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    prefer = (prefer or os.environ.get("EDGEJEV_PROVIDER") or "auto").lower()

    if prefer == "cpu":
        return ["CPUExecutionProvider"], "CPU（显式指定）"

    is_mac_arm = sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")
    if prefer in ("auto", "coreml") and "CoreMLExecutionProvider" in available:
        if is_mac_arm or prefer == "coreml":
            # MLProgram + FP16 关闭：量化图走 CoreML 时用默认设置更稳，
            # 不被支持的子图会自动回退到 CPU。
            return (["CoreMLExecutionProvider", "CPUExecutionProvider"],
                    "CoreML（Apple Silicon，不支持的算子回退 CPU）")

    return ["CPUExecutionProvider"], "CPU（%s %s）" % (sys.platform, platform.machine())


def describe():
    import onnxruntime as ort

    p, why = detect()
    return "onnxruntime %s | 选用 %s | 可用 %s" % (
        ort.__version__, why, ", ".join(sorted(ort.get_available_providers())))
