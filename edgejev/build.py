"""构建期：把上游 checkpoint 转成 EdgeJev 目录（ONNX + tokenizer + 配置）。

只有这一步需要 torch（`pip install 'edgejev[build]'`），转完运行时就不需要了。
"""
import json
import os
import shutil
import sys

PRECISIONS = ("int8", "int8-pc", "int8-static", "mixed", "fp32")
DEFAULT_MODELS = {"laya": "convaiinnovations/laya-multilingual",
                  "kev": "jaredpalmer/kev-0.5b",
                  "playjev": "OmniJev/PlayJev-0.8B"}


# ---------------------------------------------------------------- laya 导出
def _export_laya(model_id, subfolder, out_dir):
    """返回 (fp32_onnx_path, cfg_dict, tokenizer_json_path, decision_node_cut)。"""
    import torch
    from torch.export import Dim
    import laya
    from laya.common import QTYPES, build_sequence, collate_items

    ag = laya.load(model_id, device="cpu", subfolder=subfolder)
    ag.model.eval().float()
    cfg, tok = ag.cfg, ag.tok

    qs = [{"t": "choice", "ins": "which team", "crit": {"a": "1", "b": "2", "c": "3"}},
          {"t": "score", "ins": "how severe", "crit": ["low", "med", "high"]},
          {"t": "noul", "ins": "is it urgent", "crit": None}]
    items = []
    for q in qs:
        seq, mk = build_sequence(tok, "a short piece of state text", q,
                                 cfg.get("max_len", 1024), cfg.get("head_max_len", 256))
        items.append({"ids": seq, "markers": mk, "qtype": QTYPES[q["t"]]})
    b = collate_items([items], tok.pad_token_id)
    args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])

    path = os.path.join(out_dir, "_fp32.onnx")
    B, L, K = Dim("B", min=1, max=64), Dim("L", min=8, max=1024), Dim("K", min=2, max=255)
    # 必须 dynamo：旧的 TorchScript 导出器会把 head 里 nn.MultiheadAttention 的 batch/seq
    # 固化成导出时的形状，换个输入长度就抛 Reshape 错误，而且用同一批次做比对发现不了。
    torch.onnx.export(
        ag.model, args, path,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_shapes={"input_ids": {0: B, 1: L}, "attention_mask": {0: B, 1: L},
                        "marker_pos": {0: B, 1: K}, "marker_mask": {0: B, 1: K},
                        "qtype": {0: B}},
        opset_version=20, dynamo=True, external_data=False)

    import onnx
    m = onnx.load(path)
    del m.graph.value_info[:]       # dynamo 留下的标注与推断冲突，会挡住量化
    onnx.save(m, path)

    tmp = os.path.join(out_dir, "_tok")
    tok.save_pretrained(tmp)
    tok_json = os.path.join(tmp, "tokenizer.json")
    if not os.path.exists(tok_json):
        sys.exit("这个 checkpoint 不是 fast tokenizer，导不出 tokenizer.json")

    conf = {"backend": "laya", "model_name": cfg.get("model_name", "laya"),
            "source_model": model_id, "max_len": cfg.get("max_len", 1024),
            "head_max_len": cfg.get("head_max_len", 256),
            "temperature": cfg.get("temperature", [1.0, 1.0, 1.0]),
            "temperature_by_options": cfg.get("temperature_by_options", {}),
            "cls_id": tok.cls_token_id, "sep_id": tok.sep_token_id,
            "mask_id": tok.mask_token_id, "pad_id": tok.pad_token_id,
            "mask_token": tok.mask_token}
    return path, conf, tok_json, "type_emb.weight"


def _write_playjev(model_id, subfolder, out_dir):
    """playjev 不导 ONNX：Qwen3.5 的线性注意力层没有对应的 ONNX 算子。

    这里只落一份配置，运行时直接用 transformers 加载；EdgeJev 统一的是 API 和 serve。
    """
    from huggingface_hub import snapshot_download

    local = os.path.exists(model_id)
    if not local:
        print("  预拉权重 ...", flush=True)
        snapshot_download(model_id)
    conf = {"backend": "playjev", "runtime": "torch-vlm", "template": "plain",
            "model_name": "playjev", "source_model": model_id,
            "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}
    return None, conf, None, None


EXPORTERS = {"laya": _export_laya, "playjev": _write_playjev}


# ------------------------------------------------------------------- 量化
class _Calib:
    """给静态量化用的标定数据。

    动态量化在运行时按实际张量算激活 scale，padding 一变 scale 就变，
    于是**同一条输入跟谁一批会影响它的答案**（实测 logits 能差 2.4）。
    静态量化把 scale 在这里固定下来，换来批次无关、可复现的结果。
    """

    def __init__(self, enc, cfg, backend, n=64):
        import itertools
        states = ["The customer was charged twice and is asking for a refund.",
                  "客户反馈 App 打开就闪退，已经第三次提工单了。",
                  "Server returned 502 for every request during the sale. " * 6,
                  "请问企业版一年多少钱？有没有教育优惠？"]
        qs = [{"t": "choice", "ins": "which team", "crit": {"a": "one", "b": "two", "c": "three"}},
              {"t": "score", "ins": "how severe", "crit": ["low", "med", "high"]},
              {"t": "noul", "ins": "is it urgent", "crit": None},
              {"t": "choice", "ins": "pick", "crit": {c: None for c in "abcdefgh"}}]
        self.data = []
        for st, q in itertools.islice(itertools.product(states, qs), n):
            self.data.append(backend.prepare(enc, st, [q], cfg)["feed"])
        # 也放几个多题批次，让激活范围覆盖 batch>1 的情形
        for st in states:
            self.data.append(backend.prepare(enc, st, qs[:3], cfg)["feed"])
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter(self.data)


def quantize(src, dst, precision, decision_marker=None, calib=None):
    import onnx
    from onnxruntime.quantization import (QuantType, quantize_dynamic, quantize_static,
                                          CalibrationMethod, QuantFormat)

    if precision == "int8-static":
        if calib is None:
            raise ValueError("静态量化需要标定数据")
        print("  [实验性] int8-static 目前会大幅掉点：MinMax 标定下实测 AG News 25.8%"
              "（随机基线 25%）、emotion 29.8%，且比动态量化慢约 4 倍。"
              "它能做到批次无关，但标定方法还没调好——生产请用 int8 或 fp32。", flush=True)
        quantize_static(src, dst, calib, quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        per_channel=True, calibrate_method=CalibrationMethod.MinMax)
        return []

    # 有符号 int8 是唯一值得选的：x86 的 AVX512-VNNI 只对 QInt8 有快路径，
    # 实测同体积的 QUInt8 慢将近一倍（27.9ms vs 15.6ms）。ARM 走 SDOT，差距没这么大。
    per_channel = precision in ("int8-pc", "mixed")
    exclude = []
    if precision == "mixed" and decision_marker:
        g = onnx.load(src, load_external_data=False).graph
        cut = next((i for i, n in enumerate(g.node) if decision_marker in n.input), None)
        if cut is not None:
            exclude = [n.name for n in g.node[cut:] if n.op_type in ("MatMul", "Gemm")]
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8,
                     per_channel=per_channel, nodes_to_exclude=exclude)
    return exclude


# ------------------------------------------------------------------- 自检
def verify(out_dir):
    """多形状自检：不同问题数、不同 state 长度、不同选项数都要跑通。

    这一步是硬性的——导出器把形状固化成导出时的批次是真实会发生的 bug，
    只在换输入形状时才暴露，所以不能只靠一次数值比对。
    """
    from .agent import Agent

    print("\n自检 ...", flush=True)
    ag = Agent(out_dir)
    print("  provider: %s" % ag.provider_note)
    cases = [
        ("三题 / 短 state", "short state",
         {"a": {"type": "choice", "instructions": "x", "criteria": {"p": "1", "q": "2", "r": "3"}},
          "b": {"type": "score", "instructions": "y", "criteria": ["l", "m", "h"]},
          "c": {"type": "noul", "instructions": "z"}}),
        ("单题 / 长 state", "a much longer piece of state text " * 30,
         {"only": {"type": "noul", "instructions": "is this long"}}),
        ("八选项 / 中文", "客户说账号被扣了两次款，很生气。",
         {"k": {"type": "choice", "instructions": "选一个",
                "criteria": {c: None for c in "abcdefgh"}}}),
    ]
    ok = True
    for name, state, qs in cases:
        try:
            r = ag.system_one(state, qs)
            assert set(r["answers"]) == set(qs)
            print("  [OK] %s" % name)
        except Exception as e:
            ok = False
            print("  [失败] %s: %s" % (name, str(e)[:160]))

    # 批次无关性：同一条输入单独跑和跟别人一批跑，结果应该一致。
    # 动态量化做不到这一点（激活 scale 随 padding 变），所以这里只警告不失败。
    import numpy as np
    from .backends import laya as _laya
    q = {"t": "noul", "ins": "is this urgent", "crit": None}
    texts = ["short one", "a considerably longer piece of state text " * 8]
    solo = []
    for t in texts:
        p = _laya.prepare(ag.enc, t, [q], ag.cfg)
        solo.append(ag.sess.run(None, p["feed"])[0][0, :2].copy())
    items = [_laya.build_sequence(ag.enc, t, q, ag.cfg["max_len"], ag.cfg["head_max_len"])
             for t in texts]
    L = max(len(s_) for s_, _ in items)
    ids = np.full((2, L), ag.enc.pad_id, dtype=np.int64)
    att = np.zeros((2, L), dtype=np.int64)
    mp = np.zeros((2, 2), dtype=np.int64); mm = np.zeros((2, 2), dtype=bool)
    for i, (s_, m_) in enumerate(items):
        ids[i, :len(s_)] = s_; att[i, :len(s_)] = 1
        mp[i, :len(m_)] = m_; mm[i, :len(m_)] = True
    lgb = ag.sess.run(None, {"input_ids": ids, "attention_mask": att, "marker_pos": mp,
                             "marker_mask": mm, "qtype": np.full(2, 2, dtype=np.int64)})[0]
    drift = max(float(np.abs(lgb[i, :2] - solo[i]).max()) for i in range(2))
    if drift < 1e-3:
        print("  [OK] 批次无关性（单条与批量一致，漂移 %.1e）" % drift)
    else:
        print("  [注意] 批次会影响结果：logits 漂移 %.2f。动态量化按实际张量算激活 scale，"
              "padding 一变 scale 就变。要可复现请用 --precision int8-static 或 fp32，"
              "或者固定 batch=1。" % drift)
    return ok


def run(backend="laya", model=None, subfolder=None, out=None, precision="int8",
        keep_fp32=False):
    if backend not in EXPORTERS:
        sys.exit("后端 %r 还没有构建器，目前支持：%s" % (backend, ", ".join(EXPORTERS)))
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("缺少转换依赖，请先： pip install 'edgejev[build]'")

    model = model or DEFAULT_MODELS[backend]
    os.makedirs(out, exist_ok=True)
    print("加载 %s（后端 %s）..." % (model, backend), flush=True)
    fp32_path, conf, tok_json, marker = EXPORTERS[backend](model, subfolder, out)

    if fp32_path is None:          # 不走 ONNX 的后端（playjev）
        with open(os.path.join(out, "edgejev.json"), "w", encoding="utf-8") as f:
            json.dump(conf, f, ensure_ascii=False, indent=2)
        print("  运行时 %s（不导 ONNX，无量化）" % conf["runtime"])
        print("\n完成 -> %s" % out)
        print('  from edgejev import Agent; Agent("%s").system_one(image, questions)' % out)
        return
    print("  fp32 %.0f MB" % (os.path.getsize(fp32_path) / 1e6), flush=True)

    final = os.path.join(out, "model.onnx")
    if precision == "fp32":
        shutil.move(fp32_path, final)
    else:
        print("量化 (%s) ..." % precision, flush=True)
        calib = None
        if precision == "int8-static":
            from .tokenize import Encoder
            from . import backends as _b
            shutil.copyfile(tok_json, os.path.join(out, "tokenizer.json"))
            ids_map = {k: conf[k] for k in conf if k.endswith("_id")}
            ids_map["mask_token"] = conf.get("mask_token", "<mask>")
            calib = _Calib(Encoder.from_file(os.path.join(out, "tokenizer.json"), ids_map),
                           conf, _b.get(backend))
        excl = quantize(fp32_path, final, precision, marker, calib)
        if excl:
            print("  决策路径保持 fp32 的节点：%d 个" % len(excl))
        if not keep_fp32:
            os.remove(fp32_path)
    print("  model.onnx %.0f MB" % (os.path.getsize(final) / 1e6), flush=True)

    shutil.copyfile(tok_json, os.path.join(out, "tokenizer.json"))
    shutil.rmtree(os.path.join(out, "_tok"), ignore_errors=True)
    conf.update(onnx_file="model.onnx", precision=precision)
    with open(os.path.join(out, "edgejev.json"), "w", encoding="utf-8") as f:
        json.dump(conf, f, ensure_ascii=False, indent=2)

    if not verify(out):
        sys.exit("自检未通过，产出的模型不可用")
    print("\n完成 -> %s" % out)
    print('  from edgejev import Agent; Agent("%s").system_one(state, questions)' % out)
