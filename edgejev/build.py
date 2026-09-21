"""构建期：把上游 checkpoint 转成 EdgeJev 目录（ONNX + tokenizer + 配置）。

只有这一步需要 torch（`pip install 'edgejev[build]'`），转完运行时就不需要了。
"""
import json
import os
import shutil
import sys

PRECISIONS = ("int8", "int8-pc", "int8-static", "mixed", "fp32")
def default_model(backend):
    from . import backends
    return backends.get(backend).extras.get("default_model")


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

    conf = {"backend": "laya", "runtime": "onnx", "model_name": cfg.get("model_name", "laya"),
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




# ----------------------------------------------------------------- kev 导出
def _export_kev(model_id, subfolder, out_dir):
    """Qwen + LoRA + PointerHead -> 单个 ONNX 图。

    checkpoint 里只有 LoRA adapter 和 head.pt，骨干要从 adapter_config 指的 base 拉。
    LoRA 在导出前合并进基座权重，PointerHead 一起进图，于是运行时不需要 peft。
    """
    import math
    import torch
    import torch.nn as nn
    from torch.export import Dim
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local = os.path.exists(model_id)
    ckpt = model_id if local else snapshot_download(model_id)
    meta = torch.load(os.path.join(ckpt, "head.pt"), map_location="cpu", weights_only=False)
    base, lora_r = meta["base"], meta.get("lora", 16)
    print("  base %s | LoRA r=%d" % (base, lora_r), flush=True)

    tok = AutoTokenizer.from_pretrained(ckpt)
    lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32,
                                              attn_implementation="eager").model
    try:
        from peft import PeftModel
    except ImportError:
        sys.exit("kev 需要 peft： uv tool install \"edgejev[build]\"")
    lm = PeftModel.from_pretrained(lm, ckpt).merge_and_unload()
    lm.eval()
    print("  LoRA 已合并", flush=True)

    d = lm.config.hidden_size
    hsd = meta["head"]
    dp = hsd["q.weight"].shape[0]

    class KevGraph(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm = lm
            self.q = nn.Linear(d, dp)
            self.k = nn.Linear(d, dp)
            self.q.load_state_dict({"weight": hsd["q.weight"], "bias": hsd["q.bias"]})
            self.k.load_state_dict({"weight": hsd["k.weight"], "bias": hsd["k.bias"]})
            self.scale = 1.0 / math.sqrt(dp)

        def forward(self, input_ids, position_ids, attn_mask_4d,
                    decide_idx, opt_idx, opt_mask):
            h = self.lm(input_ids=input_ids, position_ids=position_ids,
                        attention_mask=attn_mask_4d, use_cache=False).last_hidden_state[0]
            qv = self.q(h[decide_idx])                    # [Q, dp]
            kv = self.k(h[opt_idx])                       # [Q, K, dp]
            logits = (kv * qv[:, None, :]).sum(-1) * self.scale
            return logits.masked_fill(~opt_mask, -1e4)

    net = KevGraph().eval()

    # 造一个示例输入：两题，选项数不同
    from .core.layout import packed_branches
    from .tokenize import Encoder
    sp = [tok.convert_tokens_to_ids(t) for t in
          ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]]
    enc = Encoder(None, {"pad_id": tok.pad_token_id or 0})
    enc.tok = tok.backend_tokenizer if hasattr(tok, "backend_tokenizer") else None
    enc.ids = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    enc.special = sp
    cfg0 = {"max_state": 384, "max_branch": 1024, "option_isolation": True}
    sample = packed_branches(enc, "a short piece of state text",
                             [{"t": "choice", "ins": "which team",
                               "crit": {"a": "one", "b": "two", "c": "three"}},
                              {"t": "noul", "ins": "is it urgent", "crit": None}], cfg0)
    args = tuple(torch.from_numpy(sample.feed[n]) for n in
                 ["input_ids", "position_ids", "attn_mask_4d", "decide_idx", "opt_idx", "opt_mask"])

    path = os.path.join(out_dir, "_fp32.onnx")
    L, Q, K = Dim("L", min=8, max=2048), Dim("Q", min=1, max=64), Dim("K", min=2, max=255)
    torch.onnx.export(
        net, args, path,
        input_names=["input_ids", "position_ids", "attn_mask_4d",
                     "decide_idx", "opt_idx", "opt_mask"],
        output_names=["logits"],
        dynamic_shapes={"input_ids": {1: L}, "position_ids": {1: L},
                        "attn_mask_4d": {2: L, 3: L},
                        "decide_idx": {0: Q}, "opt_idx": {0: Q, 1: K},
                        "opt_mask": {0: Q, 1: K}},
        opset_version=20, dynamo=True, external_data=False)
    import onnx
    m = onnx.load(path)
    del m.graph.value_info[:]
    onnx.save(m, path)

    tmp = os.path.join(out_dir, "_tok")
    tok.save_pretrained(tmp)
    conf = {"backend": "kev", "runtime": "onnx", "model_name": "kev",
            "source_model": model_id, "base_model": base,
            "max_state": 384, "max_branch": 1024, "option_isolation": True,
            "special_ids": sp, "pad_id": tok.pad_token_id or 0,
            "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}
    return path, conf, os.path.join(tmp, "tokenizer.json"), None


EXPORTERS = {"laya": _export_laya, "playjev": _write_playjev, "kev": _export_kev}


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

    # 批次无关性只对「每题一行」这类会 padding 的布局有意义；
    # packed_branches 永远是单序列，不存在同批互相影响。
    if ag.spec.layout != "per_question_row":
        return ok

    import numpy as np
    q = {"t": "noul", "ins": "is this urgent", "crit": None}
    texts = ["short one", "a considerably longer piece of state text " * 8]
    solo = []
    for t in texts:
        p_ = ag.spec.prepare(ag.enc, t, [q], ag.cfg)
        solo.append(ag.sess.run(None, p_.feed)[0][0, :2].copy())
    both = ag.spec.prepare(ag.enc, texts[0], [q], ag.cfg)   # 仅为拿到字段名
    packed = _pack_two(ag, texts, q)
    lgb = ag.sess.run(None, packed)[0]
    drift = max(float(np.abs(lgb[i, :2] - solo[i]).max()) for i in range(2))
    if drift < 1e-3:
        print("  [OK] 批次无关性（漂移 %.1e）" % drift)
    else:
        print("  [注意] 批次会影响结果：logits 漂移 %.2f。动态量化按实际张量算激活 scale，"
              "padding 一变 scale 就变。要可复现请固定 batch=1 或用 --precision fp32。" % drift)
    return ok



def _pack_two(ag, texts, q):
    """把两条不同长度的输入打进一个 batch，用于批次无关性检查。"""
    import numpy as np

    rows = [ag.spec.prepare(ag.enc, t, [q], ag.cfg).feed for t in texts]
    L = max(r["input_ids"].shape[1] for r in rows)
    K = max(r["marker_pos"].shape[1] for r in rows)
    n = len(rows)
    ids = np.full((n, L), ag.enc.pad_id, dtype=np.int64)
    att = np.zeros((n, L), dtype=np.int64)
    mp = np.zeros((n, K), dtype=np.int64)
    mm = np.zeros((n, K), dtype=bool)
    qt = np.zeros(n, dtype=np.int64)
    for i, r in enumerate(rows):
        l_ = r["input_ids"].shape[1]
        k_ = r["marker_pos"].shape[1]
        ids[i, :l_] = r["input_ids"][0]
        att[i, :l_] = r["attention_mask"][0]
        mp[i, :k_] = r["marker_pos"][0]
        mm[i, :k_] = r["marker_mask"][0]
        qt[i] = r["qtype"][0]
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mp,
            "marker_mask": mm, "qtype": qt}


def run(backend="laya", model=None, subfolder=None, out=None, precision=None,
        keep_fp32=False):
    if backend not in EXPORTERS:
        sys.exit("后端 %r 还没有构建器，目前支持：%s" % (backend, ", ".join(EXPORTERS)))
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("缺少转换依赖，请先： pip install 'edgejev[build]'")

    from . import backends as _bk
    spec = _bk.get(backend)
    if precision is None:
        precision = spec.extras.get("default_precision", "int8")
        print("  后端 %s 的默认精度：%s" % (backend, precision), flush=True)
    model = model or default_model(backend)
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
            # 超过 2GB 的图 protobuf 装不下，导出器会另写一个 .data，一并清掉
            for extra in (fp32_path + ".data", fp32_path.replace(".onnx", "") + ".data"):
                if os.path.exists(extra):
                    os.remove(extra)
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
