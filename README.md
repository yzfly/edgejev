# EdgeJev

在本地设备上跑 [Jev](https://typesafe.ai) 式的类型化决策。ONNX + INT8，运行时不需要 torch。

Jev 不生成文本：给一份 state 和几个带类型的问题，一次前向答完，返回能直接 `if` 的值加一个校准概率。
EdgeJev 把各家开源复现统一成「转换 → 量化 → 部署」一条路径。

4 vCPU Xeon 上 `laya-multilingual`（322M）单题 **15.6 ms**，模型 324 MB。

## 安装

```bash
pip install edgejev                # 运行时：onnxruntime + tokenizers + numpy
pip install 'edgejev[build]'       # 转换时额外需要 torch / transformers
```

## 用法

```bash
edgejev build --backend laya --out ./jev-int8
```

```python
from edgejev import Agent

ag = Agent("./jev-int8")
r = ag.system_one("我的信用卡被扣了两次款，麻烦退一笔。", {
    "dept":   {"type": "choice", "instructions": "该转给哪个组？",
               "criteria": {"billing": "支付、扣款、发票、退款",
                            "technical": "程序缺陷、报错",
                            "sales": "售前咨询、定价"}},
    "urgent": {"type": "noul",  "instructions": "这条消息表达了紧急或时间压力"},
    "anger":  {"type": "score", "instructions": "客户的不满程度",
               "criteria": ["平静陈述", "有情绪但讲道理", "非常愤怒"]},
})
r["answers"]["dept"]["choice"]          # 'billing'
r["answers"]["dept"]["probabilities"]   # {'billing': 0.92, 'technical': 0.08, 'sales': 0.00}
r["answers"]["anger"]["confidence"]     # 0.41
```

官方协议端点：

```bash
edgejev serve --model ./jev-int8 --port 8009
export TYPESAFE_BASE_URL=http://127.0.0.1:8009
export TYPESAFE_API_KEY=local
```

## 命令

| 命令 | 作用 |
| :-- | :-- |
| `edgejev build` | checkpoint → ONNX → 量化 → 多形状自检 |
| `edgejev serve` | `POST /v1/systemone` |
| `edgejev eval` | AG News / dair-ai emotion 上跑指标 |
| `edgejev bench` | 测延迟 |
| `edgejev info` | provider 与已注册后端 |

## 后端

| 后端 | 骨干 | 读出方式 | 运行时 | 状态 |
| :-- | :-- | :-- | :-- | :-- |
| `laya` | mmBERT-base 322M | 每题一行，读 `[MASK]` 标记位 | ONNX | ✅ |
| `playjev` | Qwen3.5-0.8B VLM | 画面输入，读最后位置的字母槽 logits | torch | ✅ |
| `kev` | Qwen + LoRA 0.5B–8B | 多题打包，block-causal，PointerHead | ONNX | 🚧 |

`playjev` 走 torch：Qwen3.5 的文本塔是混合线性注意力（`layer_types` 含 `linear_attention`），
依赖 `causal_conv1d` / flash-linear-attention 的递归状态核，没有对应的 ONNX 算子。
用这个后端只统一 API 和 `serve`，没有量化加速。

```bash
pip install 'edgejev[vlm]'
edgejev build --backend playjev --out ./playjev      # 只落配置，不导 ONNX
```

```python
ag = Agent("./playjev")
ag.system_one("frame.png", {                          # state 传图片路径 / URL / bytes / PIL / ndarray
    "move": {"type": "choice", "instructions": "Which move should the player make next?",
             "criteria": {"left": "move the paddle left", "right": "move the paddle right",
                          "stay": "keep the paddle still"}}})
```

4 vCPU CPU 上单次决策约 1.2 s（未装 `causal_conv1d` / `flash-linear-attention` 优化核，
transformers 回退到参考实现）。返回里多一个 `allowed_mass`：全词表 softmax 落在 K 个字母槽上的质量。
这个模型的基座是 base 而非 instruct，概率质量天然分散，`allowed_mass` 在 1e-4 量级属正常，
有判别力的是字母之间的相对排序。实测五个游戏的缩略图，argmax 与分布形状各不相同，
max(p) 与上游回放记录的真实对局分布（167 步，中位 0.827）落在同一区间。

加后端＝在 `edgejev/backends/` 写一个模块并在注册表登记。

## 基准

4 vCPU Intel Xeon Cascade Lake（AVX512-VNNI），`laya-multilingual`。

延迟（batch=1，seq≈36）：

| 精度 | 体积 | 单题 | 三题 | 每题 |
| :-- | --: | --: | --: | --: |
| fp32 | 1290 MB | 32.1 ms | 84.1 ms | 28.0 ms |
| **int8**（默认，per-tensor） | **324 MB** | **15.6 ms** | **44.8 ms** | **14.9 ms** |
| int8-pc | 325 MB | 16.7 ms | 46.5 ms | 15.5 ms |
| mixed | 366 MB | 19.2 ms | 53.0 ms | 17.7 ms |
| uint8-pc | 325 MB | 27.9 ms | 83.4 ms | 27.8 ms |

准确率（各 400 条，`edgejev eval --task all --n 400`）：

| | AG News | emotion |
| :-- | --: | --: |
| laya fp32 | 92.8% | 54.0% |
| laya int8 | 91.0% | 50.0% |
| Jev 1.13（Vercel AI Gateway） | 85.5% | 61.5% |
| Jev 1.13（classifier.dev） | 88.0% | 62.7% |

`--precision fp32` 与上游 laya（PyTorch）逐位一致，最大概率偏差 `0.00000`。

## 已知问题

**动态量化的结果依赖 batch。** 激活 scale 在运行时按实际张量算，padding 变了 scale 就变：

| | 单条 vs 批量 logits 最大差 |
| :-- | --: |
| 上游 laya（PyTorch） | 1e-06 |
| EdgeJev fp32 ONNX | 0.000e+00 |
| EdgeJev int8 动态 | 2.43 |

`Agent.system_one` 是单请求路径，上面的指标不受影响。自己写批量推理的固定 `batch=1` 或用 fp32。
`edgejev build` 会检查这一项并在不满足时警告。

**`--precision int8-static` 不要用。** 批次无关（漂移 `0.0e+00`）但 MinMax 标定掉到随机水平：
AG News 25.8%（随机 25%）、emotion 29.8%、延迟 195.5 ms。保留是为了后续换 Percentile / Entropy 标定。

**别用 QUInt8。** 同 8 bit 同体积，x86 的 VNNI 只对有符号 int8 有快路径：QUInt8 27.9 ms、QInt8 15.6 ms。
ARM 走 SDOT，不适用。

**保留嵌入表不量化没有收益。** 322M 里 196.6M 是 256k 词表的嵌入表，保留后精度没变（91.0% / 51.5%），
体积从 325 MB 涨到 915 MB。

**fp16 在 x86 CPU 上没有意义。** 没有 `avx512_fp16`，ORT 的 CPU EP 转回 fp32 算；
且 `onnxconverter_common` 的 fp16 pass 处理不了 dynamo 图里的 `_to_copy` 节点，转出来加载失败。

**导出必须用 dynamo。** 旧的 `torch.onnx.export` 把 head 里 `nn.MultiheadAttention` 的 batch/seq
固化成导出时的形状，换输入长度就抛 Reshape 错，而用同一批次做数值比对时误差 3.87e-06 看不出来。
`edgejev build` 因此强制跑多形状自检。

## 平台

| 平台 | Execution Provider |
| :-- | :-- |
| Linux / Windows x86 | CPU（AVX512-VNNI / AVX2） |
| macOS Apple Silicon | CoreML，不支持的算子回退 CPU |
| Linux ARM | CPU（int8 走 SDOT） |

`edgejev info` 看实际选用的；`--provider cpu` 或 `EDGEJEV_PROVIDER=cpu` 强制。

## 训练

`edgejev.train` 与推理共用同一个渲染器，训练序列和线上请求逐 token 相同。

* `losses.py` —— 严格恰当评分规则（log score + spherical；score 型加 ranked probability score）。
  优化校准而不是 argmax 正确。另含 ECE 与可靠性分桶。
* `data.py` —— 硬标签与软标签统一成目标分布。
* `model.py` —— 任意 HF 编码器 + 两层 transformer head + marker 打分头，训完可直接被
  `edgejev build --backend laya` 导出。

## 许可

Apache-2.0。权重与 tokenizer 的许可归上游 [laya](https://huggingface.co/convaiinnovations/laya-multilingual)、
[mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base)、[PlayJev](https://huggingface.co/OmniJev/PlayJev-0.8B) 所有。
