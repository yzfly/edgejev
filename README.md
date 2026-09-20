<div align="center">

# EdgeJev

### 在本地跑类型化决策模型

把开源 Jev 复现转成 ONNX，量化，部署到 CPU

[![PyPI](https://img.shields.io/pypi/v/edgejev.svg)](https://pypi.org/project/edgejev/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

</div>

---

EdgeJev 把 [laya](https://github.com/NandhaKishorM/laya)、[kev](https://github.com/jaredpalmer/kev)、[PlayJev](https://github.com/OmniJev/PlayJev) 这些开源 Jev 复现统一成一条「转换 → 量化 → 部署」的路径。跑在普通 CPU 上，一台 4 vCPU 的机器单题 15.6 ms；运行时只要 onnxruntime、tokenizers、numpy 三个包，不装 torch。

## Quick start

```bash
uv tool install "edgejev[build]"
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

转换那一步需要 torch，转完就可以卸掉。之后只装 `edgejev` 即可运行。

## Model zoo

4 vCPU Intel Xeon Cascade Lake（AVX512-VNNI）。准确率由 `edgejev eval --task all --n 400` 跑出，
AG News 4 分类、dair-ai emotion 6 分类，各 400 条，batch=1。

| 后端 | 模型 | 精度 | 体积 | 单题 | 三题 | AG News | emotion | 构建命令 |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | :-- |
| laya | mmBERT-base 322M | fp32 | 1290 MB | 32.1 ms | 84.1 ms | 92.8% | 54.0% | `--precision fp32` |
| laya | mmBERT-base 322M | int8 | 324 MB | 15.6 ms | 44.8 ms | 91.2% | 48.2% | 默认 |
| playjev | Qwen3.5-0.8B VLM | fp32 | 2214 MB | 1.2 s | — | — | — | `--backend playjev` |
| kev | Qwen + LoRA 0.5B–8B | — | — | — | — | — | — | 导出器开发中 |

同一份权重的横向参照：laya 自己公布的是 T4 GPU 上 32.8 ms、CPU 上 200–500 ms；
官方 Jev 1.13 托管 API 实测中位 314 ms（含网络往返）。

官方 API 在同样 400 条上的成绩：AG News 85.5%、emotion 61.5%（Vercel AI Gateway，原生 criteria）；
88.0% / 62.7%（classifier.dev，光标签）。laya 在 AG News 上高 5–7 点，emotion 上低 8–11 点。
laya 的 README 声称 DAIR Emotion 0.595 对 Jev 0.480，这个方向在上表里没有复现出来。

## 与上游的一致性

`--precision fp32` 与上游 laya（PyTorch）逐位一致。对拍覆盖 choice / score / noul 三种原语、
中英文、字符串与 dict 状态、带描述与不带描述的 criteria：

| 项 | 偏差 |
| :-- | --: |
| 概率 | 0.00000 |
| 置信度 | 0.000000 |
| input_tokens | 完全相同 |

playjev 后端没有上游数值可对，改用分布核对：五个游戏画面给出不同的 argmax 与分布形状，
max(p) 落在上游公开回放的区间内（167 步真实对局，最小 0.604 / 中位 0.827 / 最大 1.000）。

## 怎么选精度

下表由 `quant_dataset.py` 跑出，两个任务都用带描述的标签、batch=16，所以数值与上面的 zoo 表
不能直接比较，但档位之间可以横向比：

| 精度 | 体积 | 单题 | AG News | emotion | 说明 |
| :-- | --: | --: | --: | --: | :-- |
| fp32 | 1290 MB | 32.1 ms | 92.8% | 47.0% | 与上游逐位一致 |
| int8 per-tensor | 324 MB | 15.6 ms | 90.5% | 52.2% | 默认 |
| int8 per-channel | 325 MB | 16.7 ms | 90.2% | 53.0% | `--precision int8-pc` |
| int8 pc + reduce_range | 325 MB | 16.5 ms | 91.0% | 54.2% | 两项都接近最好 |
| int8 混合（决策路径 fp32） | 366 MB | 19.2 ms | 90.2% | 52.8% | `--precision mixed` |
| uint8 per-channel | 325 MB | 27.9 ms | 91.0% | 48.5% | 见下 |
| int8 但嵌入表保持 fp32 | 915 MB | 23.9 ms | 91.0% | 51.5% | 见下 |
| int8-static | 326 MB | 195 ms | 25.8% | 29.8% | 不可用，见下 |

几条实测结论：

**动态量化的结果依赖 batch。** 激活的量化 scale 在运行时按实际张量计算，padding 一变 scale 就变。
同一条输入单独跑和跟别人一批跑，logits 最大差 2.43；fp32 ONNX 同样对比是 0.000。
`Agent.system_one` 是单请求路径，zoo 表的数字不受影响；自己写批量推理的固定 batch=1 或用 fp32。
`edgejev build` 会检查这一项并在不满足时警告。

**int8-static 目前不能用。** 它做到了批次无关（漂移 0.0e+00），但 MinMax 标定下精度掉到随机水平，
而且比动态量化慢约 4 倍。入口保留着，等换 Percentile 或 Entropy 标定。

**QUInt8 没有理由选。** 同样 8 bit、同样体积，x86 的 AVX512-VNNI 只对有符号 int8 有快路径。
ARM 走 SDOT，这一条不适用。

**把嵌入表排除在量化之外没有收益。** 322M 参数里 196.6M 是 256k 词表的嵌入表，看上去像精度损失的大头，
但保留它精度并不回升，体积从 324 MB 涨到 915 MB。

**fp16 在 x86 CPU 上没有意义。** 没有 avx512_fp16，ONNX Runtime 的 CPU EP 会转回 fp32 计算；
另外 onnxconverter_common 的 fp16 pass 处理不了 dynamo 导出图里的 `_to_copy` 节点，转出来加载失败。

## 起一个官方协议的服务

```bash
edgejev serve --model ./jev-int8 --port 8009
export TYPESAFE_BASE_URL=http://127.0.0.1:8009
export TYPESAFE_API_KEY=local
```

官方 SDK 改一个 `base_url` 就切过来，可以先用官方 API 把代码写完再换本地。

## CLI

| 命令 | 作用 |
| :-- | :-- |
| `edgejev build` | checkpoint → ONNX → 量化 → 多形状自检 |
| `edgejev serve` | `POST /v1/systemone` |
| `edgejev eval` | AG News / emotion 上跑指标 |
| `edgejev bench` | 测延迟 |
| `edgejev info` | provider 与已注册后端 |

## 后端

开源 Jev 复现的序列构造、注意力、读出方式各不相同。EdgeJev 把共性放进 `edgejev/core/`，
一个后端只声明四个维度：

```python
SPEC = BackendSpec(
    name="kev",
    layout="packed_branches",
    attention="block_causal",
    readout="model_logits",
    runtime="onnx",
)
```

| 后端 | 布局 | 注意力 | 读出 | 运行时 |
| :-- | :-- | :-- | :-- | :-- |
| laya | 每题一行，`[MASK]` 标记位 | bidirectional | 打分头随模型进图 | ONNX |
| kev | 多题打包一条序列 | block-causal + 选项隔离 | PointerHead | ONNX |
| playjev | 画面 + 字母清单 | causal | 词表字母槽 | torch |

渲染、掩码、温度标定、置信度、答案组装都在 core 共用。`laya.py` 17 行、`kev.py` 20 行、`playjev.py` 32 行。

playjev 走 torch 而不是 ONNX：Qwen3.5 的文本塔是混合线性注意力，`linear_attention` 层依赖
causal_conv1d 和 flash-linear-attention 的递归状态核，没有对应的 ONNX 算子。

## 安装

```bash
uv tool install edgejev              # 全局命令
uv add edgejev                       # 或加进项目，跑的时候 uv run edgejev
uvx --from edgejev edgejev info      # 或临时跑一次
```

| extra | 什么时候需要 |
| :-- | :-- |
| `edgejev[build]` | 跑 `edgejev build`，需要 torch / transformers / laya |
| `edgejev[vlm]` | playjev 后端，需要 torch / torchvision / pillow |
| `edgejev[train]` | `edgejev.train` 训练模块 |

## 平台

| 平台 | Execution Provider |
| :-- | :-- |
| Linux / Windows x86 | CPU（AVX512-VNNI / AVX2） |
| macOS Apple Silicon | CoreML，不支持的算子回退 CPU |
| Linux ARM | CPU（int8 走 SDOT） |

`edgejev info` 看实际选用的，`--provider cpu` 或 `EDGEJEV_PROVIDER=cpu` 强制。

## 训练

`edgejev.train` 与推理共用同一个渲染器，训练序列和线上请求逐 token 相同。

- `losses.py` — 严格恰当评分规则（log score + spherical，score 型加 ranked probability score），
  优化概率校准而不是 argmax 正确。另含 ECE 与可靠性分桶。
- `data.py` — 硬标签与软标签统一成目标分布。
- `model.py` — 任意 HF 编码器 + 两层 transformer head + 标记位打分头，训完可被 `edgejev build --backend laya` 导出。

## Acknowledgements

- [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) — laya 后端的模型与渲染逻辑
- [jaredpalmer/kev](https://github.com/jaredpalmer/kev) — kev 的打包布局、block-causal 掩码与 PointerHead
- [OmniJev/PlayJev](https://github.com/OmniJev/PlayJev) — playjev 的提示格式与字母槽读出
- [jhu-clsp/mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base) — laya 的编码器骨干
- [TypeSafe AI](https://typesafe.ai) — Jev 与 `/v1/systemone` 协议

## License

Apache-2.0。权重与 tokenizer 的许可归各自上游所有。本项目与 TypeSafe AI 无隶属关系。
