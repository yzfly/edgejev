# EdgeJev

**在自己的机器上跑类型化决策模型。4 核 CPU 单题 15.6 ms，比官方托管 API 快 20 倍。**

[![PyPI](https://img.shields.io/pypi/v/edgejev?style=flat-square)](https://pypi.org/project/edgejev/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](LICENSE)

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
r["answers"]["anger"]["confidence"]     # 0.41 —— 置信度低，这条不要自动执行
```

## 为什么需要它

[Jev](https://typesafe.ai) 这类 System One 模型不生成文本。给一份 state 和几个带类型的问题，
一次前向答完，返回能直接 `if` 的值加一个校准概率。

它的用武之地是**高频、低延迟、结果要落进代码分支**的地方：Agent 每一步选哪个工具、
语音对话里要不要打断、请求入口的路由和内容护栏。这些场景的共同点是——**决策层的延迟预算只有几十毫秒**，
而且往往一轮要问好几个问题。

在这个预算下，现成的三条路都不够用：

- **官方托管 API** 每次决策一个网络往返。实测中位 **314 ms**，光这一项就吃掉整个预算。还要排 waitlist、绑信用卡，数据出内网。
- **开源复现**（laya、kev 等）权重是开放的，但都绑着 PyTorch，公布的延迟按 GPU 标。laya 自己的提示里写着 CPU 上 `~200-500 ms`。
- **自己转 ONNX** 听着简单，实际每家的序列构造、注意力掩码、读出方式都不一样，量化还有一串反直觉的坑（见[限制](#限制)）。

EdgeJev 把这条路铺平：**一条命令转换 + 量化，一条命令起服务，运行时不依赖 torch。**

## 快在哪

速度不来自更聪明的模型——跑的是同一份权重。省掉的是两样东西：

**网络往返。** 官方 API 实测中位 314 ms，真正的推理只占其中一小部分，其余是链路。本地跑直接归零。

**框架与精度开销。** PyTorch → ONNX Runtime 的图优化拿到 1.7x；再上 INT8（走 x86 的 AVX512-VNNI 整数乘加）又拿到 1.2x。叠起来 **2.06x**，模型同时从 1290 MB 缩到 324 MB。

结果是决策层从 314 ms 压到 **15.6 ms**——这才塞得进语音对话那种 50–150 ms 的预算，也才谈得上「一轮问四个问题」。

| | EdgeJev（本地 int8） | 原项目 laya（PyTorch） | 官方 Jev 1.13 API |
| :-- | :-- | :-- | :-- |
| **单题延迟** | **15.6 ms** | 32.8 ms（T4 GPU）<br>200–500 ms（自述 CPU） | **314 ms**（含网络） |
| 三题一次请求 | **44.8 ms** | — | 314 ms（多题可并行） |
| 模型体积 | **324 MB** | 1290 MB | — |
| 运行时依赖 | onnxruntime + tokenizers + numpy | torch + transformers | HTTP |
| 每次调用成本 | **0** | 0 | $0.042 / MTok 输入 |
| 准入门槛 | 无 | 无 | waitlist / 绑卡 |
| 数据位置 | 本机 | 本机 | 出内网 |
| 离线可用 | ✅ | ✅ | ❌ |

## 准确率

AG News（4 分类）与 dair-ai emotion（6 分类），各 400 条，`edgejev eval` 可复现：

| | AG News | emotion |
| :-- | --: | --: |
| EdgeJev fp32 | **92.8%** | 54.0% |
| EdgeJev int8 | 91.0% | 50.0% |
| 官方 Jev 1.13（Vercel AI Gateway） | 85.5% | **61.5%** |
| 官方 Jev 1.13（classifier.dev） | 88.0% | 62.7% |

两件需要摆明的事：

**精度损失来自量化，不来自转换。** `--precision fp32` 与上游 PyTorch **逐位一致**（最大概率偏差 `0.00000`，
覆盖三种原语、中英文、字符串与 dict 状态）。INT8 的代价是 AG News −1.8 点、emotion −4 点。要精度就用 fp32，
它依然比官方 API 快一个数量级。

**开源模型不是全面胜过官方。** laya 在 AG News 上高 5–7 点，但在 emotion 上低 8–11 点。
laya 的 README 声称「DAIR Emotion 0.595 对 Jev 0.480」——各 400 条复现下来方向相反。
细粒度情绪分类这类任务，选型前必须在自己的数据上量一遍。

## 安装

用 [uv](https://docs.astral.sh/uv/)：

```bash
uv tool install edgejev              # 装成全局命令
uv add edgejev                       # 或加进当前项目（跑的时候 uv run edgejev）
uvx --from edgejev edgejev info      # 或临时跑一次，不装
```

不加 extra 时运行时只有 onnxruntime + tokenizers + numpy。按需加：

```bash
uv tool install "edgejev[build]"     # edgejev build 转 ONNX：torch / transformers / laya
uv tool install "edgejev[vlm]"       # playjev 视觉后端：torch / torchvision / pillow
uv add "edgejev[train]"              # 训练模块
```

没装 uv：`curl -LsSf https://astral.sh/uv/install.sh | sh`

## 上手

```bash
uv tool install "edgejev[build]"
edgejev build --backend laya --out ./jev-int8    # 转换 + 量化 + 多形状自检
```

然后就不再需要 torch 了，直接 `Agent("./jev-int8")`（见开头的例子），或者起一个官方协议的端点：

```bash
edgejev serve --model ./jev-int8 --port 8009
export TYPESAFE_BASE_URL=http://127.0.0.1:8009
export TYPESAFE_API_KEY=local
```

官方 SDK 改个 `base_url` 就切过来了——**先用官方 API 把代码写完，再无痛换本地**。

| 命令 | 作用 |
| :-- | :-- |
| `edgejev build` | checkpoint → ONNX → 量化 → 多形状自检 |
| `edgejev serve` | `POST /v1/systemone`，官方协议兼容 |
| `edgejev eval` | AG News / emotion 上跑指标 |
| `edgejev bench` | 测延迟 |
| `edgejev info` | provider 与已注册后端 |

## 后端

开源 Jev 复现有十几家，序列构造、注意力、读出方式各不相同。EdgeJev 没有为每家抄一份适配器，
而是把它们的共性抽成四个维度，一个后端就是一份声明：

```python
SPEC = BackendSpec(
    name="kev",
    layout="packed_branches",      # 多题打包一条序列
    attention="block_causal",      # 问题之间互相看不见
    readout="model_logits",        # 打分头随模型导进 ONNX
    runtime="onnx",
)
```

| 后端 | 骨干 | 布局 | 读出 | 运行时 |
| :-- | :-- | :-- | :-- | :-- |
| `laya` | mmBERT-base 322M | 每题一行，`[MASK]` 标记位 | 打分头 | ONNX |
| `kev` | Qwen + LoRA 0.5B–8B | 多题打包，block-causal | PointerHead | ONNX |
| `playjev` | Qwen3.5-0.8B VLM | 画面 + 字母清单 | 词表字母槽 | torch |

渲染带类型的问题、构造掩码、温度标定、置信度、组装答案都在 `edgejev/core/` 里共用，
一处修复全体受益。加一个后端通常只要填这张表。

`playjev` 走 torch 而不是 ONNX：Qwen3.5 的文本塔是混合线性注意力，`linear_attention`
层依赖 causal_conv1d / flash-linear-attention 的递归状态核，没有对应的 ONNX 算子。
用它只统一 API 和 `serve`，拿不到量化加速。

## 限制

**动态量化的结果依赖 batch。** 激活的量化 scale 在运行时按实际张量算，padding 变了 scale 就变，
同一条输入跟谁同批会影响它的答案（实测 logits 最大差 2.43；fp32 是 0.000）。
`Agent.system_one` 是单请求路径，上面的指标不受影响；**自己写批量推理的固定 `batch=1` 或用 fp32**。
`edgejev build` 会检查这一项并在不满足时警告。

**`--precision int8-static` 目前不可用。** 它能做到批次无关，但 MinMax 标定下精度掉到随机水平
（AG News 25.8%，随机基线 25%），且比动态量化慢 4 倍。保留入口是为了后续换 Percentile / Entropy 标定。

**不要用 QUInt8。** 同样 8 bit、同样体积，但 x86 的 VNNI 只对有符号 int8 有快路径：
QUInt8 27.9 ms vs QInt8 15.6 ms。ARM 走 SDOT，不适用此条。

**fp16 在 x86 CPU 上没有意义。** 没有 `avx512_fp16`，ONNX Runtime 的 CPU EP 会转回 fp32 算。

**保留嵌入表不量化不划算。** 322M 里 196.6M 是 256k 词表的嵌入表，看着像精度损失的大头，
但保留它精度并不回升（91.0% / 51.5%），体积却从 324 MB 涨到 915 MB。

## 平台

| 平台 | Execution Provider |
| :-- | :-- |
| Linux / Windows x86 | CPU（AVX512-VNNI / AVX2） |
| macOS Apple Silicon | CoreML，不支持的算子回退 CPU |
| Linux ARM | CPU（int8 走 SDOT） |

`edgejev info` 看实际选用的；`--provider cpu` 或 `EDGEJEV_PROVIDER=cpu` 强制。

## 训练自己的模型

`edgejev.train` 与推理共用同一个渲染器，训练序列和线上请求逐 token 相同。

- `losses.py` —— 严格恰当评分规则（log score + spherical，score 型加 ranked probability score）。
  优化的是概率校准而不是 argmax 正确：只有如实报告信念才能取得最优期望得分。另含 ECE 与可靠性分桶。
- `data.py` —— 硬标签与软标签统一成目标分布。软标签（多人标注分歧、teacher 概率）是拿到好校准的关键。
- `model.py` —— 任意 HF 编码器 + 两层 transformer head + 标记位打分头，训完直接能被
  `edgejev build --backend laya` 导出。

## 许可

Apache-2.0。权重与 tokenizer 的许可归上游 [laya](https://huggingface.co/convaiinnovations/laya-multilingual)、
[mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base)、[kev](https://github.com/jaredpalmer/kev)、
[PlayJev](https://huggingface.co/OmniJev/PlayJev-0.8B) 所有。本项目与 TypeSafe AI 无隶属关系。
