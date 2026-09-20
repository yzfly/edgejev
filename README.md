# EdgeJev

在自己的设备上跑 [Jev](https://typesafe.ai) 式的类型化决策：**ONNX + INT8，运行时不需要 torch，
Linux / macOS / Windows 都能跑。**

Jev 不生成文本——给它一份 state 和几个带类型的问题，它一次前向答完，返回代码能直接 `if` 的值加一个校准概率。
官方模型要排 waitlist，开源复现又各自绑定 PyTorch 和 GPU。EdgeJev 把它们统一成一条
「转换 → 量化 → 部署」的路径，跑在普通 CPU 上。

上游 [laya](https://github.com/NandhaKishorM/laya) 自己的提示写着 CPU 上 `~200-500 ms`。
EdgeJev 在一台 **4 vCPU** 的机器上单题 **15.6 ms**。

```bash
pip install 'edgejev[build]'
edgejev build --backend laya --out ./jev-int8    # 只此一步需要 torch，转完可卸载
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
r["answers"]["anger"]["confidence"]     # 0.41  —— 低置信度，这条别自动执行
```

也可以起一个官方协议的端点，官方 SDK 改个 `base_url` 就切过来：

```bash
edgejev serve --model ./jev-int8 --port 8009
export TYPESAFE_BASE_URL=http://127.0.0.1:8009
export TYPESAFE_API_KEY=local
```

## 命令

| 命令 | 作用 |
| :-- | :-- |
| `edgejev build` | 上游 checkpoint → ONNX + 量化 + 多形状自检 |
| `edgejev serve` | 起一个 Jev 兼容的 `POST /v1/systemone` |
| `edgejev eval` | 在 AG News / dair-ai emotion 上复现指标 |
| `edgejev bench` | 测延迟 |
| `edgejev info` | 打印 provider 与已注册后端 |

## 数值一致性

`--precision fp32` 与上游 laya（PyTorch）**逐位一致**：8 个问题覆盖 choice / score / noul、
中英文、字符串与 dict 状态，最大概率偏差 `0.00000`，token 计数完全相同。

## 实测

4 vCPU Intel Xeon Cascade Lake（AVX512-VNNI），`laya-multilingual`（mmBERT-base, 322M）。

**延迟**（batch=1，短输入 seq≈36，贴近实时交互场景）

| 精度 | 体积 | 单题 | 三题一次 | 每题 | 加速 |
| :-- | --: | --: | --: | --: | --: |
| fp32 | 1290 MB | 32.1 ms | 84.1 ms | 28.0 ms | 1.00x |
| **int8（默认，per-tensor）** | **324 MB** | **15.6 ms** | **44.8 ms** | **14.9 ms** | **2.06x** |
| int8-pc（per-channel） | 325 MB | 16.7 ms | 46.5 ms | 15.5 ms | 1.92x |
| mixed（决策路径保持 fp32） | 366 MB | 19.2 ms | 53.0 ms | 17.7 ms | 1.67x |
| *（对照）* uint8 per-channel | 325 MB | 27.9 ms | 83.4 ms | 27.8 ms | 1.15x |

**准确率**（AG News 4 分类 / dair-ai emotion 6 分类，各 400 条，口径对齐 classifier.dev 公布的 benchmark）

| 方案 | AG News | emotion |
| :-- | --: | --: |
| 本地 laya fp32 | **92.8%** | 54.0% |
| 本地 laya int8 | 91.0% | 50.0% |
| 托管真 Jev 1.13（Vercel AI Gateway，原生 criteria） | 85.5% | 61.5% |
| 托管真 Jev 1.13（classifier.dev，光标签） | 88.0% | **62.7%** |

复现：`edgejev eval --model ./jev-int8 --task all --n 400`

两点值得单独说：

* **laya 在 AG News 上赢真 Jev 5–7 个点，但在 emotion 上输 8–9 个点。** laya 的 README 声称
  「DAIR Emotion 0.595 对 Jev 0.480」，**在我的复现里方向是反的**（Jev 61.5% > laya 54.0%）。
  细粒度情绪分类这类任务，上线前一定要自己量。
* **提示写法不能跨模型迁移。** 同一个任务，给选项加描述对两者的影响方向**相反**：
  AG News 上 laya +1.6 / Jev −2.5；emotion 上 laya −7.2 / Jev +1.4。

## ⚠️ 动态量化会让结果依赖 batch

这是本项目过程中最值得记的一个坑：

| | 单条 vs 批量的 logits 最大差 |
| :-- | --: |
| 上游 laya（PyTorch fp32） | 1e-06 |
| EdgeJev fp32 ONNX | **0.000e+00** |
| EdgeJev int8 **动态**量化 | **2.43** |

ONNX Runtime 的动态量化在**运行时**按实际张量算激活 scale，padding 一变范围就变、scale 就变，
于是**同一条输入，跟谁同批会影响它的答案**。对普通分类器也许能忍，对一个卖校准概率、
概率要拿去写阈值门控的模型是硬伤。

要可复现的结果，目前只有两条路：

1. **固定 `batch=1`** —— `Agent.system_one` 本来就是单请求路径，所以上面公布的指标不受影响。
   自己写批量推理的才需要注意。
2. `--precision fp32` —— 逐位一致，代价是慢一倍、大四倍。

**`--precision int8-static` 目前不要用。** 它确实做到了批次无关（漂移 `0.0e+00`），
但 MinMax 标定把模型打废了，而且更慢——实测 AG News **25.8%**（随机基线 25%）、
emotion **29.8%**、延迟 **195.5 ms**（动态量化是 91.2% / 48.2% / 48.9 ms）。
选项保留着是为了继续调标定方法（Percentile / Entropy、更大更有代表性的标定集），
在调好之前它被标为实验性，CLI 会警告。

`edgejev build` 的自检会**强制检查批次无关性**并在不满足时明确警告。

## 其他踩过的坑

* **别用 QUInt8。** 同 8 bit、同体积，但 x86 的 AVX512-VNNI 只对有符号 int8 有快路径：
  实测 QUInt8 27.9 ms、QInt8 15.6 ms，白丢一半加速。*（此条为 x86 专属；ARM / Apple Silicon 走 SDOT，差距没这么大。）*
* **保留嵌入表不量化没有用。** 322M 里 196.6M 是 256k 词表的嵌入表，看着像精度损失大头，
  但保留后精度一点没回来（91.0% / 51.5%），体积却从 325 MB 涨到 915 MB。
* **fp16 在 CPU 上别想了。** 这类 x86 没有 `avx512_fp16`，ORT 的 CPU EP 会转回 fp32 算；
  而且 `onnxconverter_common` 的 fp16 pass 处理不了 dynamo 导出图里的 `_to_copy` 节点，三种转法都加载失败。
* **导出必须用 dynamo。** 旧的 `torch.onnx.export` 会把 head 里 `nn.MultiheadAttention` 的
  batch/seq 固化成导出时的形状，换个输入长度直接崩——而且拿同一个批次做数值比对时**发现不了**
  （误差 3.87e-06，看着完美）。`edgejev build` 因此强制跑多形状自检。

## 平台

| 平台 | Execution Provider |
| :-- | :-- |
| Linux / Windows x86 | CPU（AVX512-VNNI 或 AVX2 跑 int8） |
| macOS Apple Silicon | CoreML（可落 ANE / GPU），不支持的算子自动回退 CPU |
| Linux ARM | CPU（int8 走 SDOT） |

`edgejev info` 打印实际选用的 provider。用 `EDGEJEV_PROVIDER=cpu` 或 `--provider cpu` 可强制。

## 后端

不同的开源 Jev 复现，序列构造和打分头完全不同，所以做成了可插拔适配器：

| 后端 | 骨干 | 机制 | 状态 |
| :-- | :-- | :-- | :-- |
| `laya` | mmBERT-base 编码器 322M | 每题一个 batch 行，读 `[MASK]` 标记位隐状态 | ✅ 已验证逐位一致 |
| `kev` | Qwen + LoRA 0.5B–8B | 多题打包一条序列，block-causal 掩码，PointerHead 读 `<decide>` / `</opt>` | 🚧 开发中 |

加一个后端＝在 `edgejev/backends/` 写一个模块并在注册表里登记。

## 训练自己的模型

`edgejev.train` 提供一条与推理**共用同一个渲染器**的训练路径——训练看到的序列和线上请求逐 token 相同，
避免「训练/推理渲染不一致」这类最难查的 bug。

```python
from edgejev.train import load_jsonl, proper_scoring_loss, ece
```

* `losses.py` —— 严格恰当评分规则（log score + spherical，score 型额外加 ranked probability score）。
  不用普通交叉熵是因为 System One 卖的是**校准**而不是 argmax 正确：严格恰当评分规则只有「如实报告信念」
  才能取得最优期望得分。另含 ECE 与可靠性分桶。
* `data.py` —— 硬标签与软标签统一成目标分布。软标签（多人标注分歧、teacher 概率）是拿到好校准的关键：
  人都会犹豫的样本，模型也该犹豫。
* `model.py` —— 任意 HF 编码器 + 两层 transformer head + marker 打分头，训完直接能被
  `edgejev build --backend laya` 导出。

## 许可

Apache-2.0。模型权重与 tokenizer 的许可归上游 [laya](https://huggingface.co/convaiinnovations/laya-multilingual)
与 [mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base) 所有。本项目与 TypeSafe AI 无隶属关系。
