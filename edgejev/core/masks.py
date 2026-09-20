"""注意力掩码。这是各家架构差异最大的一维。

* `bidirectional` —— 编码器骨干（laya），padding 之外全可见，模型自己处理
* `block_causal` —— 解码器骨干把多个问题打包进一条序列（kev）：
  attend(i,j) 当且仅当 j<=i 且（j 属于 state，或 i、j 同属一个问题分支）。
  这保证了「每题都能看到 state、但互相看不见」，也是 Jev「问题并行且隔离」的实现方式。
* `option_isolation` —— 在 block_causal 之上再加一层：选项 span 之间互不可见，
  只有 <decide> 能看到全部选项。配合「所有选项 span 共享同一段 position id」，
  选项顺序就不影响结果了（排列不变）。
"""
import numpy as np

OPT_NONE, OPT_DECIDE = -1, -2


def bidirectional(attention_mask: np.ndarray) -> None:
    """编码器：直接用 2D attention_mask，不需要构造 4D。"""
    return None


def block_causal(segs, opts=None, dtype=np.float32, neg=None):
    """加性 4D 掩码 [B,1,L,L]。segs[b][i] 为 0 表示 state，k 表示第 k 个问题。

    右 padding 用 -1 标记：真实 token 永远看不到 pad（pad 在其后且不属于任何段），
    pad 行保留对角线以免整行被 mask 掉导致 softmax 出 NaN。
    """
    neg = neg if neg is not None else np.finfo(dtype).min
    B = len(segs)
    L = max(len(s) for s in segs)
    s = np.full((B, L), -1, dtype=np.int64)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = seg

    causal = np.tril(np.ones((L, L), dtype=bool))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = causal[None] & same & valid_key

    if opts is not None:
        o = np.full((B, L), OPT_NONE, dtype=np.int64)
        for b, op in enumerate(opts):
            o[b, : len(op)] = op
        key_is_option = o[:, None, :] >= 0
        query_is_decide = o[:, :, None] == OPT_DECIDE
        same_option = o[:, None, :] == o[:, :, None]
        allow = allow & (~key_is_option | query_is_decide | same_option)

    allow = allow | np.eye(L, dtype=bool)[None]
    m = np.zeros((B, L, L), dtype=dtype)
    m[~allow] = neg
    return m[:, None]


BUILDERS = {"bidirectional": bidirectional, "block_causal": block_causal}
