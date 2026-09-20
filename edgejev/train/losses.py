"""严格恰当评分规则（strictly proper scoring rules）。

为什么不用普通交叉熵：System One 模型卖的是**校准的概率**，不是 argmax 正确。
严格恰当评分规则的性质是「如实报告自己的信念」才能取得最优期望得分，
所以它直接优化校准，而不是只优化排序。

log score + spherical score 是通用项；ranked probability score 只对 score 型问题加，
因为 score 的档位之间有序（把「非常愤怒」预测成「有情绪」比预测成「平静」错得轻）。
"""
import numpy as np

try:
    import torch
except ImportError:      # 运行时不需要 torch，只有训练才需要
    torch = None


def proper_scoring_loss(logits, target, qtype, mask,
                        w_spherical=0.5, w_rps=1.0, log_floor=-9.21):
    """logits [B,K]，target [B,K]（one-hot 或软标签），qtype [B]（0choice/1score/2noul），mask [B,K]。

    返回标量 loss（= -reward，可直接 backward）。
    """
    if torch is None:
        raise ImportError("训练需要 torch：pip install 'edgejev[train]'")
    logits = logits.masked_fill(~mask, -1e4)
    q = torch.softmax(logits, -1) * mask

    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    spherical = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    reward = log_score + w_spherical * spherical

    is_score = (qtype == 1).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        rps = (((torch.cumsum(q, -1) - torch.cumsum(target, -1)) ** 2) * mask).sum(-1) / (k - 1)
        reward = reward - w_rps * rps * is_score
    return -reward.mean()


def ece(confidences, correct, bins=15):
    """Expected Calibration Error。上线前用它看「说 90% 的时候是不是真有 90% 对」。"""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - corr[sel].mean())
    return float(e)


def reliability_table(confidences, correct, bins=10):
    """返回每个置信度分桶的 (区间, 样本数, 平均置信度, 实际正确率)，用来画可靠性图。"""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            rows.append(("(%.1f, %.1f]" % (lo, hi), int(sel.sum()),
                         float(conf[sel].mean()), float(corr[sel].mean())))
    return rows
