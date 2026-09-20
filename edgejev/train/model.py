"""训练用的模型：任意 HF 编码器 + 两层 transformer head + marker 打分头。

与 `edgejev.backends.laya` 的推理路径严格对应：同样的序列渲染、同样的 marker 位置、
同样的 scorer 读法。训练完直接能被 `edgejev build --backend laya` 导出。
"""
import torch
import torch.nn as nn


class PointerScorer(nn.Module):
    """读 marker 位置的隐状态，每个选项出一个 logit。"""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h, marker_pos, marker_mask):
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.net(m).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, dropout,
                                           batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) \
            if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = PointerScorer(d)
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, 2))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        logits = self.scorer(h, marker_pos, marker_mask)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act_logits = self.act_head(torch.cat([h[:, 0].float(), feats], -1))
        return logits, act_logits


def build(encoder_name="jhu-clsp/mmBERT-base", head_layers=2):
    from transformers import AutoModel
    enc = AutoModel.from_pretrained(encoder_name, attn_implementation="sdpa")
    try:
        enc.config.reference_compile = False   # ModernBERT 默认会 torch.compile，小 batch 上是负收益
    except Exception:
        pass
    return DecisionModel(enc, head_layers)
