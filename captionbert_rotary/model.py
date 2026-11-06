import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ._v2 import CaptionEncoder as V2Encoder
from .config import ModelConfig, ARMS
from .rotary import RopeMulti, bands_for_arm


class RotaryLayer(nn.TransformerEncoderLayer):
    def __init__(self, cfg, arm):
        super().__init__(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                         activation="gelu", batch_first=True, norm_first=True)
        self.rope = RopeMulti(*bands_for_arm(arm, cfg.d_model // cfg.n_heads))

    def attention(self, x, valid, positions):
        batch, length, width = x.shape
        heads = self.self_attn.num_heads
        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = [t.reshape(batch, length, heads, width // heads).transpose(1, 2)
                   for t in qkv.chunk(3, -1)]
        q, k = self.rope(q, k, positions, positions)
        # SDPA True = allowed; Transformer key padding True = masked.
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=valid[:, None, None, :].bool(),
            dropout_p=self.self_attn.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(batch, length, width)
        return self.self_attn.out_proj(y)

    def forward(self, x, valid, positions):
        x = x + self.dropout1(self.attention(self.norm1(x), valid, positions))
        x = x + self._ff_block(self.norm2(x))
        return x


class RotaryEncoder(V2Encoder):
    def __init__(self, cfg, arm):
        super().__init__(**vars(cfg))
        self.arm = arm
        del self.pos_emb
        template = RotaryLayer(cfg, arm)
        template.load_state_dict(self.encoder.layers[0].state_dict(), strict=True)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([copy.deepcopy(template) for _ in range(cfg.n_layers)])

    def forward(self, input_ids, attention_mask=None, positions=None):
        valid = input_ids.ne(self.pad_token_id) if attention_mask is None else attention_mask.bool()
        if positions is None:
            if self.arm != "A1":
                raise ValueError("Sentence arms require tokenized structural positions")
            index = torch.arange(input_ids.shape[1], device=input_ids.device)
            positions = index[None, :, None].expand(input_ids.shape[0], -1, -1)
        if self.arm in ("A1", "A2g"):
            positions = positions.clone()
            positions[..., 0] = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.emb_drop(self.emb_norm(self.token_emb(input_ids)))
        for layer in self.encoder.layers:
            if self.training and self.grad_checkpointing:
                x = checkpoint(layer, x, valid, positions, use_reentrant=False)
            else:
                x = layer(x, valid, positions)
        mask = valid[..., None].float()
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(self.output_proj(pooled), dim=-1)


def build_model(arm, cfg=None):
    cfg = cfg or ModelConfig()
    if arm not in ARMS:
        raise ValueError(arm)
    return V2Encoder(**vars(cfg)) if arm == "A0" else RotaryEncoder(cfg, arm)


def embed(model, ids, mask, positions):
    if getattr(getattr(model, "config", None), "model_type", None) == "captionbert_structured_rotary":
        return model(input_ids=ids, attention_mask=mask).pooler_output
    if isinstance(model, RotaryEncoder):
        return model(ids, mask, positions)
    return model(ids, mask)
