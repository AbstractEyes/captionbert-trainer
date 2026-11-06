# MIT; AbstractPhil, CaptionBERT v2. Verbatim encoder from upstream revision
# ed468fe5aed64a3d03a56a7f7811131d20f81ddb; see reference/ and THIRD_PARTY.md.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

class CaptionEncoder(nn.Module):
    """Standalone caption encoder. No experts at inference. No bank."""

    def __init__(self, vocab_size=30522, max_len=8192, d_model=512, n_heads=8,
                 n_layers=12, d_ff=2048, output_dim=768, dropout=0.1,
                 pad_token_id=0, pooling="mean", grad_checkpointing=False):
        super().__init__()
        self.pad_token_id, self.pooling = pad_token_id, pooling
        self.grad_checkpointing = grad_checkpointing
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim))

    def forward(self, input_ids, attention_mask=None):
        L = input_ids.shape[1]
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.emb_drop(self.emb_norm(self.token_emb(input_ids) + self.pos_emb(pos)))
        kpm = (~attention_mask.bool()) if attention_mask is not None \
            else (input_ids == self.pad_token_id)
        if self.grad_checkpointing and self.training:
            for layer in self.encoder.layers:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, None, kpm, use_reentrant=False)
        else:
            x = self.encoder(x, src_key_padding_mask=kpm)
        if self.pooling == "cls":
            pooled = x[:, 0]
        else:
            m = (attention_mask.unsqueeze(-1).float() if attention_mask is not None
                 else (~kpm).unsqueeze(-1).float())
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.output_proj(pooled), dim=-1)
