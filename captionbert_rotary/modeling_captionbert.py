"""CaptionBERT with the final key-at-every-position rotary adaptation.

This file and configuration_captionbert.py are deliberately self-contained for
AutoModel remote-code loading. No training package is needed for inference.
"""
import copy
from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from .configuration_captionbert import CaptionBertConfig


@dataclass
class CaptionBertOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    pooler_output: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    pointer_loss: Optional[torch.Tensor] = None


def hierarchical_frequencies():
    """v0.1.103's 8x3 form verbatim in value: 2 relative, 2 content, 12 anchor pairs."""
    base = 10000.0 * 4.0 ** (4.0 / 2.0)
    w = [base ** (-j / 2) for j in range(2)] + [0.0, 0.0]
    kinds = [0, 0, 1, 1]
    for level in range(3):
        w += [2 * math.pi * k / 9 for k in range(1, 5)]
        kinds += [2 + level] * 4
    return torch.tensor(w, dtype=torch.float64), torch.tensor(kinds, dtype=torch.long)


def rotate_half(t, angle):
    pairs = t.shape[-1] // 2
    a, b = t[..., :pairs].float(), t[..., pairs:].float()
    c, s = angle.float().cos(), angle.float().sin()
    return torch.cat((a * c - b * s, a * s + b * c), dim=-1).to(t.dtype)


def hold_pairs(key, on, rho):
    pairs = key.shape[-1] // 2
    a, b = key[..., :pairs], key[..., pairs:]
    # Clamp before sqrt: the same forward floor, with finite derivatives at
    # zero/padded pairs (sqrt(0) followed by clamp can yield 0 * inf in backward).
    length = (a.float().square() + b.float().square()).clamp_min(1e-12).sqrt()
    factor = torch.where(on[..., None], rho / length, torch.ones_like(length)).to(key.dtype)
    return torch.cat((a * factor, b * factor), dim=-1)


def word_features(ids, valid, continuation, special, pairs=16, seed=1662197431):
    """Stateless whole-WordPiece-key binding; all pieces and offsets contribute.

    The source's fixed byte/offset phase table becomes a fixed 31-bit counter
    hash over WordPiece IDs and reverse offsets. Integer accumulation is exact
    and independent of scatter order. It has no key-length truncation. This is
    a declared tokenizer adaptation, not the original byte table's draw.
    """
    batch, length = ids.shape
    position = torch.arange(length, device=ids.device)[None].expand(batch, -1)
    active = valid & ~special[ids]
    starts = valid & ~continuation[ids]
    group = starts.long().cumsum(1)
    first = torch.where(starts, position, torch.zeros_like(position)).cummax(1).values
    offset = position - first
    counts = torch.zeros(batch, length + 1, device=ids.device, dtype=torch.long)
    counts.scatter_add_(1, group, active.long())
    reverse = (counts.gather(1, group) - 1 - offset).clamp_min(0)
    pair = torch.arange(1, pairs + 1, device=ids.device, dtype=torch.long)
    mask31 = 2147483647
    u = (ids[..., None].long() * 1103515245 + reverse[..., None] * 1234567891
         + pair * 1013904223 + int(seed)) & mask31
    u = ((u ^ (u >> 16)) * 73244475) & mask31
    u = ((u ^ (u >> 16)) * 73244475) & mask31
    u = (u ^ (u >> 16)) & mask31
    sums = torch.zeros(batch, length + 1, pairs, device=ids.device, dtype=torch.long)
    sums.scatter_add_(1, group[..., None].expand(-1, -1, pairs), u * active[..., None])
    keys = sums.gather(1, group[..., None].expand(-1, -1, pairs)) & mask31
    phase = keys.float() * (2 * math.pi / 2147483648.0)
    next_start = F.pad(starts[:, 1:], (0, 1), value=True)
    next_active = F.pad(active[:, 1:], (0, 1), value=False)
    ends = active & (next_start | ~next_active)

    # First occurrence's ending is the record; later occurrences are queries.
    # Two independent 31-bit components identify candidate groups. Training
    # targets are computed from exact token tuples separately, never this hash.
    identity = keys[..., 0] | (keys[..., 1] << 31)
    identity = torch.where(active, identity, torch.full_like(identity, -1))
    order = identity.argsort(dim=1, stable=True)
    sorted_id = identity.gather(1, order)
    changed = F.pad(sorted_id[:, 1:] != sorted_id[:, :-1], (1, 0), value=True)
    sorted_group = changed.long().cumsum(1)
    candidate = torch.where(ends, position, torch.full_like(position, length)).gather(1, order)
    minimum = torch.full((batch, length + 1), length, device=ids.device, dtype=torch.long)
    minimum.scatter_reduce_(1, sorted_group, candidate, reduce="amin", include_self=True)
    sorted_record = minimum.gather(1, sorted_group)
    record_at = torch.empty_like(sorted_record).scatter_(1, order, sorted_record)
    records = ends & (record_at == position)
    return phase, active, ends, records


class CaptionBertLayer(nn.TransformerEncoderLayer):
    def __init__(self, config):
        super().__init__(config.hidden_size, config.num_attention_heads,
                         config.intermediate_size, config.hidden_dropout_prob,
                         activation="gelu", batch_first=True, norm_first=True)
        self.lookup_head = config.lookup_head
        self.rho = config.key_rho

    def forward(self, x, valid, aq, ak, phase, binding_on, records,
                pointer_targets=None, bind=False, output_attentions=False):
        z = self.norm1(x)
        batch, length, width = z.shape
        heads = self.self_attn.num_heads
        qkv = F.linear(z, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = [a.reshape(batch, length, heads, width // heads).transpose(1, 2)
                   for a in qkv.chunk(3, -1)]
        qr, kr = rotate_half(q, aq), rotate_half(k, ak)
        if bind:
            h = self.lookup_head
            qa = torch.where(binding_on[..., None], phase, aq[:, 0])
            ka = torch.where(binding_on[..., None], phase, ak[:, 0])
            kh = hold_pairs(k[:, h], records, self.rho) if self.rho else k[:, h]
            qh, kh = rotate_half(q[:, h], qa), rotate_half(kh, ka)
            qr = torch.cat((qr[:, :h], qh[:, None], qr[:, h + 1:]), 1)
            kr = torch.cat((kr[:, :h], kh[:, None], kr[:, h + 1:]), 1)
        pointer_loss = x.new_zeros((), dtype=torch.float32)
        if pointer_targets is not None:
            logits = qr[:, self.lookup_head].float() @ kr[:, self.lookup_head].float().transpose(1, 2)
            logits = logits / math.sqrt(width // heads)
            logits = logits.masked_fill(~valid[:, None, :], -1e9)
            selected = pointer_targets >= 0
            nll = -logits.log_softmax(-1).gather(-1, pointer_targets.clamp_min(0)[..., None]).squeeze(-1)
            pointer_loss = (nll * selected).sum() / selected.sum().clamp_min(1)
        y = F.scaled_dot_product_attention(qr, kr, v, attn_mask=valid[:, None, None, :],
            dropout_p=self.self_attn.dropout if self.training else 0.0)
        x = x + self.dropout1(self.self_attn.out_proj(y.transpose(1, 2).reshape(batch, length, width)))
        x = x + self._ff_block(self.norm2(x))
        if output_attentions:
            logits = qr.float() @ kr.float().transpose(-1, -2) / math.sqrt(width // heads)
            attention = logits.masked_fill(~valid[:, None, None, :], -1e9).softmax(-1)
            return x, pointer_loss, attention
        return x, pointer_loss


class CaptionBertModel(PreTrainedModel):
    config_class = CaptionBertConfig
    base_model_prefix = "captionbert"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.emb_norm = nn.LayerNorm(config.hidden_size)
        self.emb_drop = nn.Dropout(config.hidden_dropout_prob)
        template = CaptionBertLayer(config)
        self.layers = nn.ModuleList([copy.deepcopy(template) for _ in range(config.num_hidden_layers)])
        self.output_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
                                        nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, config.output_dim))
        self.structured = config.rotary_mode in ("hierarchical", "key_end", "key_all")
        self.value_address = nn.Parameter(torch.zeros(3, 8, config.hidden_size)) if self.structured and config.use_value_addresses else None
        continuation = torch.zeros(config.vocab_size, dtype=torch.bool)
        continuation[config.continuation_ids] = True
        special = torch.zeros_like(continuation)
        special[[i for i in config.special_ids if i < config.vocab_size]] = True
        self.register_buffer("wordpiece_continuation", continuation)
        self.register_buffer("special_token", special)
        if self.structured:
            frequencies, kinds = hierarchical_frequencies()
        else:
            pairs = config.hidden_size // config.num_attention_heads // 2
            dim = pairs * 2
            base = 10000.0 * (4.0 ** (dim / (dim - 2)) if config.rotary_mode == "ntk" else 1.0)
            frequencies = base ** (-torch.arange(pairs, dtype=torch.float64) / pairs)
            kinds = torch.zeros(pairs, dtype=torch.long)
        # Save the small immutable rotary tables alongside the weights. Current
        # Transformers materializes non-persistent buffers with empty_like when
        # loading on meta; constructor-only initialization would then be lost.
        self.register_buffer("frequencies", frequencies.float())
        self.register_buffer("kinds", kinds)
        self.post_init()

    def _init_weights(self, module):
        # Keep PyTorch's native CaptionBERT initialization; value-address tables
        # are born zero. No HF reinitialization changes the paired controls.
        pass

    def get_input_embeddings(self):
        return self.token_emb

    def set_input_embeddings(self, value):
        self.token_emb = value

    def addresses(self, ids, valid):
        punctuation = torch.zeros_like(valid)
        for token in self.config.punctuation_ids:
            punctuation = punctuation | (ids == token)
        punctuation = punctuation & valid
        sentence = punctuation.long().cumsum(1) - punctuation.long()
        digits = torch.stack(((sentence // 64) % 8, (sentence // 8) % 8, sentence % 8), -1)
        return torch.where((valid & (sentence < 512))[..., None], digits, torch.full_like(digits, 8))

    def angles(self, ids, valid, query_addresses=None, key_addresses=None):
        batch, length = ids.shape
        position = torch.arange(length, device=ids.device, dtype=torch.float32)
        base = position[:, None] * torch.where(self.kinds == 0, self.frequencies, 0)[None]
        aq = base[None].expand(batch, -1, -1).clone()
        ak = aq.clone()
        addresses = self.addresses(ids, valid) if self.structured else None
        if self.structured:
            dq = addresses if query_addresses is None else query_addresses
            dk = addresses if key_addresses is None else key_addresses
            if dq.shape != addresses.shape or dk.shape != addresses.shape:
                raise ValueError("Address streams must have shape (batch, length, 3)")
            for level in range(3):
                selected = self.kinds == level + 2
                aq[..., selected] = (dq[..., level:level + 1].float() - 8) * self.frequencies[selected]
                ak[..., selected] = (dk[..., level:level + 1].float() - 8) * self.frequencies[selected]
            addresses = dk
        return aq[:, None], ak[:, None], addresses

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                query_addresses=None, key_addresses=None, pointer_targets=None,
                output_hidden_states=None, output_attentions=None, return_dict=None, **kwargs):
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.config.max_position_embeddings:
            raise ValueError("input_ids must be (batch, length), with length in the configured range")
        valid = input_ids.ne(self.config.pad_token_id) if attention_mask is None else attention_mask.bool()
        if valid.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        aq, ak, addresses = self.angles(input_ids, valid, query_addresses, key_addresses)
        binding = self.config.rotary_mode in ("key_end", "key_all")
        if binding:
            phase, active, ends, records = word_features(input_ids, valid, self.wordpiece_continuation,
                                                       self.special_token, seed=self.config.phase_seed)
            on = ends if self.config.rotary_mode == "key_end" else active
        else:
            phase = aq[:, 0]
            on = records = torch.zeros_like(valid)
        x = self.token_emb(input_ids)
        if self.value_address is not None:
            for level in range(3):
                d = addresses[..., level].long()
                d = torch.where((d >= 0) & (d < 8), d, torch.full_like(d, 8))
                table = torch.cat((self.value_address[level], x.new_zeros(1, x.shape[-1])), 0)
                x = x + F.embedding(d, table, padding_idx=8)
        x = self.emb_drop(self.emb_norm(x))
        states = (x,) if output_hidden_states else None
        attentions = () if output_attentions else None
        pointer_loss = x.new_zeros((), dtype=torch.float32)
        for index, layer in enumerate(self.layers):
            target = pointer_targets if index == self.config.lookup_layer else None
            if target is not None and self.config.pointer_positions == "end":
                if not binding:
                    raise ValueError("End-position supervision requires the word-binding features")
                target = torch.where(ends, target, torch.full_like(target, -1))
            args = (x, valid, aq, ak, phase, on, records)
            opts = dict(pointer_targets=target, bind=binding and index == self.config.lookup_layer,
                        output_attentions=bool(output_attentions))
            if self.training and self.gradient_checkpointing and not output_attentions:
                result = checkpoint(layer, *args, use_reentrant=False, **opts)
            else:
                result = layer(*args, **opts)
            x, term = result[:2]
            pointer_loss = pointer_loss + term
            if output_hidden_states:
                states += (x,)
            if output_attentions:
                attentions += (result[2],)
        mask = valid[..., None].float()
        pooled = (x.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
        embedding = F.normalize(self.output_proj(pooled).float(), dim=-1)
        embedding = torch.where(valid.any(1)[:, None], embedding, torch.zeros_like(embedding))
        if not return_dict:
            return tuple(v for v in (x, embedding, states, attentions, pointer_loss) if v is not None)
        return CaptionBertOutput(last_hidden_state=x, pooler_output=embedding,
                                 hidden_states=states, attentions=attentions, pointer_loss=pointer_loss)
