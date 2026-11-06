import math
import torch
from torch import nn


def rope_half(t, cos, sin):
    pairs = cos.shape[-1]
    a, b = t[..., :pairs].float(), t[..., pairs:].float()
    return torch.cat((a * cos - b * sin, a * sin + b * cos), -1).to(t.dtype)


def freqs_rope(n, theta=10000.0):
    return theta ** (-torch.arange(n, dtype=torch.float64) / n)


def slot_ladder(n, lo=3.0, hi=256.0):
    wavelengths = lo * (hi / lo) ** (torch.arange(n, dtype=torch.float64) / max(1, n - 1))
    return 2 * math.pi / wavelengths


def anchor_simplex(n):
    return 2 * math.pi * torch.arange(1, n + 1, dtype=torch.float64) / (2 * n + 1)


def mpar_bands(pairs, content, slot, anchor, lo=3.0, hi=256.0):
    if min(content, slot, anchor) < 0 or content + slot + anchor != pairs:
        raise ValueError("Band sizes must be nonnegative and sum to the number of pairs")
    w = torch.cat((torch.zeros(content, dtype=torch.float64),
                   slot_ladder(slot, lo, hi), anchor_simplex(anchor)))
    stream = torch.cat((torch.zeros(content + slot, dtype=torch.long),
                        torch.ones(anchor, dtype=torch.long)))
    return w, stream


class RopeMulti(nn.Module):
    def __init__(self, w, stream):
        super().__init__()
        self.register_buffer("w", w.float(), persistent=False)
        self.register_buffer("stream", stream.long(), persistent=False)

    def forward(self, q, k, pq, pk):
        with torch.autocast(device_type=q.device.type, enabled=False):
            aq = (pq.float()[..., self.stream] * self.w)[:, None]
            ak = (pk.float()[..., self.stream] * self.w)[:, None]
            return rope_half(q, aq.cos(), aq.sin()), rope_half(k, ak.cos(), ak.sin())


def bands_for_arm(arm, head_dim):
    pairs = head_dim // 2
    if arm == "A1":
        return freqs_rope(pairs), torch.zeros(pairs, dtype=torch.long)
    content = pairs // 4
    slot = 3 * pairs // 8
    if arm == "A3":
        return mpar_bands(pairs, content, pairs - content, 0)
    if arm in ("A2", "A2g"):
        return mpar_bands(pairs, content, slot, pairs - content - slot)
    raise ValueError(arm)
