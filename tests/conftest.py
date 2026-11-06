import numpy as np
import pytest
import torch
from captionbert_rotary.config import ModelConfig


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def tiny_config():
    return ModelConfig(vocab_size=64, max_len=256, d_model=32, n_heads=2,
                       n_layers=2, d_ff=64, output_dim=16, dropout=0.1)


class FakeStore:
    def __init__(self, n=16):
        generator = torch.Generator().manual_seed(123)
        self.ids = torch.randint(1, 64, (n, 9), generator=generator)
        self.targets = torch.randn(n, 16, generator=generator)
        self.lengths = np.full(n, 9, dtype=np.int32)

    def __len__(self):
        return len(self.ids)

    def batch(self, ix):
        ids = self.ids[ix]
        pos = torch.zeros(*ids.shape, 2, dtype=torch.long)
        pos[..., 0] = torch.arange(ids.shape[1])
        return ids, torch.ones_like(ids), pos, self.targets[ix]
