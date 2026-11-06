import numpy as np
import torch
from torch.nn import functional as F
from .model import embed


def objective(embedding, target, temperature=0.07):
    embedding = embedding.float()
    target = F.normalize(target.float(), dim=-1)
    logits = embedding @ target.T / temperature
    labels = torch.arange(len(embedding), device=embedding.device)
    nce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    mse = F.mse_loss(embedding, target)
    return nce + mse, nce, mse


def geometry(embeddings):
    # Participation ratio of centered variance, as in the original harness.
    x = embeddings.double()
    n = len(x)
    if n < 2:
        return {"erank": None, "self_cos": None}
    centered = x - x.mean(0)
    cov = centered.T @ centered
    eigenvalues = torch.linalg.eigvalsh(cov).clamp_min(0)
    denominator = eigenvalues.square().sum()
    rank = float(eigenvalues.sum().square() / denominator) if denominator > 0 else 0.0
    # Exact mean off-diagonal cosine without allocating N x N.
    unit = F.normalize(x, dim=-1)
    self_cos = (unit.sum(0).square().sum() - unit.square().sum()) / (n * (n - 1))
    return {"erank": rank, "self_cos": float(self_cos)}


def target_metrics(embeddings, targets):
    return {"n": len(embeddings), **geometry(embeddings),
            "cos_to_target": float(F.cosine_similarity(embeddings.float(), targets.float(), dim=-1).mean())}


@torch.no_grad()
def evaluate(model, store, cfg):
    was_training = model.training
    model.eval()
    # Sample once deterministically, then ensure a separate >128-token gauge is populated.
    rng = np.random.default_rng(0)
    count = min(len(store), cfg.eval_rows)
    indices = rng.choice(len(store), size=count, replace=False)
    long_indices = np.flatnonzero(store.lengths > 128)
    long_indices = rng.choice(long_indices, size=min(len(long_indices), cfg.eval_rows), replace=False)

    def collect(ix):
        embeddings, targets = [], []
        for i in range(0, len(ix), cfg.eval_batch):
            ids, mask, pos, tg = store.batch(ix[i:i + cfg.eval_batch])
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.device.startswith("cuda")):
                value = embed(model, ids.to(cfg.device), mask.to(cfg.device), pos.to(cfg.device))
            embeddings.append(value.float().cpu())
            targets.append(tg.float())
        return target_metrics(torch.cat(embeddings), torch.cat(targets)) if embeddings else {"n": 0}

    try:
        return {"heldout": collect(indices), "above_128_tokens": collect(long_indices),
                "heldout_total_rows": len(store), "above_128_total_rows": len(np.flatnonzero(store.lengths > 128))}
    finally:
        model.train(was_training)
