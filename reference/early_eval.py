# ============================================================================
# CAPTIONBERT-8192-V2 — CAPABILITY + GEOMETRY EVAL   (single standalone cell)
#
# Self-contained. Pulls the checkpoint from the hub, redefines the encoder
# inline (no trainer import), runs the capability gauges the training loop
# cannot see, and measures the baselines IN THE SAME HARNESS so the numbers
# are comparable rather than cited.
#
# WHY THIS EXISTS
#   Training reports student->consensus R@1. That is MIMICRY: how well the
#   student reproduces its target. It says nothing about whether the space
#   means anything. Capability is STS/SICK against models that never saw the
#   consensus. Keep the two on separate lines, always.
#
# WHAT IT REPORTS
#   spearman   the capability gauge (STS-B, SICK-R, STS12-16 optional)
#   self_cos   isotropy. mean-pooled BERT sits in a narrow cone (~0.57);
#              a good sentence encoder is near 0 (MiniLM ~0.02)
#   erank      participation ratio = how many directions the embedding
#              actually uses. THE KEY COLUMN. Measured 2026-07-31:
#                consensus target  28.7 / 768
#                v2 in-domain      80.5
#                v2 on STS-B       31.2   <- falls back to the target's rank
#                all-MiniLM-L6-v2 103.1   <- on the SAME sentences
#              Averaging teachers cannot create rank they do not share. If
#              v2's OOD erank stays ~30 while STS stays ~0.55, the ceiling is
#              the consensus construction, not the student.
#
# BASELINE (measured, CPU, same harness, 2026-07-31, checkpoint step 3327):
#   bert-base mean-pooled  109.5M  STS-B .4729  self_cos .570  erank  34.3
#   captionbert v2 @ 6%     58.3M  STS-B .5444  self_cos .139  erank  31.2
#   all-MiniLM-L6-v2        22.7M  STS-B .8203  self_cos .023  erank 103.1
#   (bert-base reproduced v1's published .4729073 to 7 digits -> harness valid)
#
# L4 (24GB) is plenty; it runs on CPU too, just slower.
# ============================================================================

import subprocess, sys, json, os
for _p in ("datasets", "transformers", "huggingface_hub", "scipy"):
    try:
        __import__(_p)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _p], check=False)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

REPO        = "AbstractPhil/captionbert-8192-v2"
CKPT        = "checkpoints/best_model.pt"   # or "checkpoints/model_sNNNN.pt"
TOKENIZER   = "google-bert/bert-base-uncased"
BASELINES   = ["google-bert/bert-base-uncased",
               "sentence-transformers/all-MiniLM-L6-v2"]
RUN_BASELINES = True          # False once you have them; they do not change
EXTRA_STS   = False           # STS12-16 as well as STS-B/SICK-R (slower)
MAX_LEN     = 64
BATCH       = 256
GEOM_N      = 1500            # sentences for the isotropy / erank probe

# architecture — must match config/config.json in the repo
ARCH = dict(vocab_size=30522, max_len=8192, d_model=512, n_heads=8,
            n_layers=12, d_ff=2048, output_dim=768, dropout=0.1,
            pad_token_id=0, pooling="mean")


# ══════════════════════════════════════════════════════════════════
# STUDENT (inline copy — keys must match the checkpoint exactly)
# ══════════════════════════════════════════════════════════════════

class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size=30522, max_len=8192, d_model=512, n_heads=8,
                 n_layers=12, d_ff=2048, output_dim=768, dropout=0.1,
                 pad_token_id=0, pooling="mean"):
        super().__init__()
        self.pad_token_id, self.pooling = pad_token_id, pooling
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
        x = self.encoder(x, src_key_padding_mask=kpm)
        if self.pooling == "cls":
            pooled = x[:, 0]
        else:
            m = (attention_mask.unsqueeze(-1).float() if attention_mask is not None
                 else (~kpm).unsqueeze(-1).float())
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.output_proj(pooled), dim=-1)


# ══════════════════════════════════════════════════════════════════
# GAUGES
# ══════════════════════════════════════════════════════════════════

def effective_rank(x: torch.Tensor) -> float:
    """Participation ratio of the singular spectrum: how many directions are used."""
    xc = (x - x.mean(0, keepdim=True)).double()
    s2 = torch.linalg.svdvals(xc) ** 2
    return float((s2.sum() ** 2 / (s2 ** 2).sum()).item())


def geometry(E: torch.Tensor) -> dict:
    n = min(GEOM_N, E.shape[0])
    X = E[:n]
    S = X @ X.T
    S.fill_diagonal_(0)
    return {"self_cos": float(S.sum() / (n * n - n)), "erank": effective_rank(X)}


def line(t=""):
    print("-" * 76 if not t else f"-- {t} " + "-" * max(0, 72 - len(t)))


# ══════════════════════════════════════════════════════════════════
# ENCODERS
# ══════════════════════════════════════════════════════════════════

def load_student():
    line("STUDENT")
    p = hf_hub_download(REPO, CKPT)
    sd = torch.load(p, weights_only=True, map_location="cpu")
    model = CaptionEncoder(**ARCH)
    model.load_state_dict(sd, strict=True)      # strict: a silent mismatch is worse
    model.eval().to(DEVICE)
    n = sum(q.numel() for q in model.parameters())
    print(f"   {REPO}/{CKPT}")
    print(f"   {n:,} params ({n/109_482_240:.2f}x bert-base) | strict load OK | {DEVICE}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    @torch.no_grad()
    def enc(texts):
        out = []
        for i in range(0, len(texts), BATCH):
            t = tok(list(texts[i:i + BATCH]), max_length=MAX_LEN, padding=True,
                    truncation=True, return_tensors="pt").to(DEVICE)
            out.append(model(t["input_ids"], t["attention_mask"]).float().cpu())
        return torch.cat(out)
    return enc, n


def load_baseline(name):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).eval().to(DEVICE)
    n = sum(q.numel() for q in mdl.parameters())

    @torch.no_grad()
    def enc(texts):
        out = []
        for i in range(0, len(texts), BATCH):
            t = tok(list(texts[i:i + BATCH]), max_length=MAX_LEN, padding=True,
                    truncation=True, return_tensors="pt").to(DEVICE)
            h = mdl(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)   # mean pool, as published
            out.append(F.normalize(pooled, dim=-1).float().cpu())
        return torch.cat(out)
    return enc, n, mdl


# ══════════════════════════════════════════════════════════════════
# TASKS
# ══════════════════════════════════════════════════════════════════

TASKS = [("STS-B", "mteb/stsbenchmark-sts", "test"),
         ("SICK-R", "mteb/sickr-sts", "test")]
if EXTRA_STS:
    TASKS += [(f"STS{y}", f"mteb/sts{y}-sts", "test") for y in (12, 13, 14, 15, 16)]


def load_task(path, split):
    ds = load_dataset(path, split=split)
    cols = ds.column_names
    a = "sentence1" if "sentence1" in cols else cols[0]
    b = "sentence2" if "sentence2" in cols else cols[1]
    s = "score" if "score" in cols else ("similarity_score" if "similarity_score" in cols else None)
    return list(ds[a]), list(ds[b]), np.asarray(ds[s], dtype=float)


def score(enc, a, b, gold):
    ea, eb = enc(a), enc(b)
    cos = F.cosine_similarity(ea, eb, dim=-1).numpy()
    return (float(spearmanr(cos, gold).correlation),
            float(pearsonr(cos, gold)[0]),
            torch.cat([ea, eb]))


# ══════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 76)
    print("CAPTIONBERT-8192-V2 - CAPABILITY + GEOMETRY")
    print("=" * 76)
    if DEVICE == "cuda":
        print(f"gpu={torch.cuda.get_device_name()} "
              f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")

    results = {}
    data = {}
    for name, path, split in TASKS:
        try:
            data[name] = load_task(path, split)
            print(f"   {name}: {len(data[name][2])} pairs")
        except Exception as e:
            print(f"   {name}: SKIPPED ({type(e).__name__}: {str(e)[:60]})")

    enc, n_par = load_student()
    line("STUDENT SCORES")
    results["captionbert-v2"] = {"params": n_par}
    for name in data:
        a, b, g = data[name]
        sp, pe, E = score(enc, a, b, g)
        geo = geometry(E)
        results["captionbert-v2"][name] = {"spearman": sp, "pearson": pe, **geo}
        print(f"   {name:8s} spearman {sp:.4f}  pearson {pe:.4f}  "
              f"self_cos {geo['self_cos']:+.4f}  erank {geo['erank']:.1f}/768")
    del enc
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    if RUN_BASELINES:
        for bn in BASELINES:
            line(f"BASELINE {bn}")
            benc, bn_par, mdl = load_baseline(bn)
            results[bn] = {"params": bn_par}
            for name in data:
                a, b, g = data[name]
                sp, pe, E = score(benc, a, b, g)
                geo = geometry(E)
                results[bn][name] = {"spearman": sp, "pearson": pe, **geo}
                print(f"   {name:8s} spearman {sp:.4f}  pearson {pe:.4f}  "
                      f"self_cos {geo['self_cos']:+.4f}  erank {geo['erank']:.1f}")
            del mdl, benc
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # ---- table ----
    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)
    tasks = list(data.keys())
    hdr = f"   {'model':34s}{'params':>10s}" + "".join(f"{t:>10s}" for t in tasks) \
          + f"{'self_cos':>10s}{'erank':>8s}"
    print(hdr)
    for k, v in results.items():
        row = f"   {k[-34:]:34s}{v['params']/1e6:>9.1f}M"
        for t in tasks:
            row += f"{v[t]['spearman']:>10.4f}" if t in v else f"{'-':>10s}"
        ref = tasks[0]
        row += f"{v[ref]['self_cos']:>+10.4f}{v[ref]['erank']:>8.1f}" if ref in v else ""
        print(row)

    # ---- the read ----
    print()
    line("READ")
    cb = results.get("captionbert-v2", {})
    ref = tasks[0] if tasks else None
    if ref and ref in cb:
        er = cb[ref]["erank"]
        print(f"   erank on {ref} = {er:.1f}. Consensus target measured 28.7/768;")
        print(f"   v2 in-domain (CC12M val) measured 80.5. On out-of-domain text the")
        print(f"   student falls back toward its target's intrinsic rank.")
        mini = results.get("sentence-transformers/all-MiniLM-L6-v2")
        if mini and ref in mini:
            print(f"   all-MiniLM uses {mini[ref]['erank']:.1f} directions on the SAME "
                  f"sentences at {mini['params']/1e6:.1f}M params.")
            print(f"   Averaging teachers cannot create rank they do not share -- if this")
            print(f"   gap holds, the ceiling is the CONSENSUS, not the student, and the")
            print(f"   fix is heterogeneous teachers rather than a bigger model.")
    print("   Training's student->consensus R@1 is MIMICRY. This table is capability.")

    with open("v2_capability.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n   wrote v2_capability.json")
    return results


RESULTS = main()