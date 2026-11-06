# ============================================================================
# CAPTIONBERT-8192-v2 — CONSENSUS DISTILLATION AT CC12M SCALE
#
# Once ModernBert's teacher captions are repaired the system will have the full 36m.
#
# v2 vs the shipped 500k model, per Phil's 2026-07-31 guidance:
#   - NO ALIGNMENT BANK. v1's bank was additive and experimental; measured on real
#     embeddings its expert-consistency block varied 0.2% across samples and took
#     0.23% of geo_proj energy while anchor distances took 98.7%. Banks in this
#     format are content extensions — an AMOE-LORA is the right carrier, attached
#     as a separate finetune pass on the prefitted core. Not here.
#   - LEGROOM. d 384->512, 6L->12L, ff 1536->2048, heads 6->8. 26.0M -> 58.3M
#     (0.53x bert-base, so the compression story survives). Sized for many
#     overlapping sources at ~36M features/teacher, not one 500k census.
#   - CHAMPION OBJECTIVE. InfoNCE + per-sample MSE against the consensus — the
#     consensus_nce_mse form that won the CC12M vision matrix on every task gauge,
#     both seeds. NO shipped rotation needed here: that line aligns to a running
#     mean (frame free), this one aligns to a REFERENCE MEMBER (bert), so the frame
#     is pinned by construction. A frame-fit gauge runs anyway to confirm it.
#   - CULL-PROOF. Colab kills the VM every 24h and takes local disk with it.
#     Full state (model/opt/sched/scaler/step/epoch/chunk-order/RNG) checkpoints on
#     a TIME cadence, and pushes to HF so a cull costs minutes, not the run.
#   - FULL TENSORBOARD. per-step losses + lr + grad-norm, per-eval gauges
#     (mimicry, cos, isotropy, effective rank, CV), histograms, and the alignment
#     report as text.
#
# STAGES (each resumable, each gated) — carried from the cc12m pipeline:
#   0 PARITY   which caption field was embedded + row alignment. Hard gate.
#   1 FIT      one global whitened-Procrustes map per expert -> bert, stratified
#              random fit, reported OUT-OF-SAMPLE on held-out chunks.
#   2 TARGETS  per-chunk consensus -> fp16, ledgered, expert shards deleted after.
#   3 TRAIN    streams (captions, consensus) pairs, dynamic padding.
#
# Colab-cell-safe. HF_TOKEN from Colab secrets (key icon) or env.
# ============================================================================

import gc, json, math, os, random, sys, time, subprocess, shutil
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

for _p in ("datasets", "transformers", "huggingface_hub", "tensorboard", "safetensors"):
    try:
        __import__(_p)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _p], check=False)

# Variable-length batches fragment the caching allocator badly; this is the
# documented mitigation and must be set BEFORE torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, HfApi, create_repo
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════
# BASE CONFIG
# ══════════════════════════════════════════════════════════════════

@dataclass
class BaseConfig:
    run_name: str = "captionbert-8192-v2"

    # ── sources ── (list so overlapping datasets can be added later)
    sources: Tuple[Dict[str, Any], ...] = (
        {"repo": "AbstractPhil/conceptual-captions-12m-webdataset-berts",
         "n_chunks": 66, "chunk_rows": 500_000,
         "missing": {"modern": (5, 7, 8, 21, 25, 26, 28, 32, 38, 46)}},
    )
    experts: Tuple[str, ...] = ("bert", "modern", "roberta", "albert", "distil")
    ref_expert: str = "bert"
    ref_hf_name: str = "google-bert/bert-base-uncased"
    require_all_experts: bool = True
    caption_field: Optional[str] = None
    caption_field_candidates: Tuple[str, ...] = (
        "caption_llava", "caption", "caption_llava_short")

    work_dir: str = "/content/cbv2"
    keep_expert_shards: bool = False

    # ── hardware allowance (Colab Pro+ / RTX 6000 Pro, measured 2026-07-31) ──
    # disk 235.7GB (~176 free) | RAM 176.9GB | GPU 95.6GB | 401.5 units @ 8.9/h = 45.1h
    # The expert shards are 507GB — 2.1x the WHOLE DISK. They are streamed one chunk
    # at a time and deleted; only the 43GB consensus is kept.
    disk_floor_gb: float = 25.0               # abort a chunk if free disk drops below
    ram_resident: bool = True                 # hold tokens+targets in RAM (48.8GB)
    preflight: bool = True

    # ── backup (Colab culls at 24h; local disk dies with the VM) ──
    hf_repo: str = "AbstractPhil/captionbert-8192-v2"
    targets_repo: str = "AbstractPhil/captionbert-8192-v2-consensus"
    push_targets: bool = True                 # 43GB; re-derivable only from a 507GB pull
    hf_push: bool = True
    push_every_min: float = 30.0
    keep_local_ckpts: int = 3

    # ── stage 0 ──
    parity_chunk: int = 0
    parity_n: int = 64
    parity_min_cos: float = 0.999

    # ── stage 1 ──
    fit_chunks: Tuple[int, ...] = (0, 11, 22, 33, 44, 55)
    fit_rows_per_chunk: int = 4000            # 24k vs d=768 -> N/d = 31
    holdout_chunks: Tuple[int, ...] = (60, 61)
    fit_seed: int = 0

    # ── student (LEGROOM) ──
    d_model: int = 512                        # was 384
    n_heads: int = 8                          # was 6
    n_layers: int = 12                        # was 6
    d_ff: int = 2048                          # was 1536
    max_len: int = 8192                       # name-bearing; costs 4.2M params
    output_dim: int = 768                     # consensus space = teacher dim
    dropout: float = 0.1
    pooling: str = "mean"                     # arm: "cls". teachers are mean-pooled
    max_tokens: int = 256                     # dynamic pad ceiling
    # OOM FIX (2026-07-31, observed at B=2048): dynamic padding pads to the BATCH
    # max, and with 2048 draws the max is essentially always the ceiling. The corpus
    # mean is 48 tokens but every batch ran at L=256 -- attention memory goes as L^2,
    # so 12 layers needed ~120 GB against 95 available.
    #   length_bucketing  sorts within a shuffled window so a batch is length-
    #                     homogeneous: L tracks the corpus mean (~48-64) instead of
    #                     the ceiling. ~5x less memory AND ~5x less compute.
    #   grad_checkpointing bounds the worst case. The longest bucket IS a full batch
    #                     at L=256; checkpointing puts that at ~19 GB instead of
    #                     ~148 GB, for about 30% more compute.
    length_bucketing: bool = True
    bucket_window: int = 64                   # batches per sort window
    grad_checkpointing: bool = True
    vram_probe: bool = True                   # forward+backward at worst case first

    # ── training (sized for 95.6GB GPU: batch size IS the InfoNCE negative count) ──
    epochs: int = 4                           # 13.7k steps/ep at 2048 -> ~55k total
    batch_size: int = 2048                    # was 512; ~19GB activations, 4x negatives
    lr: float = 6e-4                          # sqrt-scaled from 3e-4 @ 512
    min_lr: float = 1e-6
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    seed: int = 42
    amp: bool = True
    num_workers: int = 0                      # RAM-resident: no workers needed
    log_every: int = 50
    eval_every: int = 1000
    ckpt_every_min: float = 20.0              # TIME-based: culls are wall-clock

    # ── loss: the champion form ──
    nce_weight: float = 1.0
    mse_weight: float = 1.0
    nce_temperature: float = 0.07
    cv_weight: float = 0.0                    # arm: 0.1 reproduces the v1 stack
    cv_target: float = 0.084

    # ── stages ──
    run_stage0: bool = True
    run_stage1: bool = True
    run_stage2: bool = True
    run_stage3: bool = True
    resume: bool = True


CFG = BaseConfig()


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def line(t=""):
    print("─" * 78 if not t else f"── {t} " + "─" * max(0, 74 - len(t)))


def paths(cfg) -> Dict[str, str]:
    w = cfg.work_dir
    d = {"root": w, "targets": f"{w}/targets", "maps": f"{w}/maps",
         "ckpt": f"{w}/checkpoints", "tb": f"{w}/tensorboard", "shards": f"{w}/shards",
         "config": f"{w}/config"}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def src0(cfg) -> Dict[str, Any]:
    return cfg.sources[0]


def usable_chunks(cfg) -> List[int]:
    s = src0(cfg)
    c = set(range(s["n_chunks"]))
    if cfg.require_all_experts:
        for miss in s.get("missing", {}).values():
            c -= set(miss)
    return sorted(c - set(cfg.holdout_chunks))


def fetch(cfg, fname: str) -> str:
    return hf_hub_download(src0(cfg)["repo"], fname, repo_type="dataset",
                           local_dir=paths(cfg)["shards"])


def load_captions_chunk(cfg, c: int) -> List[str]:
    raw = json.load(open(fetch(cfg, f"captions_{c:03d}.json")))
    f = cfg.caption_field
    if isinstance(raw, dict):
        return list(raw[f])
    if raw and isinstance(raw[0], dict):
        return [r[f] for r in raw]
    return list(raw)


def load_expert_chunk(cfg, expert: str, c: int) -> torch.Tensor:
    return torch.load(fetch(cfg, f"{expert}_{c:03d}.pt"),
                      weights_only=True, map_location="cpu")


def drop_shard(cfg, fname: str):
    if cfg.keep_expert_shards:
        return
    p = os.path.join(paths(cfg)["shards"], fname)
    if os.path.exists(p):
        os.remove(p)


def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def purge_hf_cache(cfg):
    """
    The expert shards total 507GB against a 235.7GB disk. hf_hub_download with
    local_dir does not populate the global cache on modern hub versions, but a
    stale HF_HOME cache or an older version WILL duplicate every shard and blow
    the disk mid-run. Purge both, every chunk.
    """
    for d in (os.path.join(paths(cfg)["shards"], ".cache"),
              os.environ.get("HF_HUB_CACHE", ""),
              os.path.expanduser("~/.cache/huggingface/hub")):
        if d and os.path.isdir(d):
            for entry in os.listdir(d):
                if entry.startswith("datasets--"):
                    shutil.rmtree(os.path.join(d, entry), ignore_errors=True)


def preflight(cfg):
    """Hard-check the allowance before anything expensive starts."""
    line("PREFLIGHT — disk / RAM / GPU vs the plan")
    P = paths(cfg)
    disk = free_gb(P["root"])
    s = src0(cfg)
    n_keep = len(usable_chunks(cfg)) + len(cfg.holdout_chunks)
    rows = n_keep * s["chunk_rows"]
    targets_gb = rows * cfg.output_dim * 2 / 1e9
    transient_gb = len(cfg.experts) * 1.536
    caps_gb = s["n_chunks"] * 0.120
    need = targets_gb + caps_gb + transient_gb + 10.0
    print(f"    source on HF : {s['n_chunks'] * len(cfg.experts) * 1.536:.0f} GB expert shards "
          f"(streamed one chunk at a time, deleted after)")
    print(f"    disk free    : {disk:.1f} GB | stage-2 peak need ≈ {need:.1f} GB "
          f"(targets {targets_gb:.1f} + captions {caps_gb:.1f} + transient {transient_gb:.1f})")
    if disk < need:
        raise RuntimeError(
            f"DISK: {disk:.1f} GB free, need ≈ {need:.1f} GB. Free space, reduce chunks, "
            f"or set push_targets=True and drop consensus locally after each push.")
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1e9
    except Exception:
        ram = float("nan")
    ram_need = (rows * 100 * 2 + rows * 8 + rows * cfg.output_dim * 2) / 1e9
    print(f"    RAM total    : {ram:.1f} GB | ram_resident need ≈ {ram_need:.1f} GB "
          f"(ragged tokens + offsets + fp16 targets)")
    if cfg.ram_resident and ram == ram and ram_need > 0.7 * ram:
        print(f"    !! ram_resident wants {ram_need:.1f} GB of {ram:.1f}. "
              f"Set ram_resident=False to stream per chunk from disk instead.")
    if DEVICE == "cuda":
        g = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    GPU          : {torch.cuda.get_device_name()} {g:.1f} GB | "
              f"batch {cfg.batch_size} -> {cfg.batch_size} InfoNCE negatives")
    print(f"    plan         : {rows:,} rows, {rows // cfg.batch_size:,} steps/epoch "
          f"x {cfg.epochs} = {rows // cfg.batch_size * cfg.epochs:,} steps")


def effective_rank(x: torch.Tensor) -> float:
    xc = (x - x.mean(0, keepdim=True)).double()
    s2 = torch.linalg.svdvals(xc) ** 2
    return float((s2.sum() ** 2 / (s2 ** 2).sum()).item())


def hf_token() -> Optional[str]:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# BACKUP — a Colab cull must cost minutes, not the run
# ══════════════════════════════════════════════════════════════════

class Backup:
    def __init__(self, cfg):
        self.cfg, self.api, self.ok, self.last = cfg, None, False, 0.0
        if not cfg.hf_push:
            return
        tok = hf_token()
        if not tok:
            print("    [backup] no HF_TOKEN — LOCAL ONLY. A cull will lose the run.")
            return
        try:
            create_repo(cfg.hf_repo, token=tok, exist_ok=True, private=True)
            self.api = HfApi(token=tok)
            self.ok = True
            print(f"    [backup] -> {cfg.hf_repo} (private)")
        except Exception as e:
            print(f"    [backup] disabled: {type(e).__name__}: {str(e)[:100]}")

    def push(self, force: bool = False, msg: str = "checkpoint"):
        if not self.ok:
            return
        if not force and (time.time() - self.last) / 60 < self.cfg.push_every_min:
            return
        P = paths(self.cfg)
        try:
            for folder, dest in ((P["ckpt"], "checkpoints"), (P["tb"], "tensorboard"),
                                 (P["maps"], "maps"), (P["config"], "config")):
                if os.path.isdir(folder) and os.listdir(folder):
                    self.api.upload_folder(folder_path=folder, path_in_repo=dest,
                                           repo_id=self.cfg.hf_repo,
                                           commit_message=f"{msg} ({dest})")
            self.last = time.time()
            print(f"    [backup] pushed ({msg})")
        except Exception as e:
            print(f"    [backup] push failed: {type(e).__name__}: {str(e)[:100]}")

    def pull_latest(self) -> Optional[str]:
        """Recover state.pt after a cull."""
        if not self.ok:
            return None
        try:
            p = hf_hub_download(self.cfg.hf_repo, "checkpoints/state.pt",
                                token=hf_token(), local_dir=paths(self.cfg)["root"])
            print(f"    [backup] recovered {p}")
            return p
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════
# STAGE 0 — PARITY GATE
# ══════════════════════════════════════════════════════════════════

def stage0_parity(cfg) -> str:
    """
    Which caption field was embedded, and is row i of <expert>_XXX.pt caption i?
    The manifest names three fields and does not say which was used. If the stored
    vectors came from caption_llava and the student trains on caption_llava_short,
    every target is silently wrong. Re-embed with the real reference model, demand
    cos ~ 1.0. Nothing downstream runs until this passes.
    """
    from transformers import AutoModel, AutoTokenizer
    line("STAGE 0 — PARITY GATE (caption field + row alignment)")
    stored = load_expert_chunk(cfg, cfg.ref_expert, cfg.parity_chunk)[: cfg.parity_n].float()
    raw = json.load(open(fetch(cfg, f"captions_{cfg.parity_chunk:03d}.json")))
    if isinstance(raw, dict):
        fields = {k: list(v)[: cfg.parity_n] for k, v in raw.items()
                  if k in cfg.caption_field_candidates}
    elif raw and isinstance(raw[0], dict):
        fields = {k: [r[k] for r in raw[: cfg.parity_n]]
                  for k in raw[0] if k in cfg.caption_field_candidates}
    else:
        fields = {"(flat)": list(raw[: cfg.parity_n])}
    print(f"    stored rows {tuple(stored.shape)} | fields {list(fields)}")

    tok = AutoTokenizer.from_pretrained(cfg.ref_hf_name)
    mdl = AutoModel.from_pretrained(cfg.ref_hf_name).to(DEVICE).eval()
    best, best_cos = None, -1.0
    for f, texts in fields.items():
        with torch.no_grad():
            inp = tok(list(texts), max_length=512, padding=True, truncation=True,
                      return_tensors="pt").to(DEVICE)
            h = mdl(**inp).last_hidden_state
            m = inp.attention_mask.unsqueeze(-1).float()
            pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu()
        cos = F.cosine_similarity(pooled, stored, dim=-1)
        print(f"      {f:22s} cos mean {cos.mean():.6f}  min {cos.min():.6f}")
        if cos.mean().item() > best_cos:
            best, best_cos = f, cos.mean().item()
    del mdl; gc.collect(); torch.cuda.empty_cache()
    if best_cos < cfg.parity_min_cos:
        raise RuntimeError(
            f"PARITY GATE FAIL: best field '{best}' only reaches cos {best_cos:.6f} "
            f"(need >= {cfg.parity_min_cos}). Either the field is not among "
            f"{cfg.caption_field_candidates}, row order differs, or the extraction used "
            f"different pooling/truncation. DO NOT SPEND GPU TIME until this resolves.")
    print(f"    GATE PASS: field = '{best}' at cos {best_cos:.6f}")
    return best


# ══════════════════════════════════════════════════════════════════
# STAGE 1 — GLOBAL WHITENED PROCRUSTES (out-of-sample reported)
# ══════════════════════════════════════════════════════════════════

def symmetric_inv_sqrt(cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    ev, evec = torch.linalg.eigh(cov.double())
    return (evec @ torch.diag(torch.clamp(ev, min=eps).rsqrt()) @ evec.T).float()


def fit_map(S: torch.Tensor, T: torch.Tensor) -> Dict[str, torch.Tensor]:
    N = S.shape[0]
    s_mean, t_mean = S.mean(0, keepdim=True), T.mean(0, keepdim=True)
    Sc, Tc = S - s_mean, T - t_mean
    s_w = symmetric_inv_sqrt((Sc.T @ Sc) / max(N - 1, 1))
    t_w = symmetric_inv_sqrt((Tc.T @ Tc) / max(N - 1, 1))
    U, _, Vt = torch.linalg.svd(
        (F.normalize(Tc @ t_w, dim=-1).T @ F.normalize(Sc @ s_w, dim=-1)).double(),
        full_matrices=False)
    return {"rotation": (U @ Vt).float(), "source_mean": s_mean.squeeze(0),
            "target_mean": t_mean.squeeze(0), "source_whitener": s_w,
            "target_whitener": t_w, "target_unwhitener": torch.linalg.pinv(t_w)}


def apply_map(emb: torch.Tensor, a) -> torch.Tensor:
    x = (emb.float() - a["source_mean"]) @ a["source_whitener"]
    return (x @ a["rotation"].T) @ a["target_unwhitener"]


def score_map(S, T, a) -> Dict[str, float]:
    Sw = F.normalize((S - a["source_mean"]) @ a["source_whitener"], dim=-1)
    Tw = F.normalize((T - a["target_mean"]) @ a["target_whitener"], dim=-1)
    cos = F.cosine_similarity(Sw @ a["rotation"].T, Tw, dim=-1).mean().item()
    n = min(2000, S.shape[0])
    sim = F.normalize(apply_map(S[:n], a), dim=-1) @ F.normalize(T[:n], dim=-1).T
    return {"cos": cos, "r1": (sim.argmax(1) == torch.arange(n)).float().mean().item(),
            "n": int(S.shape[0]), "chance": 1.0 / n}


def stage1_fit(cfg, bk: "Backup"):
    line("STAGE 1 — GLOBAL ALIGNMENT (stratified fit, OUT-OF-SAMPLE report)")
    P = paths(cfg)
    mp = f"{P['maps']}/alignment_maps.pt"
    if os.path.exists(mp):
        print("    maps exist, loading"); return torch.load(mp, weights_only=False)

    g = torch.Generator().manual_seed(cfg.fit_seed)
    fit = {e: [] for e in cfg.experts}
    for c in cfg.fit_chunks:
        idx = None
        for e in cfg.experts:
            X = load_expert_chunk(cfg, e, c)
            if idx is None:
                idx = torch.randperm(X.shape[0], generator=g)[: cfg.fit_rows_per_chunk]
            fit[e].append(X[idx].float()); del X; gc.collect()
            drop_shard(cfg, f"{e}_{c:03d}.pt")
        print(f"    fit chunk {c:03d}: {len(idx)} random rows")
    fit = {e: torch.cat(v) for e, v in fit.items()}
    N = fit[cfg.ref_expert].shape[0]
    print(f"    fit set {N} rows, d=768 -> N/d = {N/768:.1f}")

    hold = {e: [] for e in cfg.experts}
    for c in cfg.holdout_chunks:
        for e in cfg.experts:
            X = load_expert_chunk(cfg, e, c)
            hold[e].append(X[: cfg.fit_rows_per_chunk].float()); del X; gc.collect()
    hold = {e: torch.cat(v) for e, v in hold.items()}

    maps, report, T = {}, {}, fit[cfg.ref_expert]
    for e in cfg.experts:
        a = fit_map(fit[e], T)
        ins, oos = score_map(fit[e], T, a), score_map(hold[e], hold[cfg.ref_expert], a)
        maps[e], report[e] = a, {"in_sample": ins, "out_of_sample": oos}
        tag = "  (ref: must read ~1.0)" if e == cfg.ref_expert else ""
        print(f"    {e:9s} cos in {ins['cos']:.4f} / OUT {oos['cos']:.4f}   "
              f"R@1 in {ins['r1']:.4f} / OUT {oos['r1']:.4f} "
              f"(chance {oos['chance']:.5f}){tag}")
    print("    READ THE 'OUT' COLUMN. A 768x768 rotation is 294,528 free parameters;")
    print("    at low N/d the in-sample cosine reproduces strong numbers from nothing.")
    torch.save(maps, mp)
    json.dump(report, open(f"{P['maps']}/fit_report.json", "w"), indent=2)
    bk.push(force=True, msg="stage1 alignment maps")
    return maps


# ══════════════════════════════════════════════════════════════════
# STAGE 2 — CONSENSUS TARGETS
# ══════════════════════════════════════════════════════════════════

def stage2_targets(cfg, maps, bk: "Backup") -> List[int]:
    line("STAGE 2 — CONSENSUS TARGETS (fp16, per chunk, resumable)")
    P = paths(cfg)
    lp = f"{P['targets']}/ledger.json"
    ledger = json.load(open(lp)) if os.path.exists(lp) else {}
    want = sorted(set(usable_chunks(cfg)) | set(cfg.holdout_chunks))
    print(f"    {len(want)} chunks with all {len(cfg.experts)} experts | "
          f"streaming {len(want)*len(cfg.experts)*1.536:.0f} GB through "
          f"{free_gb(P['root']):.0f} GB of free disk")
    tapi = None
    if cfg.push_targets and bk.ok:
        try:
            create_repo(cfg.targets_repo, token=hf_token(), exist_ok=True,
                        private=True, repo_type="dataset")
            tapi = HfApi(token=hf_token())
            print(f"    targets -> {cfg.targets_repo} (dataset, private)")
        except Exception as e:
            print(f"    target push disabled: {type(e).__name__}: {str(e)[:80]}")
    for c in want:
        k, out_p = f"{c:03d}", f"{P['targets']}/consensus_{c:03d}.pt"
        if ledger.get(k) and os.path.exists(out_p):
            continue
        if free_gb(P["root"]) < cfg.disk_floor_gb:
            raise RuntimeError(f"DISK FLOOR: {free_gb(P['root']):.1f} GB free at chunk {k}. "
                               f"Push and drop earlier consensus files, then resume.")
        acc, n = None, None
        for e in cfg.experts:
            X = load_expert_chunk(cfg, e, c).float()
            if n is None:
                n = X.shape[0]
            elif X.shape[0] != n:
                raise RuntimeError(f"chunk {k}: {e} has {X.shape[0]} rows, expected {n}")
            A = apply_map(X, maps[e])
            acc = A if acc is None else acc + A
            del X, A; gc.collect()
            drop_shard(cfg, f"{e}_{c:03d}.pt")
        purge_hf_cache(cfg)
        cons = F.normalize(acc / len(cfg.experts), dim=-1).half()
        torch.save(cons, out_p)
        er = effective_rank(cons[:4000].float())
        ledger[k] = {"rows": int(cons.shape[0]), "target_erank": er, "ts": time.time()}
        json.dump(ledger, open(lp, "w"), indent=2)
        if tapi is not None:
            try:
                tapi.upload_file(path_or_fileobj=out_p,
                                 path_in_repo=f"consensus_{k}.pt",
                                 repo_id=cfg.targets_repo, repo_type="dataset",
                                 commit_message=f"consensus chunk {k}")
            except Exception as ex:
                print(f"      target push failed for {k}: {str(ex)[:70]}")
        print(f"    chunk {k}: {cons.shape[0]} targets | TARGET erank {er:.1f}/768 | "
              f"disk free {free_gb(P['root']):.0f} GB")
        del acc, cons; gc.collect()
    eranks = [v["target_erank"] for v in ledger.values() if "target_erank" in v]
    if eranks:
        print(f"    consensus target erank: mean {np.mean(eranks):.1f} "
              f"min {min(eranks):.1f} max {max(eranks):.1f} of 768")
        print("    (v1's STUDENT read 23.6 — compare against this to tell 'student")
        print("     collapsed' from 'student faithfully matched a low-rank target')")
    bk.push(force=True, msg="stage2 target ledger")
    return want


# ══════════════════════════════════════════════════════════════════
# STUDENT
# ══════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════
# LOSS / GAUGES
# ══════════════════════════════════════════════════════════════════

def infonce(a, b, temperature=0.07):
    logits = (a @ b.T) / temperature
    lab = torch.arange(logits.shape[0], device=logits.device)
    loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
    with torch.no_grad():
        acc = (logits.argmax(-1) == lab).float().mean().item()
    return loss, acc


def cayley_menger_vol2(pts):
    pts = pts.float()
    d = pts.unsqueeze(-2) - pts.unsqueeze(-3)
    d2 = (d * d).sum(-1)
    B, V, _ = d2.shape
    cm = torch.zeros(B, V + 1, V + 1, device=d2.device, dtype=torch.float32)
    cm[:, 0, 1:] = 1; cm[:, 1:, 0] = 1; cm[:, 1:, 1:] = d2
    f = math.factorial(V - 1)
    return ((-1.0) ** V) / ((2.0 ** (V - 1)) * f * f) * torch.linalg.det(cm)


def cv_loss(emb, target=0.084, n_samples=16):
    B = emb.shape[0]
    if B < 5:
        return torch.zeros((), device=emb.device)
    s = torch.stack([torch.sqrt(F.relu(cayley_menger_vol2(
        emb[torch.randperm(B, device=emb.device)[:5]].unsqueeze(0))[0]) + 1e-12)
        for _ in range(n_samples)])
    return (s.std() / (s.mean() + 1e-8) - target).abs()


@torch.no_grad()
def cv_metric(emb, n=200):
    v = [float(torch.sqrt(F.relu(cayley_menger_vol2(
        emb[torch.randperm(emb.shape[0], device=emb.device)[:5]].unsqueeze(0))[0])
        + 1e-12).item()) for _ in range(n)]
    a = np.array([x for x in v if x > 0])
    return float(a.std() / (a.mean() + 1e-8)) if len(a) >= 10 else 0.0


@torch.no_grad()
def frame_fit_gauge(E: torch.Tensor, T: torch.Tensor, n_pairs: int = 2500) -> Dict[str, float]:
    """
    Standing rider: judge relational objectives with a frame fit or they read as false
    floors. MSE anchors the frame here and the consensus aligns to a REFERENCE MEMBER,
    so a rotation should buy ~nothing. If it buys a lot, the frame is NOT pinned and
    this model needs a shipped rotation after all. Held-out split, fp64.
    """
    N = E.shape[0]
    k = min(n_pairs, N // 2)
    if k < 64:
        return {"skipped": True}
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    fit_i, hold_i = perm[:k], perm[k:]
    U, _, Vt = torch.linalg.svd(E[fit_i].double().T @ T[fit_i].double(), full_matrices=False)
    Er = F.normalize((E.double() @ (U @ Vt)).float(), dim=-1)
    m = min(2000, len(hold_i))
    hi = hold_i[:m]
    sim = Er[hi] @ T[hi].T
    return {"r1_after_rotation": (sim.argmax(1) == torch.arange(m)).float().mean().item(),
            "cos_after_rotation": F.cosine_similarity(Er[hi], T[hi], dim=-1).mean().item(),
            "n_heldout": int(m)}


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════

class RamStore:
    """
    Everything resident in system RAM: ragged uint16 tokens + fp16 targets.

    On the Pro+ box this is 48.8 GB of 176.9 — so the training loop does ZERO disk
    I/O and needs no DataLoader workers. Ragged storage (flat token buffer + offsets)
    keeps dynamic padding available at ~5.6 GB instead of the 14 GB a fixed 256-token
    matrix would cost, and captions average ~100 tokens against a 256 ceiling.
    """

    def __init__(self, cfg, chunks: List[int], tokenizer, tag=""):
        self.cfg, self.tok = cfg, tokenizer
        self.pad = tokenizer.pad_token_id
        flat, offs, tgts, total = [], [0], [], 0
        for c in chunks:
            caps = load_captions_chunk(cfg, c)
            t = torch.load(f"{paths(cfg)['targets']}/consensus_{c:03d}.pt",
                           weights_only=True, map_location="cpu")
            n = min(len(caps), t.shape[0])
            caps, t = caps[:n], t[:n]
            for i in range(0, n, 20000):
                enc = tokenizer(caps[i:i + 20000], max_length=cfg.max_tokens,
                                truncation=True, padding=False)["input_ids"]
                for ids in enc:
                    flat.append(np.asarray(ids, dtype=np.uint16))
                    total += len(ids)
                    offs.append(total)
            tgts.append(t)
            print(f"      chunk {c:03d}: {n:,} rows | flat tokens {total/1e6:.1f}M")
            del caps, t; gc.collect()
        self.flat = np.concatenate(flat) if flat else np.zeros(0, np.uint16)
        del flat; gc.collect()
        self.offs = np.asarray(offs, dtype=np.int64)
        self.tgt = torch.cat(tgts)
        del tgts; gc.collect()
        self.n = len(self.offs) - 1
        self.lens = (self.offs[1:] - self.offs[:-1]).astype(np.int32)
        gb = (self.flat.nbytes + self.offs.nbytes + self.tgt.numel() * 2) / 1e9
        mean_len = total / max(self.n, 1)
        q = np.percentile(self.lens, [50, 90, 99, 100]).astype(int)
        print(f"    RamStore{tag}: {self.n:,} rows | {gb:.1f} GB RAM | "
              f"mean {mean_len:.0f} tokens (ceiling {cfg.max_tokens})")
        print(f"      length p50 {q[0]} | p90 {q[1]} | p99 {q[2]} | max {q[3]}"
              f"  -- unbucketed, a batch pads to the BATCH MAX, i.e. ~{q[3]}")

    def plan_batches(self, batch_size, seed, window_batches=64, bucket=True):
        """
        Deterministic batch plan for one epoch. Returns a list of index arrays.

        With bucket=True: shuffle, cut into windows of window_batches*batch_size,
        sort each window by length, slice into batches, then shuffle the BATCH ORDER.
        Batches end up length-homogeneous (so padding is near-free) while batch
        composition stays random across the window and the model never sees the
        corpus in length order. Deterministic in (seed), so a resume mid-epoch
        regenerates the identical plan and the stored batch index stays valid.
        """
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n)
        if not bucket:
            n_full = self.n // batch_size
            return [perm[i * batch_size:(i + 1) * batch_size] for i in range(n_full)]
        W = batch_size * max(window_batches, 1)
        batches = []
        for i in range(0, self.n, W):
            win = perm[i:i + W]
            win = win[np.argsort(self.lens[win], kind="stable")]
            for j in range(0, len(win) - batch_size + 1, batch_size):
                batches.append(win[j:j + batch_size])
        rng.shuffle(batches)
        return batches

    def __len__(self):
        return self.n

    def batch(self, idx: np.ndarray):
        """Gather a batch with DYNAMIC padding to the batch max."""
        seqs = [self.flat[self.offs[i]:self.offs[i + 1]] for i in idx]
        L = max(len(s) for s in seqs)
        ids = np.full((len(seqs), L), self.pad, dtype=np.int64)
        am = np.zeros((len(seqs), L), dtype=np.int64)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = s
            am[r, :len(s)] = 1
        return (torch.from_numpy(ids), torch.from_numpy(am),
                self.tgt[torch.from_numpy(idx)])


class ChunkPairs(torch.utils.data.Dataset):
    """Disk-streaming fallback when ram_resident=False."""
    def __init__(self, cfg, chunk, tokenizer):
        self.caps = load_captions_chunk(cfg, chunk)
        self.tgt = torch.load(f"{paths(cfg)['targets']}/consensus_{chunk:03d}.pt",
                              weights_only=True, map_location="cpu")
        n = min(len(self.caps), self.tgt.shape[0])
        self.caps, self.tgt = self.caps[:n], self.tgt[:n]
        self.tok, self.max_tokens = tokenizer, cfg.max_tokens

    def __len__(self):
        return len(self.caps)

    def __getitem__(self, i):
        return self.caps[i], self.tgt[i]

    def collate(self, batch):
        texts, tg = zip(*batch)
        enc = self.tok(list(texts), max_length=self.max_tokens, padding=True,
                       truncation=True, return_tensors="pt")   # DYNAMIC
        return enc["input_ids"], enc["attention_mask"], torch.stack(tg)


@torch.no_grad()
def evaluate(student, source, cap=5000, batch=512) -> Dict[str, float]:
    student.eval()
    E, T = [], []
    if isinstance(source, RamStore):
        for i in range(0, min(cap, len(source)), batch):
            ids, am, tg = source.batch(np.arange(i, min(i + batch, len(source))))
            E.append(student(ids.to(DEVICE), am.to(DEVICE)).float().cpu())
            T.append(tg.float())
    else:
        for ids, am, tg in source:
            E.append(student(ids.to(DEVICE), am.to(DEVICE)).float().cpu())
            T.append(tg.float())
            if sum(x.shape[0] for x in E) >= cap:
                break
    E, T = torch.cat(E), F.normalize(torch.cat(T), dim=-1)
    n = min(2000, E.shape[0])
    sim = E[:n] @ T[:n].T
    ss = E[:n] @ E[:n].T
    ss.fill_diagonal_(0)
    out = {"mimicry_r1": (sim.argmax(1) == torch.arange(n)).float().mean().item(),
           "cos_to_target": F.cosine_similarity(E, T, dim=-1).mean().item(),
           "self_cos": ss.mean().item(),
           "erank": effective_rank(E),
           "cv": cv_metric(E[:2000].to(DEVICE)),
           "n": int(E.shape[0])}
    out.update({f"frame_{k}": v for k, v in frame_fit_gauge(E, T).items()})
    student.train()
    return out


# ══════════════════════════════════════════════════════════════════
# STAGE 3 — TRAIN (cull-proof)
# ══════════════════════════════════════════════════════════════════

def vram_probe(cfg, student):
    """
    One forward+backward at the WORST case (full batch at the pad ceiling) before
    any data is loaded. With bucketing the longest bucket really is a full batch at
    max_tokens, so this is the case that decides whether the run survives -- and it
    is far cheaper to discover here than 20 minutes into a RamStore build.
    """
    if DEVICE != "cuda":
        return
    line("VRAM PROBE - worst-case batch before spending time on data")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    ids = torch.randint(1, 30000, (cfg.batch_size, cfg.max_tokens), device=DEVICE)
    am = torch.ones_like(ids)
    tgt = F.normalize(torch.randn(cfg.batch_size, cfg.output_dim, device=DEVICE), dim=-1)
    opt = torch.optim.Adam(student.parameters(), lr=1e-9)
    try:
        student.train()
        with torch.amp.autocast("cuda", enabled=cfg.amp):
            emb = student(ids, am)
        emb = emb.float()
        loss = infonce(emb, tgt, cfg.nce_temperature)[0] + F.mse_loss(emb, tgt)
        loss.backward()
        opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"    batch {cfg.batch_size} x L {cfg.max_tokens} "
              f"(checkpointing={cfg.grad_checkpointing}) -> peak {peak:.1f} GB "
              f"of {total:.1f} GB")
        if peak > 0.85 * total:
            print("    !! within 15% of the limit. Reduce batch_size or max_tokens,")
            print("    !! or set grad_checkpointing=True, before starting the run.")
        else:
            ok = (total - peak)
            print(f"    PASS - {ok:.1f} GB headroom")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"VRAM PROBE FAILED at batch {cfg.batch_size} x L {cfg.max_tokens} "
            f"(checkpointing={cfg.grad_checkpointing}). Options, cheapest first: "
            f"grad_checkpointing=True; lower max_tokens (corpus mean is ~48); "
            f"halve batch_size (costs InfoNCE negatives). Nothing was loaded, so "
            f"changing the config and re-running is quick.")
    finally:
        del ids, am, tgt, opt
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def save_state(cfg, path, student, opt, sched, scaler, step, epoch, chunk_i, order, best):
    torch.save({"model": student.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "epoch": epoch, "chunk_i": chunk_i, "order": order,
                "best": best, "config": asdict(cfg),
                "rng": {"torch": torch.get_rng_state(), "np": np.random.get_state(),
                        "py": random.getstate()}}, path)


def stage3_train(cfg, chunks: List[int], bk: "Backup"):
    from transformers import AutoTokenizer
    line("STAGE 3 — TRAIN")
    P = paths(cfg)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.ref_hf_name)
    json.dump(asdict(cfg), open(f"{P['config']}/config.json", "w"), indent=2, default=str)

    student = CaptionEncoder(
        vocab_size=tok.vocab_size, max_len=cfg.max_len, d_model=cfg.d_model,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers, d_ff=cfg.d_ff,
        output_dim=cfg.output_dim, dropout=cfg.dropout,
        pad_token_id=tok.pad_token_id, pooling=cfg.pooling,
        grad_checkpointing=cfg.grad_checkpointing).to(DEVICE)
    n_par = sum(p.numel() for p in student.parameters())
    train_chunks = [c for c in chunks if c not in cfg.holdout_chunks]
    rows = len(train_chunks) * src0(cfg)["chunk_rows"]
    spe = rows // cfg.batch_size
    total = spe * cfg.epochs
    print(f"    {cfg.run_name}: {n_par:,} params ({n_par/109_482_240:.2f}x bert-base)")
    print(f"    {cfg.n_layers}L {cfg.d_model}d {cfg.n_heads}h ff{cfg.d_ff} pool={cfg.pooling}")
    print(f"    {len(train_chunks)} chunks ≈ {rows:,} rows | {spe:,} steps/ep x "
          f"{cfg.epochs} = {total:,} steps @ batch {cfg.batch_size}")
    print(f"    loss = {cfg.nce_weight}*InfoNCE(T={cfg.nce_temperature}) + "
          f"{cfg.mse_weight}*MSE + {cfg.cv_weight}*CV   [champion consensus_nce_mse]")

    if cfg.vram_probe:
        vram_probe(cfg, student)

    opt = torch.optim.Adam(student.parameters(), lr=cfg.lr)   # pure Adam, no wd
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, cfg.warmup_steps),
              torch.optim.lr_scheduler.CosineAnnealingLR(
                  opt, T_max=max(total - cfg.warmup_steps, 1), eta_min=cfg.min_lr)],
        milestones=[cfg.warmup_steps])
    scaler = torch.amp.GradScaler(enabled=cfg.amp and DEVICE == "cuda")
    tb = SummaryWriter(log_dir=f"{P['tb']}/{cfg.run_name}")
    tb.add_text("config", f"```json\n{json.dumps(asdict(cfg), indent=2, default=str)}\n```")
    if os.path.exists(f"{P['maps']}/fit_report.json"):
        tb.add_text("alignment/fit_report",
                    f"```json\n{open(f'{P['maps']}/fit_report.json').read()}\n```")

    step, ep0, chunk_i0, best = 0, 0, 0, -1.0
    order = None
    sp = f"{P['ckpt']}/state.pt"
    if cfg.resume:
        if not os.path.exists(sp):
            bk.pull_latest()
            alt = f"{P['root']}/checkpoints/state.pt"
            if os.path.exists(alt) and alt != sp:
                shutil.copy(alt, sp)
        if os.path.exists(sp):
            st = torch.load(sp, weights_only=False, map_location=DEVICE)
            student.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
            step, ep0, chunk_i0, best = st["step"], st["epoch"], st["chunk_i"], st["best"]
            order = st.get("order")
            try:
                torch.set_rng_state(st["rng"]["torch"].cpu())
                np.random.set_state(st["rng"]["np"]); random.setstate(st["rng"]["py"])
            except Exception:
                pass
            print(f"    RESUMED at step {step:,} epoch {ep0+1} chunk_i {chunk_i0}")

    print("    building val store...")
    if cfg.ram_resident:
        val_src = RamStore(cfg, [cfg.holdout_chunks[-1]], tok, tag=" [val]")
    else:
        vds = ChunkPairs(cfg, cfg.holdout_chunks[-1], tok)
        val_src = torch.utils.data.DataLoader(
            vds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=vds.collate)

    if cfg.ram_resident:
        print("    building train store (one pass, then zero disk I/O)...")
        train_src = RamStore(cfg, train_chunks, tok, tag=" [train]")
        N = len(train_src)
        spe = N // cfg.batch_size
        total = spe * cfg.epochs
        print(f"    {N:,} rows resident | {spe:,} steps/ep x {cfg.epochs} = {total:,} steps")

    t0 = last_ck = time.time()
    for ep in range(ep0, cfg.epochs):
        if cfg.ram_resident:
            # deterministic bucketed plan; chunk_i doubles as the batch index, so a
            # mid-epoch resume regenerates the identical plan and lands on the same batch
            plan = train_src.plan_batches(cfg.batch_size, cfg.seed + ep,
                                          cfg.bucket_window, cfg.length_bucketing)
            if ep == ep0:
                spe = len(plan); total = spe * cfg.epochs
                bl = np.array([train_src.lens[b].max() for b in plan[:200]])
                print(f"    batch plan: {spe:,} batches/epoch | padded length "
                      f"p50 {int(np.percentile(bl,50))} p90 {int(np.percentile(bl,90))} "
                      f"max {int(bl.max())} (bucketing={cfg.length_bucketing})")
            for ci in range(chunk_i0 if ep == ep0 else 0, len(plan)):
                ids, am, tg = train_src.batch(plan[ci])
                ids = ids.to(DEVICE, non_blocking=True)
                am = am.to(DEVICE, non_blocking=True)
                tgt = F.normalize(tg.to(DEVICE, non_blocking=True).float(), dim=-1)
                with torch.amp.autocast("cuda", enabled=cfg.amp and DEVICE == "cuda"):
                    emb = student(ids, am)
                emb = emb.float()
                l_nce, acc = infonce(emb, tgt, cfg.nce_temperature)
                l_mse = F.mse_loss(emb, tgt)
                loss = cfg.nce_weight * l_nce + cfg.mse_weight * l_mse
                l_cv = torch.zeros((), device=emb.device)
                if cfg.cv_weight > 0:
                    l_cv = cv_loss(emb, cfg.cv_target)
                    loss = loss + cfg.cv_weight * l_cv
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sched.step()
                step += 1

                if step % cfg.log_every == 0:
                    lr = opt.param_groups[0]["lr"]
                    tb.add_scalar("train/loss", loss.item(), step)
                    tb.add_scalar("train/nce", l_nce.item(), step)
                    tb.add_scalar("train/mse", l_mse.item(), step)
                    tb.add_scalar("train/cv", float(l_cv), step)
                    tb.add_scalar("train/batch_acc", acc, step)
                    tb.add_scalar("train/lr", lr, step)
                    tb.add_scalar("train/grad_norm", float(gn), step)
                    tb.add_scalar("train/tokens_per_seq", ids.shape[1], step)
                    print(f"    e{ep+1} {step:>7,}/{total:,} loss {loss.item():.4f} "
                          f"nce {l_nce.item():.4f} mse {l_mse.item():.5f} acc {acc:.3f} "
                          f"lr {lr:.2e} L{ids.shape[1]} {(time.time()-t0)/60:.0f}m")

                if step % cfg.eval_every == 0:
                    m = evaluate(student, val_src)
                    for k, v in m.items():
                        if isinstance(v, (int, float)):
                            tb.add_scalar(f"val/{k}", v, step)
                    for nm, p in student.named_parameters():
                        if p.grad is not None and ("output_proj" in nm or "token_emb" in nm):
                            tb.add_histogram(f"grad/{nm}", p.grad, step)
                            tb.add_histogram(f"weight/{nm}", p, step)
                    print(f"      VAL r1 {m['mimicry_r1']:.4f} cos {m['cos_to_target']:.4f} "
                          f"self_cos {m['self_cos']:+.4f} erank {m['erank']:.1f} "
                          f"cv {m['cv']:.4f} | frame r1 "
                          f"{m.get('frame_r1_after_rotation', float('nan')):.4f}")
                    if m["cos_to_target"] > best:
                        best = m["cos_to_target"]
                        save_state(cfg, f"{P['ckpt']}/best_state.pt", student, opt,
                                   sched, scaler, step, ep, ci, order, best)
                        torch.save(student.state_dict(), f"{P['ckpt']}/best_model.pt")

                if (time.time() - last_ck) / 60 >= cfg.ckpt_every_min:
                    save_state(cfg, sp, student, opt, sched, scaler, step, ep, ci, order, best)
                    torch.save(student.state_dict(), f"{P['ckpt']}/model_s{step}.pt")
                    ck = sorted([f for f in os.listdir(P["ckpt"]) if f.startswith("model_s")],
                                key=lambda f: int(f.split("_s")[1].split(".")[0]))
                    for old in ck[:-cfg.keep_local_ckpts]:
                        os.remove(os.path.join(P["ckpt"], old))
                    tb.flush(); bk.push(msg=f"step {step}")
                    last_ck = time.time()
        else:
            if order is None or ep != ep0:
                order = train_chunks[:]; random.shuffle(order)
            for ci in range(chunk_i0 if ep == ep0 else 0, len(order)):
                c = order[ci]
                ds = ChunkPairs(cfg, c, tok)
                dl = torch.utils.data.DataLoader(
                    ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                    num_workers=cfg.num_workers, collate_fn=ds.collate,
                    pin_memory=(DEVICE == "cuda"))
                for ids, am, tg in dl:
                    ids = ids.to(DEVICE, non_blocking=True)
                    am = am.to(DEVICE, non_blocking=True)
                    tgt = F.normalize(tg.to(DEVICE, non_blocking=True).float(), dim=-1)
                    with torch.amp.autocast("cuda", enabled=cfg.amp and DEVICE == "cuda"):
                        emb = student(ids, am)
                    emb = emb.float()
                    l_nce, acc = infonce(emb, tgt, cfg.nce_temperature)
                    l_mse = F.mse_loss(emb, tgt)
                    loss = cfg.nce_weight * l_nce + cfg.mse_weight * l_mse
                    l_cv = torch.zeros((), device=emb.device)
                    if cfg.cv_weight > 0:
                        l_cv = cv_loss(emb, cfg.cv_target)
                        loss = loss + cfg.cv_weight * l_cv
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True); sched.step()
                    step += 1

                    if step % cfg.log_every == 0:
                        lr = opt.param_groups[0]["lr"]
                        tb.add_scalar("train/loss", loss.item(), step)
                        tb.add_scalar("train/nce", l_nce.item(), step)
                        tb.add_scalar("train/mse", l_mse.item(), step)
                        tb.add_scalar("train/cv", float(l_cv), step)
                        tb.add_scalar("train/batch_acc", acc, step)
                        tb.add_scalar("train/lr", lr, step)
                        tb.add_scalar("train/grad_norm", float(gn), step)
                        tb.add_scalar("train/tokens_per_seq", ids.shape[1], step)
                        print(f"    e{ep+1} {step:>7,}/{total:,} loss {loss.item():.4f} "
                              f"nce {l_nce.item():.4f} mse {l_mse.item():.5f} acc {acc:.3f} "
                              f"lr {lr:.2e} L{ids.shape[1]} {(time.time()-t0)/60:.0f}m")

                    if step % cfg.eval_every == 0:
                        m = evaluate(student, val_src)
                        for k, v in m.items():
                            if isinstance(v, (int, float)):
                                tb.add_scalar(f"val/{k}", v, step)
                        for nm, p in student.named_parameters():
                            if p.grad is not None and ("output_proj" in nm or "token_emb" in nm):
                                tb.add_histogram(f"grad/{nm}", p.grad, step)
                                tb.add_histogram(f"weight/{nm}", p, step)
                        print(f"      VAL r1 {m['mimicry_r1']:.4f} cos {m['cos_to_target']:.4f} "
                              f"self_cos {m['self_cos']:+.4f} erank {m['erank']:.1f} "
                              f"cv {m['cv']:.4f} | frame r1 "
                              f"{m.get('frame_r1_after_rotation', float('nan')):.4f}")
                        if m["cos_to_target"] > best:
                            best = m["cos_to_target"]
                            save_state(cfg, f"{P['ckpt']}/best_state.pt", student, opt,
                                       sched, scaler, step, ep, ci, order, best)
                            torch.save(student.state_dict(), f"{P['ckpt']}/best_model.pt")

                    if (time.time() - last_ck) / 60 >= cfg.ckpt_every_min:
                        save_state(cfg, sp, student, opt, sched, scaler, step, ep, ci, order, best)
                        torch.save(student.state_dict(), f"{P['ckpt']}/model_s{step}.pt")
                        ck = sorted([f for f in os.listdir(P["ckpt"]) if f.startswith("model_s")],
                                    key=lambda f: int(f.split("_s")[1].split(".")[0]))
                        for old in ck[:-cfg.keep_local_ckpts]:
                            os.remove(os.path.join(P["ckpt"], old))
                        tb.flush(); bk.push(msg=f"step {step}")
                        last_ck = time.time()
                del ds, dl; gc.collect()
        chunk_i0 = 0

    save_state(cfg, sp, student, opt, sched, scaler, step, cfg.epochs, 0, order, best)
    torch.save(student.state_dict(), f"{P['ckpt']}/final_model.pt")
    tok.save_pretrained(f"{P['ckpt']}/tokenizer")
    m = evaluate(student, val_src)
    line("FINAL")
    print(f"    mimicry R@1 (student->consensus, NOT capability): {m['mimicry_r1']:.4f}")
    print(f"    cos to target : {m['cos_to_target']:.4f}")
    print(f"    self_cos      : {m['self_cos']:+.4f}   <- isotropy; teachers .81-.98")
    print(f"    effective rank: {m['erank']:.1f}/{cfg.output_dim}")
    print(f"    CV            : {m['cv']:.4f}")
    print(f"    frame-fit R@1 : {m.get('frame_r1_after_rotation', float('nan')):.4f} "
          f"(should be ~mimicry: reference-member alignment pins the frame)")
    print("    CAPABILITY is decided by STS-B / SICK vs the five teachers, not here.")
    json.dump({"config": asdict(cfg), "final": m}, open(f"{P['ckpt']}/metrics.json", "w"),
              indent=2, default=str)
    tb.flush(); tb.close(); bk.push(force=True, msg="final")
    return student


# ══════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════

def run(cfg: BaseConfig = CFG):
    print("=" * 78)
    print(f"{cfg.run_name.upper()} — CONSENSUS DISTILLATION, CC12M SCALE")
    print("=" * 78)
    paths(cfg)
    print(f"device={DEVICE}  work_dir={cfg.work_dir}")
    if DEVICE == "cuda":
        print(f"gpu={torch.cuda.get_device_name()} "
              f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
    miss = src0(cfg).get("missing", {})
    print(f"chunks: {len(usable_chunks(cfg))} train + {len(cfg.holdout_chunks)} holdout "
          f"| excluded for missing experts: {miss}")
    if not cfg.require_all_experts:
        print("    !! require_all_experts=False -> 4-expert consensus on some chunks.")
        print("    !! The target definition then differs BETWEEN chunks. Discouraged.")
    bk = Backup(cfg)

    if cfg.run_stage0:
        cfg.caption_field = stage0_parity(cfg)
    elif cfg.caption_field is None:
        raise RuntimeError("caption_field is None and stage 0 is disabled.")

    maps = stage1_fit(cfg, bk) if cfg.run_stage1 else torch.load(
        f"{paths(cfg)['maps']}/alignment_maps.pt", weights_only=False)
    chunks = stage2_targets(cfg, maps, bk) if cfg.run_stage2 else sorted(
        set(usable_chunks(cfg)) | set(cfg.holdout_chunks))
    if cfg.run_stage3:
        return stage3_train(cfg, chunks, bk)


if "get_ipython" in globals() or __name__ == "__main__":
    STUDENT = run(CFG)