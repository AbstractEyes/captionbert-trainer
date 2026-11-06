# ============================================================================
# CAPTIONBERT FULL BENCHMARK -- teachers, MiniLM, both trunks, arms
#
# One harness, one pass, every model measured on the SAME eight tasks with the
# SAME pooling and normalization. The card tables so far mixed sources: the
# teacher numbers came from a 2-task run, the trunk numbers from an 8-task run,
# and MiniLM was quoted for scale from a different pass. That is not a fair
# comparison and it is not defensible in a writeup.
#
# WHAT IS MEASURED
#   5 teachers      bert-base, ModernBERT-base, roberta-base, albert-base-v2,
#                   distilbert -- the exact models the consensus was built from
#   reference       all-MiniLM-L6-v2 (contrastive, 1B+ curated pairs: a
#                   DIFFERENT comparison class, labelled as such)
#   2 trunks        captionbert-8192-v2 (54 chunks) and -b (66 chunks)
#   2 arm sets      each trunk with ITS OWN native arms -- anchors are
#                   trunk-bound (v2 arms on -b cost 31% of their gain)
#
# 8 TASKS: STS-B, SICK-R, STS12-16, BIOSSES. BIOSSES is 100 rows and is the only
# genuinely out-of-domain gauge; it is reported but never used alone.
#
# EVERY MODEL IS MEAN-POOLED AND L2-NORMALIZED. That is the honest setting for
# an untuned encoder and it is what the teachers were consensus-averaged in.
# It is also why bert-base scores low here: raw mean-pooled BERT is a known-weak
# sentence encoder, which is the entire reason Sentence-BERT exists. Beating it
# is a real efficiency result, not a competitive sentence-embedding result --
# the card should say so and the MiniLM row is there to keep that honest.
#
# Config at the top, functionality in the body, run logic at the base.
# ============================================================================

import gc
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

for _p, _i in [("datasets", "datasets"), ("transformers", "transformers"),
               ("scipy", "scipy"), ("huggingface_hub", "huggingface_hub"),
               ("amoe-lora @ git+https://github.com/AbstractEyes/amoe-lora", "amoe")]:
    try:
        __import__(_i)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _p], check=False)

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════
# BASE CONFIG
# ══════════════════════════════════════════════════════════════════

@dataclass
class BaseConfig:
    # ---- the five teachers the consensus was built from ----
    teachers: tuple = (
        ("bert-base", "google-bert/bert-base-uncased"),
        ("ModernBERT-base", "answerdotai/ModernBERT-base"),
        ("roberta-base", "FacebookAI/roberta-base"),
        ("albert-base-v2", "albert/albert-base-v2"),
        ("distilbert", "distilbert/distilbert-base-uncased"),
    )
    # ---- reference point, NOT a teacher ----
    references: tuple = (
        ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    # ---- (label, repo, ckpt, arm_dir|None, dispatch|None) ----
    # Arm locations are EXPLICIT. Earlier versions resolved them through
    # modeling_captionbert.py, which meant the benchmark broke whenever that
    # file was mid-update: BOTH repos currently carry a pre-patch copy that
    # searches amoe/collective/ and amoe/moe/, so -b 404s. A benchmark should
    # not depend on an artifact it is measuring.
    trunks: tuple = (
        ("captionbert-v2", "AbstractPhil/captionbert-8192-v2",
         "checkpoints/best_model.pt", "amoe/collective",
         "amoe/collective/captionbert-v2-collective.dispatch.pt"),
        ("captionbert-b", "AbstractPhil/captionbert-8192-v2-B",
         "checkpoints/final_model.pt", "amoe/b-collective",
         "amoe/b-collective/captionbert-b-arms-native.dispatch.pt"),
    )
    # architecture, so the trunk class is local and needs no remote code
    vocab_size: int = 30522
    d_model: int = 512
    n_heads: int = 12 - 4
    n_layers: int = 12
    d_ff: int = 2048
    output_dim: int = 768
    max_len: int = 8192
    pooling: str = "mean"
    # anchor spec -- the certified campaign defaults every anchor was built with
    n_slots: int = 16
    K: int = 64
    D: int = 4
    tau: float = 0.1
    hidden: int = 178
    gate_init: float = -3.0
    align_emb: int = 64

    tasks: tuple = (
        ("STS-B", "mteb/stsbenchmark-sts"),
        ("SICK-R", "mteb/sickr-sts"),
        ("STS12", "mteb/sts12-sts"),
        ("STS13", "mteb/sts13-sts"),
        ("STS14", "mteb/sts14-sts"),
        ("STS15", "mteb/sts15-sts"),
        ("STS16", "mteb/sts16-sts"),
        ("BIOSSES", "mteb/biosses-sts"),
    )

    batch_size: int = 256
    max_tokens: int = 64
    geom_n: int = 2000
    seed: int = 0

    out_json: str = "full_benchmark.json"
    out_md: str = "benchmark_tables.md"
    hf_push: bool = False
    hf_repos: tuple = ("AbstractPhil/captionbert-8192-v2",
                       "AbstractPhil/captionbert-8192-v2-B")
    hf_path: str = "eval"


CFG = BaseConfig()


def free_model(*objs):
    """Drop refs, collect, empty the cache, and report if VRAM is not coming back."""
    for o in objs:
        try:
            if o is not None and hasattr(o, "to"):
                o.to("cpu")
        except Exception:
            pass
    del objs
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() / 1e9
        if held > 2.0:
            print(f"   [mem] {held:.1f} GB still allocated after teardown -- "
                  f"something is holding a reference")


def line(t=""):
    print("-" * 96 if not t else f"-- {t} " + "-" * max(0, 92 - len(t)))


# ══════════════════════════════════════════════════════════════════
# GAUGES
# ══════════════════════════════════════════════════════════════════

def effective_rank(x):
    xc = (x - x.mean(0, keepdim=True)).double()
    s2 = torch.linalg.svdvals(xc) ** 2
    return float((s2.sum() ** 2 / (s2 ** 2).sum()).item())


@torch.no_grad()
def score(enc, task, cfg):
    a, b, g = task
    ea, eb = enc(a), enc(b)
    cos = F.cosine_similarity(ea, eb, dim=-1).numpy()
    E = torch.cat([ea, eb])
    n = min(cfg.geom_n, E.shape[0])
    S = E[:n] @ E[:n].T
    S.fill_diagonal_(0)
    return {"spearman": float(spearmanr(cos, g).correlation),
            "self_cos": float(S.sum() / (n * n - n)),
            "erank": effective_rank(E[:n])}


def load_tasks(cfg):
    out = {}
    for nm, path in cfg.tasks:
        try:
            d = load_dataset(path, split="test")
            c = d.column_names
            a = "sentence1" if "sentence1" in c else c[0]
            b = "sentence2" if "sentence2" in c else c[1]
            sc = "score" if "score" in c else "similarity_score"
            out[nm] = (list(d[a]), list(d[b]), np.asarray(d[sc], dtype=float))
            print(f"   {nm:8s} {len(out[nm][2]):>6,d} pairs")
        except Exception as e:
            print(f"   {nm:8s} SKIPPED ({type(e).__name__})")
    return out


def hf_encoder(name, cfg):
    """
    Mean-pooled + L2-normalized. The same treatment every teacher gets.

    NOTE the decorator placement. A previous version put @torch.no_grad() on
    THIS function, which only covered from_pretrained -- the returned closure
    ran outside it, built an autograd graph on every batch, and exhausted a
    96 GB card (it failed to allocate 16 MiB). It also made the embeddings
    carry requires_grad, which broke .numpy() downstream. The guard belongs on
    the thing that runs per batch.
    """
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).to(DEVICE).eval()
    for q in mdl.parameters():
        q.requires_grad_(False)

    n_par = sum(q.numel() for q in mdl.parameters())

    @torch.no_grad()
    def enc(texts):
        out = []
        for i in range(0, len(texts), cfg.batch_size):
            t = tok(list(texts[i:i + cfg.batch_size]), max_length=cfg.max_tokens,
                    padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            h = mdl(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            out.append(F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1),
                                   dim=-1).float().cpu())
        return torch.cat(out)
    return enc, n_par, mdl


class CaptionEncoder(torch.nn.Module):
    """Local, key-compatible with every captionbert-v2-family checkpoint."""

    def __init__(self, cfg):
        super().__init__()
        import torch.nn as nn
        d = cfg.d_model
        self.pad_token_id, self.pooling = 0, cfg.pooling
        self.token_emb = nn.Embedding(cfg.vocab_size, d, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, d)
        self.emb_norm = nn.LayerNorm(d)
        self.emb_drop = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers,
                                             enable_nested_tensor=False)
        self.output_proj = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, cfg.output_dim))

    def forward(self, input_ids, attention_mask=None):
        L = input_ids.shape[1]
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.emb_drop(self.emb_norm(self.token_emb(input_ids) + self.pos_emb(pos)))
        kpm = (~attention_mask.bool()) if attention_mask is not None \
            else (input_ids == self.pad_token_id)
        for mod in self.encoder.layers:
            x = mod(x, src_key_padding_mask=kpm)
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        if self.pooling == "cls":
            pooled = x[:, 0]
        else:
            m = (attention_mask.unsqueeze(-1).to(x.dtype) if attention_mask is not None
                 else (~kpm).unsqueeze(-1).to(x.dtype))
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.output_proj(pooled), dim=-1)


def trunk_encoder(cfg, repo, ckpt, tok):
    m = CaptionEncoder(cfg)
    sd = torch.load(hf_hub_download(repo, ckpt), weights_only=True, map_location="cpu")
    m.load_state_dict(sd, strict=True)
    m = m.to(DEVICE).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    n_par = sum(q.numel() for q in m.parameters())

    @torch.no_grad()
    def enc(texts):
        out = []
        for i in range(0, len(texts), cfg.batch_size):
            t = tok(list(texts[i:i + cfg.batch_size]), max_length=cfg.max_tokens,
                    padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            out.append(m(t["input_ids"], t["attention_mask"]).float().cpu())
        return torch.cat(out)
    return enc, n_par, m


def attach_arms(cfg, model, repo, arm_dir, dispatch_path):
    """
    Inline attach: anchors and dispatch come from EXPLICIT paths in `repo`.
    No modeling_captionbert.py, no AMOE_FALLBACKS, nothing that can go stale.
    Returns (dispatch modules, arm names) and leaves every arm enabled.
    """
    import torch.nn as nn
    from amoe.core.adapter import AdapterSpec, RelayPatchwork
    from amoe.core.dispatch import AnchorDispatch, BlockWithDispatch
    from amoe.io.checkpoint import load_anchor, load_dispatch

    dck = load_dispatch(hf_hub_download(repo, dispatch_path))
    names = list(dck.meta.get("anchors", []))
    tau = float(dck.meta.get("tau", cfg.tau))
    cks = [load_anchor(hf_hub_download(repo, f"{arm_dir}/{n}.anchor.pt")) for n in names]
    spec = AdapterSpec(n_slots=cfg.n_slots, K=cfg.K, D=cfg.D, tau=cfg.tau,
                       hidden=cfg.hidden, gate_init=cfg.gate_init, zero_init_head=True)
    layers = list(model.encoder.layers)
    model._orig_layers = layers
    new, disps = [], []
    for i, layer in enumerate(layers):
        stack = nn.ModuleList()
        for ck in cks:
            a = RelayPatchwork(cfg.d_model, spec)
            a.load_state_dict({k[len(f"{i}."):]: v for k, v in ck.adapters.items()
                               if k.startswith(f"{i}.")})
            for q in a.parameters():
                q.requires_grad_(False)
            stack.append(a)
        dp = AnchorDispatch(stack.to(DEVICE), cfg.d_model,
                            emb=int(dck.meta.get("emb", cfg.align_emb)),
                            tau=tau).to(DEVICE)
        with torch.no_grad():
            dp.dispatch.copy_(dck.dispatch[i]["dispatch"].to(DEVICE))
            dp.key_proj.copy_(dck.dispatch[i]["key_proj"].to(DEVICE))
        for q in dp.parameters():
            q.requires_grad_(False)
        disps.append(dp)
        new.append(BlockWithDispatch(layer, dp))
    model.encoder.layers = nn.ModuleList(new)
    return disps, names


def detach_arms(model):
    import torch.nn as nn
    if getattr(model, "_orig_layers", None) is not None:
        model.encoder.layers = nn.ModuleList(model._orig_layers)
        model._orig_layers = None


# ══════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════

def render(rows, tasks, title, params=None):
    tk = list(tasks)
    line(title)
    print(f"   {'model':26s}{'params':>10s}" + "".join(f"{t:>9s}" for t in tk)
          + f"{'mean':>9s}")
    for label, r in rows.items():
        vals = [r[t]["spearman"] for t in tk]
        p = params.get(label) if params else None
        ps = f"{p/1e6:>9.1f}M" if p else f"{'':>10s}"
        print(f"   {label:26s}{ps}" + "".join(f"{v:>9.4f}" for v in vals)
              + f"{np.mean(vals):>9.4f}")


def markdown(rows, tasks, params, note=""):
    tk = list(tasks)
    out = ["| model | params | " + " | ".join(tk) + " | mean |",
           "|---" * (len(tk) + 3) + "|"]
    for label, r in rows.items():
        vals = [r[t]["spearman"] for t in tk]
        p = params.get(label)
        out.append(f"| {label} | {f'{p/1e6:.1f}M' if p else '--'} | "
                   + " | ".join(f"{v:.4f}" for v in vals)
                   + f" | **{np.mean(vals):.4f}** |")
    return "\n".join(out) + ("\n\n" + note if note else "")


# ══════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════

def run(cfg: BaseConfig = CFG):
    print("=" * 96)
    print("CAPTIONBERT FULL BENCHMARK -- one harness, every model, eight tasks")
    print("=" * 96)
    if DEVICE == "cuda":
        print(f"gpu={torch.cuda.get_device_name()} "
              f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
    torch.manual_seed(cfg.seed)
    line("TASKS")
    tasks = load_tasks(cfg)
    if not tasks:
        raise RuntimeError("no tasks loaded")
    tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    rows, params, geom, groups = {}, {}, {}, {"teachers": [], "reference": [],
                                             "trunks": [], "arms": []}

    # ---- teachers ----
    for label, name in cfg.teachers:
        line(f"TEACHER {label}")
        try:
            enc, n_par, mdl = hf_encoder(name, cfg)
            rows[label] = {k: score(enc, v, cfg) for k, v in tasks.items()}
            params[label] = n_par
            groups["teachers"].append(label)
            ref = list(tasks)[0]
            geom[label] = {k: rows[label][ref][k] for k in ("self_cos", "erank")}
            print(f"   {n_par:,} params | {ref} {rows[label][ref]['spearman']:.4f} "
                  f"| self_cos {rows[label][ref]['self_cos']:+.4f} "
                  f"| erank {rows[label][ref]['erank']:.1f}")
            free_model(mdl, enc)
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {str(e)[:110]}")
            free_model(locals().get("mdl"), locals().get("enc"))

    # ---- reference ----
    for label, name in cfg.references:
        line(f"REFERENCE {label}  (contrastive, 1B+ pairs -- different class)")
        try:
            enc, n_par, mdl = hf_encoder(name, cfg)
            rows[label] = {k: score(enc, v, cfg) for k, v in tasks.items()}
            params[label] = n_par
            groups["reference"].append(label)
            ref = list(tasks)[0]
            geom[label] = {k: rows[label][ref][k] for k in ("self_cos", "erank")}
            print(f"   {n_par:,} params | {ref} {rows[label][ref]['spearman']:.4f} "
                  f"| self_cos {rows[label][ref]['self_cos']:+.4f} "
                  f"| erank {rows[label][ref]['erank']:.1f}")
            free_model(mdl, enc)
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {str(e)[:110]}")
            free_model(locals().get("mdl"), locals().get("enc"))

    # ---- trunks, bare and with their OWN arms ----
    for label, repo, ckpt, arm_dir, dispatch_path in cfg.trunks:
        line(f"TRUNK {label}")
        enc, n_par, m = trunk_encoder(cfg, repo, ckpt, tok)
        rows[label] = {k: score(enc, v, cfg) for k, v in tasks.items()}
        params[label] = n_par
        groups["trunks"].append(label)
        ref = list(tasks)[0]
        geom[label] = {k: rows[label][ref][k] for k in ("self_cos", "erank")}
        print(f"   {n_par:,} params | {ref} {rows[label][ref]['spearman']:.4f} "
              f"| self_cos {rows[label][ref]['self_cos']:+.4f} "
              f"| erank {rows[label][ref]['erank']:.1f}")

        if arm_dir:
            try:
                # EXPLICIT paths in THIS trunk's repo. Anchors are trunk-bound:
                # v2's arms on -b cost 31% of their gain, so each trunk gets its own.
                disps, anames = attach_arms(cfg, m, repo, arm_dir, dispatch_path)
                al = f"{label} + arms"
                rows[al] = {k: score(enc, v, cfg) for k, v in tasks.items()}
                params[al] = n_par + sum(p.numel() for d in disps
                                         for a in d.anchors for p in a.parameters())
                groups["arms"].append(al)
                geom[al] = {k: rows[al][ref][k] for k in ("self_cos", "erank")}
                print(f"   + arms {anames} from {repo}/{arm_dir}: "
                      f"{ref} {rows[al][ref]['spearman']:.4f}")
                detach_arms(m)
            except Exception as e:
                msg = str(e)[:110]
                print(f"   arms FAILED: {type(e).__name__}: {msg}")
                if "404" in msg or "NotFound" in type(e).__name__:
                    print(f"   !! 404: check that {repo}/{arm_dir}/ and")
                    print(f"   !! {repo}/{dispatch_path} exist.")
                detach_arms(m)
        free_model(m, enc)

    # ---- tables ----
    order = groups["teachers"] + groups["trunks"] + groups["arms"] + groups["reference"]
    ordered = {k: rows[k] for k in order if k in rows}
    render(ordered, tasks, "FULL BENCHMARK -- every model, mean-pooled, L2-normalized",
           params)

    tk = list(tasks)
    line("READ")
    tmeans = {k: np.mean([rows[k][t]["spearman"] for t in tk])
              for k in groups["teachers"] if k in rows}
    if tmeans:
        bt = max(tmeans, key=tmeans.get)
        print(f"   best teacher: {bt} {tmeans[bt]:.4f} "
              f"({params[bt]/1e6:.1f}M)")
        tot = sum(params[k] for k in tmeans)
        for k in groups["trunks"]:
            if k in rows:
                mv = np.mean([rows[k][t]["spearman"] for t in tk])
                print(f"   {k:26s} {mv:.4f} ({mv-tmeans[bt]:+.4f} vs best teacher) "
                      f"at {params[k]/tot*100:.0f}% of the teachers' combined params")
        for k in groups["arms"]:
            if k in rows:
                mv = np.mean([rows[k][t]["spearman"] for t in tk])
                print(f"   {k:26s} {mv:.4f} ({mv-tmeans[bt]:+.4f} vs best teacher)")
    for k in groups["reference"]:
        if k in rows:
            mv = np.mean([rows[k][t]["spearman"] for t in tk])
            print(f"   {k:26s} {mv:.4f}  <- 1B+ curated pairs, a DIFFERENT class")

    line("GEOMETRY (first task)")
    print(f"   {'model':26s}{'self_cos':>11s}{'erank':>9s}")
    for k in order:
        if k in geom:
            print(f"   {k:26s}{geom[k]['self_cos']:>+11.4f}{geom[k]['erank']:>9.1f}")
    print()
    print("   self_cos is the isotropy gauge: mean-pooled BERT-family embeddings sit")
    print("   in a narrow cone. Low is better and it is the mechanism behind the")
    print("   trunks' advantage -- cosine discriminates poorly inside a cone.")

    # ---- markdown for the cards ----
    note = ("All models mean-pooled and L2-normalized, no task tuning, one harness. "
            "`all-MiniLM-L6-v2` was contrastively trained on 1B+ curated pairs and is "
            "listed for scale, not as a peer.")
    md = ["## Benchmark\n", markdown(ordered, tasks, params, note), "",
          "### Geometry\n",
          "| model | self_cos | erank |", "|---|---|---|"]
    for k in order:
        if k in geom:
            md.append(f"| {k} | {geom[k]['self_cos']:+.4f} | {geom[k]['erank']:.1f} |")
    open(cfg.out_md, "w").write("\n".join(md) + "\n")
    json.dump({"rows": rows, "params": params, "geometry": geom,
               "groups": groups, "config": asdict(cfg)},
              open(cfg.out_json, "w"), indent=2, default=float)
    print(f"\n   wrote {cfg.out_json} and {cfg.out_md} (paste-ready card tables)")

    if cfg.hf_push:
        tokn = os.environ.get("HF_TOKEN")
        if not tokn:
            try:
                from google.colab import userdata
                tokn = userdata.get("HF_TOKEN")
            except Exception:
                tokn = None
        if tokn:
            from huggingface_hub import HfApi
            api = HfApi(token=tokn)
            for r in cfg.hf_repos:
                for f in (cfg.out_json, cfg.out_md):
                    try:
                        api.upload_file(path_or_fileobj=f,
                                        path_in_repo=f"{cfg.hf_path}/{f}",
                                        repo_id=r, commit_message="full benchmark")
                    except Exception as e:
                        print(f"   push {r} failed: {str(e)[:60]}")
            print(f"   pushed to {list(cfg.hf_repos)}")
    return rows


if "get_ipython" in globals() or __name__ == "__main__":
    RESULTS = run(CFG)