# ============================================================================
# modeling_captionbert.py -- AbstractPhil/captionbert-8192-v2
#
#   from transformers import AutoModel, AutoTokenizer
#   model = AutoModel.from_pretrained("AbstractPhil/captionbert-8192-v2",
#                                     trust_remote_code=True)
#   tok   = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
#   out   = model(**tok(["a cat on a windowsill"], return_tensors="pt"))
#   emb   = out.pooler_output          # (B, 768) L2-normalized
#
# OR just:  emb = model.encode(["a cat on a windowsill"])
#
# WITH AMOE ARMS (needs amoe-lora; imported lazily so the base model loads
# without it):
#   model.attach_amoe()                # shipped 3-arm collective
#   emb = model.encode([...])          # adapted
#   with model.amoe_off():             # the unsupervised baseline
#       base = model.encode([...])
#   model.set_amoe(["equiv"])          # one arm, DAMPED inside the dispatch
#   model.detach_amoe()                # bit-exact restore, asserted
#
# ---------------------------------------------------------------------------
# BREAKING CHANGE FROM v1 -- READ THIS IF YOU USED geolip-captionbert-8192
#   v1 returned the POOLED 768-d embedding as `last_hidden_state`. That is not
#   the transformers convention and it silently breaks anything expecting token
#   states. v2 follows the convention:
#       last_hidden_state : (B, L, 512)  token states
#       pooler_output     : (B, 768)     L2-normalized embedding  <-- the product
#       embedding         : (B, 768)     alias for pooler_output
#   If you are porting v1 code, `last_hidden_state` -> `pooler_output`.
#
#   v1 also shipped an AlignmentBank. v2 does NOT. Measured on v1: the bank's
#   expert-consistency block varied 0.2% across samples and took 0.23% of its
#   projection energy while anchor distances took 98.70% -- because
#   `back = x @ R.T @ R` is a rotation round-trip and carries no data. Content
#   extensions belong in an AMOE anchor, which this repo ships separately.
# ============================================================================

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


class CaptionBertV2Config(PretrainedConfig):
    model_type = "captionbert_v2"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 512,          # d_model
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        output_dim: int = 768,           # consensus space
        max_position_embeddings: int = 8192,
        hidden_dropout_prob: float = 0.1,
        pad_token_id: int = 0,
        pooling: str = "mean",           # "mean" | "cls"
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.output_dim = output_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.pooling = pooling


class CaptionBertV2Model(PreTrainedModel):
    """
    Standalone caption/sentence encoder distilled from the geometric consensus
    of five BERT-family teachers. No expert models at inference.

    Parameter names are deliberately NOT namespaced under a submodule so that
    the training checkpoint loads unchanged: token_emb, pos_emb, emb_norm,
    encoder.layers.*, output_proj.*.
    """

    config_class = CaptionBertV2Config
    base_model_prefix = "captionbert_v2"
    supports_gradient_checkpointing = True

    def __init__(self, config: CaptionBertV2Config):
        super().__init__(config)
        d = config.hidden_size
        self.token_emb = nn.Embedding(config.vocab_size, d,
                                      padding_idx=config.pad_token_id)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, d)
        self.emb_norm = nn.LayerNorm(d)
        self.emb_drop = nn.Dropout(config.hidden_dropout_prob)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_hidden_layers, enable_nested_tensor=False)
        self.output_proj = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, config.output_dim))
        self.post_init()

    # -- HF plumbing --
    def get_input_embeddings(self):
        return self.token_emb

    def set_input_embeddings(self, value):
        self.token_emb = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else True
        L = input_ids.shape[1]
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.emb_drop(self.emb_norm(self.token_emb(input_ids) + self.pos_emb(pos)))

        kpm = (~attention_mask.bool()) if attention_mask is not None \
            else (input_ids == self.config.pad_token_id)

        hidden = [x] if output_hidden_states else None
        # Iterate the layers directly rather than calling self.encoder(...):
        # nn.TransformerEncoder's fast path inspects layer types, and an AMOE
        # anchor wraps each layer in a BlockWithAdapter that is not a
        # TransformerEncoderLayer. This keeps attach() a drop-in.
        for mod in self.encoder.layers:
            x = mod(x, src_key_padding_mask=kpm)
            if output_hidden_states:
                hidden.append(x)
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)

        if self.config.pooling == "cls":
            pooled = x[:, 0]
        else:
            m = (attention_mask.unsqueeze(-1).to(x.dtype) if attention_mask is not None
                 else (~kpm).unsqueeze(-1).to(x.dtype))
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        embedding = F.normalize(self.output_proj(pooled), dim=-1)

        if not return_dict:
            return (x, embedding) + ((tuple(hidden),) if output_hidden_states else ())
        out = BaseModelOutputWithPooling(
            last_hidden_state=x,                       # (B, L, 512) token states
            pooler_output=embedding,                   # (B, 768) THE PRODUCT
            hidden_states=tuple(hidden) if output_hidden_states else None,
        )
        out.embedding = embedding                      # explicit alias
        return out

    @torch.no_grad()
    def encode(self, texts, tokenizer=None, batch_size: int = 128,
               max_length: int = 256, device=None) -> torch.Tensor:
        """Raw text -> (N, 768) L2-normalized embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        device = device or next(self.parameters()).device
        was_training = self.training
        self.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            t = tokenizer(list(texts[i:i + batch_size]), max_length=max_length,
                          padding=True, truncation=True, return_tensors="pt").to(device)
            out.append(self(**t).pooler_output.float().cpu())
        if was_training:
            self.train()
        return torch.cat(out)


# ---------------------------------------------------------------------------
# AMOE binding -- lets amoe-lora attach anchors to this trunk unmodified.
#
#   import amoe
#   from modeling_captionbert import CaptionBertV2Binding
#   h = amoe.attach(model, "amoe/moe/equiv.anchor.pt",
#                   binding=CaptionBertV2Binding(d=model.config.hidden_size))
#
# amoe's PathBinding would find encoder.layers by dotted path but then read
# model.config.hidden_size -- which works here because this IS a
# PretrainedConfig. The explicit binding is kept for plain-nn.Module use.
# ---------------------------------------------------------------------------

@dataclass
class CaptionBertV2Binding:
    d: int = 512
    name: str = "captionbert_v2"

    def layers(self, model):
        return model.encoder.layers

    def set_layers(self, model, new):
        model.encoder.layers = nn.ModuleList(new)

    def hidden_size(self, model) -> int:
        return int(self.d)


# ---------------------------------------------------------------------------
# AMOE ARMS -- attach / toggle / detach, from the same AutoModel object.
#
#   model = AutoModel.from_pretrained(REPO, trust_remote_code=True)
#   model.attach_amoe()                       # shipped 3-arm collective
#   emb = model.encode(["a cat on a windowsill"])
#   with model.amoe_off():                    # the unsupervised baseline
#       base = model.encode(["a cat on a windowsill"])
#   model.set_amoe(["equiv"])                 # one arm inside the dispatch
#   model.detach_amoe()                       # bit-exact restore, asserted
#
# `amoe` is imported LAZILY: this file is executed by every
# AutoModel.from_pretrained(trust_remote_code=True), and the base model must
# load on a machine that has never heard of amoe-lora.
#
# Masking never renormalizes (the damping law), so a single arm inside the
# dispatch is DAMPED and reads lower than that arm trained alone. That is the
# mechanism, not a bug -- see the model card.
# ---------------------------------------------------------------------------

# AMOE_REPO defaults to THE REPO THIS MODEL WAS LOADED FROM, not a hardcoded
# name. Anchors are TRUNK-BOUND: measured 2026-08-02, the captionbert-8192-v2
# arms lose 31% of their gain on the -B trunk (.7287 -> .6863), and re-aligning
# the routing keys alone recovers only 29% of that. Retrained natively they
# reach .7294. So a hardcoded arms repo silently hands a trunk the wrong arms.
AMOE_REPO = None            # None -> config._name_or_path
AMOE_LIB = "amoe/arms"
# dispatch checkpoints are searched in order; the first that exists wins
AMOE_DISPATCH_CANDIDATES = (
    "amoe/b-collective/captionbert-b-arms-native.dispatch.pt",
    "amoe/collective/captionbert-v2-collective.dispatch.pt",
)
AMOE_DEFAULT_DISPATCH = None   # None -> first candidate present in the repo
AMOE_FALLBACKS = lambda n: [f"amoe/b-collective/{n}.anchor.pt",
                            f"amoe/collective/{n}.anchor.pt",
                            f"amoe/moe/{n}.anchor.pt"]
_AMOE_HINT = ("amoe-lora is required for arms: "
              "pip install git+https://github.com/AbstractEyes/amoe-lora")


def _amoe():
    try:
        from amoe.core.adapter import AdapterSpec, RelayPatchwork
        from amoe.core.dispatch import AnchorDispatch, BlockWithDispatch
        from amoe.io.checkpoint import load_anchor, load_dispatch
    except ImportError as e:
        raise ImportError(_AMOE_HINT) from e
    return SimpleNamespace(AdapterSpec=AdapterSpec, RelayPatchwork=RelayPatchwork,
                           AnchorDispatch=AnchorDispatch,
                           BlockWithDispatch=BlockWithDispatch,
                           load_anchor=load_anchor, load_dispatch=load_dispatch)


class _ArmsOff:
    def __init__(self, model):
        self.m = model

    def __enter__(self):
        self._prev = [list(d.enabled) for d in self.m._amoe_dispatches]
        for d in self.m._amoe_dispatches:
            d.enabled = [False] * len(d.enabled)
        return self.m

    def __exit__(self, *exc):
        for d, p in zip(self.m._amoe_dispatches, self._prev):
            d.enabled = list(p)


def _attach_amoe(self, arms=None, dispatch=None, repo=None, tau=None, verify=True):
    """
    Attach AMOE arms under a trained dispatch.

    arms      list of arm names. None -> the dispatch checkpoint's own roster,
              in ITS order (the routing keys are per-arm and positional, so the
              order is not cosmetic).
    dispatch  path in `repo`, a local file, or None for an untrained dispatch.
    verify    assert that all-arms-disabled reproduces the bare trunk.
    """
    from huggingface_hub import hf_hub_download, HfApi
    A = _amoe()
    if getattr(self, "_amoe_original_layers", None) is not None:
        raise RuntimeError("arms already attached; call detach_amoe() first")

    # resolve the repo: this model's own, unless told otherwise
    if repo is None:
        repo = AMOE_REPO or getattr(self.config, "_name_or_path", None)
    if not repo:
        raise ValueError("cannot determine the arms repo; pass repo=...")
    # resolve the dispatch: first candidate actually present in THAT repo
    if dispatch is None:
        dispatch = AMOE_DEFAULT_DISPATCH
    if dispatch is None:
        try:
            present = set(HfApi().list_repo_files(repo))
        except Exception:
            present = set()
        dispatch = next((c for c in AMOE_DISPATCH_CANDIDATES if c in present), None)
        if dispatch is None:
            local = os.path.isdir(str(repo))
            hint = (f"'{repo}' is a LOCAL DIRECTORY, so there is no hub repo to "
                    f"search. Pass the repo explicitly:\n"
                    f"    model.attach_amoe(repo='AbstractPhil/captionbert-8192-v2-B')"
                    if local else
                    f"Pass dispatch=<path in {repo}>, or dispatch=False to attach "
                    f"an UNTRAINED dispatch.")
            raise FileNotFoundError(
                f"no dispatch found in {repo}. Tried "
                f"{list(AMOE_DISPATCH_CANDIDATES)}.\n         {hint}")
        print(f"[amoe] dispatch: {repo}/{dispatch}")
    if dispatch is False:
        dispatch = None

    dck = None
    if dispatch is not None:
        p = dispatch if os.path.exists(dispatch) else hf_hub_download(repo, dispatch)
        dck = A.load_dispatch(p)
        if arms is None:
            arms = list(dck.meta.get("anchors", []))
        if tau is None:
            tau = float(dck.meta.get("tau", 0.1))
    if not arms:
        raise ValueError("no arms given and the dispatch names none")
    tau = 0.1 if tau is None else tau

    cks, resolved = [], []
    for name in arms:
        if os.path.exists(str(name)):
            path = str(name)
        else:
            # amoe/arms/ is the canonical library, but it is published by a
            # consolidation step that may not have run yet. Fall back to the
            # campaign folders the anchors were originally trained into, and
            # say which one answered.
            path = None
            for cand in ([f"{AMOE_LIB}/{name}.anchor.pt"] + AMOE_FALLBACKS(name)):
                try:
                    path = hf_hub_download(repo, cand)
                    resolved.append(cand)
                    break
                except Exception:
                    continue
            if path is None:
                raise FileNotFoundError(
                    f"arm '{name}' not found in {repo}: tried {AMOE_LIB}/ and "
                    f"{AMOE_FALLBACKS(name)}. Pass an explicit path, or publish "
                    f"the arm library.")
        cks.append(A.load_anchor(path))
    if resolved and not all(r.startswith(AMOE_LIB) for r in resolved):
        print(f"[amoe] resolved outside {AMOE_LIB}: "
              f"{[r for r in resolved if not r.startswith(AMOE_LIB)]}")

    d = self.config.hidden_size
    layers = list(self.encoder.layers)
    self._amoe_original_layers = layers
    dev = next(self.parameters()).device
    new, disps = [], []
    for i, layer in enumerate(layers):
        stack = nn.ModuleList()
        for ck in cks:
            a = A.RelayPatchwork(d, A.AdapterSpec())
            a.load_state_dict({k[len(f"{i}."):]: v for k, v in ck.adapters.items()
                               if k.startswith(f"{i}.")})
            for q in a.parameters():
                q.requires_grad_(False)
            stack.append(a)
        dp = A.AnchorDispatch(stack.to(dev), d,
                              emb=int(dck.meta.get("emb", 64)) if dck else 64,
                              tau=tau).to(dev)
        if dck is not None:
            with torch.no_grad():
                dp.dispatch.copy_(dck.dispatch[i]["dispatch"].to(dev))
                dp.key_proj.copy_(dck.dispatch[i]["key_proj"].to(dev))
        for q in dp.parameters():
            q.requires_grad_(False)
        disps.append(dp)
        new.append(A.BlockWithDispatch(layer, dp))
    self.encoder.layers = nn.ModuleList(new)
    self._amoe_dispatches = disps
    self._amoe_names = list(arms)

    if verify:
        ids = (torch.arange(8, device=dev).unsqueeze(0) % 7 + 1)
        am = torch.ones_like(ids)
        was = self.training
        self.eval()
        wrapped_layers = self.encoder.layers
        try:
            with torch.no_grad():
                with self.amoe_off():
                    off = self(input_ids=ids,
                               attention_mask=am).pooler_output.clone()
                # temporarily restore the bare stack for the reference forward;
                # try/finally so a raise here cannot strand the model holding
                # unwrapped layers while _amoe_dispatches still points at arms.
                self.encoder.layers = nn.ModuleList(layers)
                pure = self(input_ids=ids,
                            attention_mask=am).pooler_output.clone()
        finally:
            self.encoder.layers = wrapped_layers
            if was:
                self.train()
        gap = (off - pure).abs().max().item()
        if gap > 1e-6:
            raise RuntimeError(
                f"TOGGLE LAW VIOLATED: arms disabled shift the trunk by {gap:.3e}. "
                "Do not trust any on/off comparison from this attachment.")
    return self


def _set_amoe(self, arms):
    """Enable a subset by name (or a boolean mask in attach order)."""
    if getattr(self, "_amoe_dispatches", None) is None:
        raise RuntimeError("no arms attached")
    if arms is None:
        mask = [True] * len(self._amoe_names)
    elif all(isinstance(x, bool) for x in arms):
        mask = list(arms)
    else:
        unknown = [a for a in arms if a not in self._amoe_names]
        if unknown:
            raise ValueError(f"unknown arms {unknown}; have {self._amoe_names}")
        mask = [n in arms for n in self._amoe_names]
    for d in self._amoe_dispatches:
        d.enabled = list(mask)
    return self


def _amoe_off(self):
    if getattr(self, "_amoe_dispatches", None) is None:
        raise RuntimeError("no arms attached")
    return _ArmsOff(self)


def _detach_amoe(self, verify=True):
    """Restore the bare trunk. Bit-exact by construction; asserted by default."""
    orig = getattr(self, "_amoe_original_layers", None)
    if orig is None:
        return self
    dev = next(self.parameters()).device
    if verify:
        was = self.training
        self.eval()
        ids = (torch.arange(8, device=dev).unsqueeze(0) % 7 + 1)
        am = torch.ones_like(ids)
        with torch.no_grad():
            with self.amoe_off():
                off = self(input_ids=ids, attention_mask=am).pooler_output.clone()
        self.encoder.layers = nn.ModuleList(orig)
        with torch.no_grad():
            pure = self(input_ids=ids, attention_mask=am).pooler_output.clone()
        if was:
            self.train()
        gap = (off - pure).abs().max().item()
        if gap > 1e-6:
            raise RuntimeError(f"detach is not bit-exact: {gap:.3e}")
    else:
        self.encoder.layers = nn.ModuleList(orig)
    self._amoe_original_layers = None
    self._amoe_dispatches = None
    self._amoe_names = None
    return self


def _amoe_arms(self):
    """Names of the attached arms, in dispatch order. Empty if none."""
    return list(getattr(self, "_amoe_names", None) or [])


CaptionBertV2Model.attach_amoe = _attach_amoe
CaptionBertV2Model.set_amoe = _set_amoe
CaptionBertV2Model.amoe_off = _amoe_off
CaptionBertV2Model.detach_amoe = _detach_amoe
CaptionBertV2Model.amoe_arms = property(_amoe_arms)

CaptionBertV2Config.register_for_auto_class()
CaptionBertV2Model.register_for_auto_class("AutoModel")
