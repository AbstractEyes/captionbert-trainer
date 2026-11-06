"""Frozen rotary mechanism measurements; no training or semantic-accuracy claim."""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import atomic_json, sentence_positions
from .model import RotaryEncoder, embed
from .rotary import RopeMulti, bands_for_arm

PROTOCOL = "captionbert-rotary-mechanisms-v1"


def structural_probes():
    sentences = ("The red box is here.", "The blue ball is nearby.",
                 "A small bird watches the street.", "A person walks past the window.")
    counts = (1, 2, 4, 8, 16, 24, 25, 26, 32, 2, 4, 8, 16, 25, 26, 32)
    return [" ".join(sentences[(j + i) % len(sentences)] for j in range(n))
            for i, n in enumerate(counts)]


def band_masks(w, stream):
    return {"content": w == 0, "slot": (w != 0) & (stream == 0),
            "anchor": (w != 0) & (stream == 1)}


def analytic_report(arm, head_dim=64):
    if arm == "A0":
        return {"applicable": False, "reason": "Learned absolute-position replication; no rotary bands"}
    w, stream = bands_for_arm(arm, head_dim)
    masks = band_masks(w, stream)
    bands = {}
    delta = torch.arange(257, dtype=torch.float64)
    for name, mask in masks.items():
        frequencies = w[mask]
        if not len(frequencies):
            bands[name] = {"pairs": 0}
            continue
        kernel = (delta[:, None] * frequencies).cos().mean(-1)
        first_zero = torch.where(kernel <= 0)[0]
        tail_start = int(first_zero[0]) if len(first_zero) else None
        bands[name] = {"pairs": len(frequencies), "frequencies": frequencies.tolist(),
                       "kernel_deltas_0_to_256": kernel.tolist(),
                       "first_nonpositive_integer_delta": tail_start,
                       "max_abs_after_first_nonpositive": float(kernel[tail_start:].abs().max())
                       if tail_start is not None else None}
    anchors = int(masks["anchor"].sum())
    capacity = 2 * anchors + 1 if anchors else None
    if anchors:
        coordinates = torch.arange(capacity, dtype=torch.float64)
        gram = ((coordinates[:, None] - coordinates[None, :])[..., None] * w[masks["anchor"]]).cos().mean(-1)
        off = gram[~torch.eye(capacity, dtype=torch.bool)]
        lifted = (2 * anchors * gram + 1) / capacity
        bands["anchor"].update(code_count=capacity, raw_off_diagonal_min=float(off.min()),
            raw_off_diagonal_max=float(off.max()), raw_coherence=float(off.abs().max()),
            explicit_constant_lift_max_identity_error=float((lifted - torch.eye(capacity)).abs().max()),
            first_periodic_alias=[0, capacity], first_clamp_collision=[capacity - 1, capacity],
            encoder_implements_constant_lift=False)
    return {"applicable": True, "bands": bands,
            "scope": "Unit-content address kernel. The raw anchor code is a regular simplex; "
                     "the explicit orthogonal lift is a mathematical reference, not a fixed learned-q/k coordinate."}


def controlled_lookup(arm, head_dim=64, compartments=8, slots=4):
    if arm == "A0":
        return {"applicable": False, "reason": "A0 has no rotary address code"}
    w, stream = bands_for_arm(arm, head_dim)
    positions = torch.tensor([[s, c] for c in range(compartments) for s in range(slots)])[None]
    address_source = "oracle compartment and in-compartment slot"
    if arm in ("A1", "A2g"):
        positions[..., 0] = torch.arange(compartments * slots)
        address_source = "oracle global target index (layout supplied, not learned)"
    q = torch.ones(1, 1, positions.shape[1], head_dim)
    qr, kr = RopeMulti(w, stream)(q, q, positions, positions)
    scores = (qr.double() @ kr.double().transpose(-1, -2))[0, 0] / head_dim
    maxima = scores.max(-1, keepdim=True).values
    tied = (scores - maxima).abs() <= 2e-6
    correct_in_tie = tied.diag().double()
    success = correct_in_tie / tied.sum(-1)
    return {"applicable": True, "queries": positions.shape[1], "compartments": compartments,
            "slots": slots, "content": "identical all-ones q/k at every address",
            "address_source": address_source, "tie_aware_success": float(success.mean()),
            "mean_maximum_tie_count": float(tied.sum(-1).double().mean()),
            "scope": "Code-only lookup with an oracle request; not trained-model retrieval accuracy."}


def probe_batch(tokenizer, texts, max_tokens=256, max_compartment=24):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
    ids, valid = encoded["input_ids"], encoded["attention_mask"]
    positions = torch.zeros(*ids.shape, 2, dtype=torch.long)
    punctuation = set(tokenizer.convert_tokens_to_ids([".", "!", "?"]))
    for row in range(len(ids)):
        n = int(valid[row].sum())
        positions[row, :n] = torch.from_numpy(sentence_positions(
            ids[row, :n].tolist(), punctuation, max_compartment=max_compartment)[0].astype(np.int64))
    return ids, valid, positions


@contextmanager
def intervention(model, name, structural_positions, raw_positions=None):
    handles = []
    def alter(module, args):
        q, k, pq, pk = args
        def change(p):
            p = p.clone()
            if name == "common_slot_shift":
                p[..., 0] += 7
            elif name == "common_sentence_shift":
                p[..., 1] += 7
            elif name == "sentence_ids_collapsed":
                p[..., 1] = 0
            elif name == "sentence_ids_permuted":
                mapping = torch.tensor(np.random.default_rng(0).permutation(25), device=p.device)
                p[..., 1] = mapping[p[..., 1].long()]
            elif name == "single_sentence_id_shift":
                p[..., 1] += (p[..., 1] == 1).long()
            elif name == "single_sentence_period_alias":
                anchor_pairs = int((module.stream == 1).sum())
                if anchor_pairs:
                    p[..., 1] += (p[..., 1] == 1).long() * (2 * anchor_pairs + 1)
            elif name == "uncapped_sentence_ids":
                p[..., 1] = raw_positions[..., 1]
            elif name == "global_slots":
                p[..., 0] = torch.arange(p.shape[1], device=p.device)
            elif name == "sentence_reset_slots":
                p[..., 0] = structural_positions[..., 0]
            elif name == "slot_phase_removed":
                p[..., 0] = 0
            return p
        return q, k, change(pq), change(pk)

    def lesion(module, args, output):
        qr, kr = output
        mask = band_masks(module.w, module.stream)[name.removeprefix("remove_")]
        dimensions = torch.cat((mask, mask))
        qr, kr = qr.clone(), kr.clone()
        qr[..., dimensions] = 0
        kr[..., dimensions] = 0
        return qr, kr
    try:
        for module in model.modules():
            if isinstance(module, RopeMulti):
                handles.append(module.register_forward_hook(lesion) if name.startswith("remove_")
                               else module.register_forward_pre_hook(alter))
        yield
    finally:
        for handle in handles:
            handle.remove()


def attention_summary(q, k, w, stream, valid, structural, special_ids):
    """Explicit fp32 score reconstruction; never changes the SDPA call's output."""
    q, k = q.float(), k.float()
    content_tokens = valid.bool().clone()
    ids = structural[0]
    for token in special_ids:
        if token is not None:
            content_tokens &= ids != token
    positions = structural[1]
    queries = content_tokens[:, None, :, None]
    query_count = int(content_tokens.sum())
    if not query_count:
        return None
    scale = q.shape[-1] ** -.5
    scores = (q @ k.transpose(-1, -2)) * scale
    allowed = valid[:, None, None, :].bool()
    attention = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    same_sentence = positions[..., 1, None] == positions[:, None, :, 1]
    same_slot = positions[..., 0, None] == positions[:, None, :, 0]
    groups = {"same_sentence": same_sentence,
              "different_sentence": ~same_sentence,
              "same_slot_other_sentence": same_slot & ~same_sentence}
    result = {"queries": query_count, "heads": []}
    bands = band_masks(w, stream)
    per_head = [{} for _ in range(q.shape[1])]
    for group_name, group in groups.items():
        group = group & content_tokens[:, None, :]
        mass = (attention * group[:, None] * queries).sum((0, 2, 3)) / query_count
        chance = ((group.sum(-1).float() / valid.sum(-1)[:, None]) * content_tokens).sum() / query_count
        for head in range(q.shape[1]):
            per_head[head][group_name + "_mass"] = float(mass[head])
            per_head[head][group_name + "_uniform_opportunity"] = float(chance)
    for head in per_head:
        head["special_key_mass"] = 1 - head["same_sentence_mass"] - head["different_sentence_mass"]
    pairs = ((content_tokens[:, :, None] & valid[:, None, :].bool())[:, None]).float()
    pair_count = pairs.sum()
    total_rms = (scores.square() * pairs).sum((0, 2, 3)).div(pair_count).sqrt()
    q_energy = (q.square() * content_tokens[:, None, :, None]).sum((0, 2, 3))
    k_energy = (k.square() * content_tokens[:, None, :, None]).sum((0, 2, 3))
    for band, pair_mask in bands.items():
        dims = torch.cat((pair_mask, pair_mask))
        band_scores = (q[..., dims] @ k[..., dims].transpose(-1, -2)) * scale
        removed = (scores - band_scores).masked_fill(~allowed, -torch.inf).softmax(-1)
        delta = ((attention - removed).abs() * queries).sum((0, 2, 3)) / query_count
        rms = ((band_scores.square() * pairs).sum((0, 2, 3)) / pair_count).sqrt()
        qe = (q[..., dims].square() * content_tokens[:, None, :, None]).sum((0, 2, 3))
        ke = (k[..., dims].square() * content_tokens[:, None, :, None]).sum((0, 2, 3))
        for head in range(q.shape[1]):
            per_head[head][band + "_pairs"] = int(pair_mask.sum())
            per_head[head][band + "_logit_rms"] = float(rms[head])
            per_head[head][band + "_logit_rms_over_total"] = float(rms[head] / total_rms[head]) if total_rms[head] else None
            per_head[head][band + "_q_energy_fraction"] = float(qe[head] / q_energy[head]) if q_energy[head] else None
            per_head[head][band + "_k_energy_fraction"] = float(ke[head] / k_energy[head]) if k_energy[head] else None
            per_head[head][band + "_removal_attention_l1"] = float(delta[head])
    result["heads"] = per_head
    return result


def embedding_change(reference, current):
    if not torch.isfinite(reference).all() or not torch.isfinite(current).all():
        return {"finite": False, "mean_cosine_to_original": None, "mean_l2_change": None,
                "max_absolute_change": None}
    difference = current.float() - reference.float()
    return {"mean_cosine_to_original": float(F.cosine_similarity(reference.float(), current.float()).mean()),
            "mean_l2_change": float(difference.norm(dim=-1).mean()),
            "max_absolute_change": float(difference.abs().max())}


@torch.no_grad()
def run_rotary_diagnostics(model, tokenizer, arm, cfg, output_path=None, identity=None, rows=16):
    started = time.perf_counter()
    was_training = model.training
    model.eval()
    texts = structural_probes()[:rows]
    if not texts:
        raise ValueError("Diagnostic probe count must be positive")
    special = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    layer_batches = {}
    ids_all, mask_all, positions_all = probe_batch(tokenizer, texts)
    _, _, raw_all = probe_batch(tokenizer, texts, max_compartment=32767)
    batch_size = min(cfg.eval_batch, 4)
    rotary = isinstance(model, RotaryEncoder)
    head_dim = cfg.model.d_model // cfg.model.n_heads

    def collect(name, capture=False):
        values = []
        for start in range(0, len(texts), batch_size):
            ids = ids_all[start:start + batch_size].to(cfg.device)
            valid = mask_all[start:start + batch_size].to(cfg.device)
            positions = positions_all[start:start + batch_size].to(cfg.device)
            raw_positions = raw_all[start:start + batch_size].to(cfg.device)
            handles = []
            if capture and rotary:
                for number, layer in enumerate(model.encoder.layers):
                    def observe(module, args, output, number=number):
                        row = attention_summary(*output, module.w, module.stream, valid,
                                                (ids, positions), special)
                        if row:
                            layer_batches.setdefault(number, []).append(row)
                    handles.append(layer.rope.register_forward_hook(observe))
            try:
                with intervention(model, name, positions, raw_positions), torch.autocast(
                        "cuda", dtype=torch.float16, enabled=str(cfg.device).startswith("cuda")):
                    values.append(embed(model, ids, valid, positions).float().cpu())
            finally:
                for handle in handles:
                    handle.remove()
        return torch.cat(values)

    try:
        print(f"Rotary diagnostics {arm}: {len(texts)} fixed structural probes; batch {batch_size}; no training", flush=True)
        baseline = collect("baseline", capture=True)
        repeat = embedding_change(baseline, collect("baseline"))
        names = ("common_slot_shift", "common_sentence_shift", "sentence_ids_collapsed",
                 "sentence_ids_permuted", "single_sentence_id_shift", "single_sentence_period_alias",
                 "uncapped_sentence_ids", "global_slots", "sentence_reset_slots", "slot_phase_removed",
                 "remove_content", "remove_slot", "remove_anchor")
        changes = {}
        variant_started = time.perf_counter()
        for number, name in enumerate(names, 1):
            changes[name] = embedding_change(baseline, collect(name)) if rotary else {
                "applicable": False, "reason": "A0 has no rotary address stream or pair bands"}
            elapsed = time.perf_counter() - started
            eta = (time.perf_counter() - variant_started) * (len(names) - number) / number
            print(f"Rotary diagnostics {number}/{len(names)} {name}; elapsed {elapsed:.1f}s; "
                  f"ETA ~{eta:.1f}s", flush=True)
        layers = []
        for number, batches in sorted(layer_batches.items()):
            count = sum(b["queries"] for b in batches)
            heads = []
            for head in range(cfg.model.n_heads):
                keys = batches[0]["heads"][head]
                heads.append({key: sum(b["heads"][head][key] * b["queries"] for b in batches) / count
                              if all(b["heads"][head][key] is not None for b in batches) else None for key in keys})
            layers.append({"layer": number, "queries": count, "heads": heads})
        finite = bool(torch.isfinite(baseline).all())
        result = {"protocol": PROTOCOL, "identity": identity or {"arm": arm}, "arm": arm,
                  "measurement_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "measurement_environment": {"torch": str(torch.__version__), "python": platform.python_version(),
                                              "device": str(cfg.device), "cpu_threads": torch.get_num_threads(),
                                              "autocast": "float16" if str(cfg.device).startswith("cuda") else "disabled"},
                  "source": "rotary address-kernel / mixed-layout / content-controlled anchor experiments",
                  "analytic": analytic_report(arm, head_dim), "content_controlled_lookup": controlled_lookup(arm, head_dim),
                  "probe_texts": texts, "probe_sha256": hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
                  "probe_tokens": ids_all.tolist(), "probe_positions": positions_all.tolist(),
                  "probe_unclamped_max_sentence_ids": raw_all[..., 1].max(1).values.tolist(),
                  "probe_rows": len(texts), "baseline_finite": finite,
                  "repeat_floor": repeat, "interventions": changes, "layers": layers,
                  "attention_scope": "Explicit fp32 q/k score reconstruction, not native SDPA weights. "
                                     "Layer/head summaries are query-weighted per-batch statistics; no semantic retrieval claim.",
                  "intervention_scope": "Frozen model. Forward hooks change only q/k addresses or zero named q/k pairs; "
                                        "token IDs, masks, values and parameters are held fixed. No AMOE members or key-binding task.",
                  "instrumented_seconds": time.perf_counter() - started,
                  "status": "measured" if finite else "nonfinite_diagnostic"}
        if output_path is not None:
            atomic_json(output_path, result)
        return result
    finally:
        model.train(was_training)
