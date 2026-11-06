"""Pre-spend auxiliary-gradient check on eight real caption/consensus pairs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from captionbert_rotary.config import CAPTION_REPO, TARGET_REPO
from captionbert_rotary.data import CAPTION_REVISION, TARGET_REVISION, atomic_json
from captionbert_rotary.structured_data import tokenizer_metadata, exact_word_targets
from captionbert_rotary.structured_campaign import model_config
from captionbert_rotary.modeling_captionbert import CaptionBertModel
from captionbert_rotary.metrics import objective
from captionbert_rotary.memory import MemoryGuard
from captionbert_rotary.train import seed_all


def cosine(left, right):
    dot = norm_left = norm_right = 0.0
    for a, b in zip(left, right):
        if a is not None and b is not None:
            dot += float((a.double() * b.double()).sum())
            norm_left += float(a.double().square().sum())
            norm_right += float(b.double().square().sum())
    return dot / max((norm_left * norm_right) ** .5, 1e-30), norm_left ** .5, norm_right ** .5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    guard = MemoryGuard(20.0, "cuda")
    caption_path = hf_hub_download(CAPTION_REPO, "captions_000.json", repo_type="dataset", revision=CAPTION_REVISION, cache_dir=root / "hub-cache")
    target_path = hf_hub_download(TARGET_REPO, "consensus_000.pt", repo_type="dataset", revision=TARGET_REVISION, cache_dir=root / "hub-cache")
    captions = json.loads(Path(caption_path).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    vocabulary, _ = tokenizer_metadata(tokenizer)
    chosen, text, labels = [], [], []
    for i, caption in enumerate(captions[:4096]):
        ids = tokenizer(caption, max_length=256, truncation=True)["input_ids"]
        target = exact_word_targets(ids, vocabulary, set(tokenizer.all_special_ids))
        if (target >= 0).sum() >= 2:
            chosen.append(i)
            text.append(caption)
            labels.append(target)
        if len(chosen) == 8:
            break
    batch = tokenizer(text, max_length=256, truncation=True, padding=True, return_tensors="pt")
    targets = torch.full_like(batch["input_ids"], -1)
    for row, value in enumerate(labels):
        targets[row, :len(value)] = torch.from_numpy(value.astype(np.int64))
    bank = torch.load(target_path, map_location="cpu", weights_only=True, mmap=True)
    teacher = bank[chosen].to("cuda")
    seed_all(1)
    model = CaptionBertModel(model_config("KALLP", tokenizer)).cuda().train()
    ids, mask, targets = batch["input_ids"].cuda(), batch["attention_mask"].cuda(), targets.cuda()
    output = model(ids, mask, pointer_targets=targets)
    base, _, _ = objective(output.pooler_output, teacher)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    base_grad = torch.autograd.grad(base, params, retain_graph=True, allow_unused=True)
    pointer_grad = torch.autograd.grad(output.pointer_loss, params, retain_graph=True, allow_unused=True)
    overlap, base_norm, pointer_norm = cosine(base_grad, pointer_grad)
    # Separate caption groups make sure the term is not one constant direction.
    terms = []
    for indices in (slice(0, 4), slice(4, 8)):
        part = model(ids[indices], mask[indices], pointer_targets=targets[indices]).pointer_loss
        terms.append(torch.autograd.grad(part, params, allow_unused=True))
    compartment_cosine, _, _ = cosine(*terms)
    finite = all(g is None or bool(torch.isfinite(g).all()) for group in (base_grad, pointer_grad, *terms) for g in group)
    novelty = 1 - abs(overlap)
    passed = finite and novelty >= .05 and abs(compartment_cosine) <= .8 and pointer_norm > 1e-10
    result = dict(passed=passed, seed=1, rows=chosen, pointer_positions=int((targets >= 0).sum()),
        base_loss=float(base.detach()), pointer_loss=float(output.pointer_loss.detach()),
        gradient_cosine_shared=overlap, novelty=novelty, base_gradient_norm_shared=base_norm,
        pointer_gradient_norm_shared=pointer_norm, weighted_gradient_ratio=.1 * pointer_norm / max(base_norm, 1e-30),
        between_caption_group_gradient_cosine=compartment_cosine, finite=finite,
        conditioning="direct cross-entropy on lookup logits; no ill-conditioned inverse-reconstruction map",
        memory=guard.report(), scope="Initialization apparatus/gradient test, eight real pairs; no optimizer step or accuracy claim")
    atomic_json(root / "gradient-gate.json", result)
    print(json.dumps(result), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
