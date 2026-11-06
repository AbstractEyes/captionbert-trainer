"""Inspect 4096 real caption rows before running the structured campaign."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from captionbert_rotary.config import CAPTION_REPO, TOKENIZER_REPO
from captionbert_rotary.data import CAPTION_REVISION, TOKENIZER_REVISION, atomic_json
from captionbert_rotary.structured_data import tokenizer_metadata, exact_word_targets
from captionbert_rotary.modeling_captionbert import word_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    source = hf_hub_download(CAPTION_REPO, "captions_000.json", repo_type="dataset",
        revision=CAPTION_REVISION, cache_dir=root / "hub-cache")
    captions = json.loads(Path(source).read_text(encoding="utf-8"))
    sample = captions[:4096]
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION, cache_dir=root / "hub-cache")
    tok.save_pretrained(root / "tokenizer")
    vocabulary, continuation_ids = tokenizer_metadata(tok)
    continuation = torch.zeros(len(tok), dtype=torch.bool)
    continuation[continuation_ids] = True
    special = torch.zeros_like(continuation)
    special[tok.all_special_ids] = True
    lengths, sentence_counts = [], []
    rows_with_queries = queries = multi_piece_queries = hash_mismatches = 0
    punct = set(tok.convert_tokens_to_ids([".", "!", "?"]))
    for i in range(0, len(sample), 64):
        batch = tok(sample[i:i + 64], max_length=256, truncation=True, padding=True, return_tensors="pt")
        ids, mask = batch["input_ids"], batch["attention_mask"].bool()
        phase, active, ends, records = word_features(ids, mask, continuation, special)
        identity = phase[..., :2]
        for row in range(len(ids)):
            n = int(mask[row].sum())
            values = ids[row, :n].tolist()
            labels = exact_word_targets(values, vocabulary, set(tok.all_special_ids))
            lengths.append(n)
            sentence_counts.append(1 + sum(t in punct for t in values))
            count = int((labels >= 0).sum())
            queries += count
            rows_with_queries += int(count > 0)
            for position in np.flatnonzero(labels >= 0):
                record = int(labels[position])
                if not records[row, record] or not torch.equal(phase[row, position], phase[row, record]):
                    hash_mismatches += 1
                multi_piece_queries += int(bool(continuation[ids[row, position]]))
        print(f"Data census {min(i + 64, len(sample))}/{len(sample)}", flush=True)
    result = dict(caption_revision=CAPTION_REVISION, tokenizer_revision=TOKENIZER_REVISION,
        source_chunk_rows=len(captions), inspected_rows=len(sample), sampling="first4096, descriptive only",
        token_length_quantiles={str(q): float(np.quantile(lengths, q)) for q in (0, .5, .9, .99, 1)},
        above128=sum(n > 128 for n in lengths), reached256=sum(n == 256 for n in lengths),
        sentence_count_quantiles={str(q): float(np.quantile(sentence_counts, q)) for q in (.5, .9, .99, 1)},
        rows_with_pointer_targets=rows_with_queries, supervised_query_positions=queries,
        continuation_piece_queries=multi_piece_queries, exact_label_phase_or_record_mismatches=hash_mismatches,
        training_target_rule="exact WordPiece tuple; first ending is record; repeated content words only")
    atomic_json(root / "census.json", result)
    print(json.dumps(result), flush=True)
    return int(hash_mismatches != 0)


if __name__ == "__main__":
    raise SystemExit(main())
