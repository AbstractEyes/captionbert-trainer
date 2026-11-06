"""Small real-caption/tokenizer integration check; no training or quality claims."""
import json
from pathlib import Path
import urllib.request
import torch
from transformers import AutoTokenizer
from captionbert_rotary.data import CAPTION_REVISION, TOKENIZER_REVISION, sentence_positions, atomic_json
from captionbert_rotary.config import TOKENIZER_REPO, CAPTION_REPO
from captionbert_rotary.inference import SentenceEncoder
from captionbert_rotary.model import build_model


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION,
                                             cache_dir="data/tokenizer-cache")
    tokenizer.save_pretrained("data/tokenizer")
    url = f"https://huggingface.co/datasets/{CAPTION_REPO}/resolve/{CAPTION_REVISION}/captions_000.json"
    with urllib.request.urlopen(url, timeout=60) as stream:
        prefix = stream.read(64000).decode("utf-8")
    decoder, cursor, captions = json.JSONDecoder(), 1, []
    for _ in range(32):
        while prefix[cursor] in " \n\r\t,":
            cursor += 1
        caption, cursor = decoder.raw_decode(prefix, cursor)
        captions.append(caption)
    punctuation = set(tokenizer.convert_tokens_to_ids([".", "!", "?"]))
    boundary = tokenizer("First. Next! Last?", max_length=256, truncation=True)["input_ids"]
    positions, _, _ = sentence_positions(boundary, punctuation)
    assert positions[0].tolist() == [0, 0]
    for i in range(len(boundary) - 1):
        if boundary[i] in punctuation:
            assert positions[i + 1, 0] == 0
            assert positions[i + 1, 1] == positions[i, 1] + 1
    long = tokenizer("A. " * 30, max_length=256, truncation=True)["input_ids"]
    pos, reaches, overflows = sentence_positions(long, punctuation)
    assert reaches and overflows and pos[:, 1].max() == 24
    torch.manual_seed(1)
    encoder = SentenceEncoder(build_model("A2"), tokenizer, device="cuda")
    vectors = encoder.encode(captions, batch_size=8)
    assert vectors.shape == (32, 768) and torch.isfinite(vectors).all()
    error = float((vectors.norm(dim=-1) - 1).abs().max())
    assert error < 1e-3
    report = {"kind": "real_caption_tokenizer_inference_gate", "rows": len(captions),
              "caption_revision": CAPTION_REVISION, "tokenizer_revision": TOKENIZER_REVISION,
              "arm": "A2", "untrained": True, "max_normalization_error": error,
              "embedding_shape": list(vectors.shape), "memory": encoder.guard.report()}
    atomic_json("validation/tokenizer_inference.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
