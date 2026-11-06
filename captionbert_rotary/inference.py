from dataclasses import asdict
import json
from pathlib import Path
import torch
from .config import ModelConfig
from .data import atomic_json
from .model import build_model
from .benchmark import encode_texts
from .memory import MemoryGuard


class SentenceEncoder:
    def __init__(self, model, tokenizer, device="cpu"):
        self.guard = MemoryGuard(20.0, device)
        self.model, self.tokenizer, self.device = model.to(device).eval(), tokenizer, device

    def encode(self, texts, batch_size=32, max_tokens=256):
        if not 1 <= max_tokens <= 256:
            raise ValueError("This screen was trained with a 256-token ceiling")
        self.guard.check()
        result = encode_texts(self.model, self.tokenizer, list(texts), self.device, batch_size, max_tokens)
        self.guard.check()
        return result

    @classmethod
    def from_pretrained(cls, path_or_repo, run=None, device="cpu"):
        from transformers import AutoTokenizer
        path = Path(path_or_repo)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            if not run:
                raise ValueError("Select an explicit experiments/<config-id>/<arm>-seed<N> Hub run")
            cache = snapshot_download(path_or_repo, allow_patterns=[run.rstrip("/") + "/*"])
            path = Path(cache) / run
        result = json.loads((path / "result.json").read_text())
        model = build_model(result["arm"], ModelConfig(**result["model_config"]))
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
        return cls(model, tokenizer, device)
