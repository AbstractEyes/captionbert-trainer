from dataclasses import asdict, dataclass, field
import hashlib
import json

ARMS = ("A0", "A1", "A2", "A3", "A2g")
CAPTION_REPO = "AbstractPhil/conceptual-captions-12m-webdataset-berts"
TARGET_REPO = "AbstractPhil/captionbert-8192-v2-consensus"
TOKENIZER_REPO = "google-bert/bert-base-uncased"
MISSING_CHUNKS = (5, 7, 8, 21, 25, 26, 28, 32, 38, 46)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 30522
    max_len: int = 8192
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    output_dim: int = 768
    dropout: float = 0.1
    pad_token_id: int = 0
    grad_checkpointing: bool = True

    def __post_init__(self):
        if min(self.vocab_size, self.max_len, self.d_model, self.n_heads,
               self.n_layers, self.d_ff, self.output_dim) < 1:
            raise ValueError("Model dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 8:
            raise ValueError("Head dimension must be a multiple of 8 (8|12|12 pair ratio).")


@dataclass(frozen=True)
class ScreenConfig:
    arms: tuple[str, ...] = ("A0", "A1", "A2", "A3")
    seeds: tuple[int, ...] = (1, 2)
    train_chunks: tuple[int, ...] = (0, 1, 2, 3)
    val_chunk: int = 61
    epochs: int = 2
    batch_size: int = 2048
    max_tokens: int = 256
    bucket_window: int = 64
    lr: float = 6e-4
    min_lr: float = 1e-6
    warmup_steps: int = 2000
    temperature: float = 0.07
    grad_clip: float = 1.0
    log_every: int = 25
    eval_every: int = 500
    checkpoint_every: int = 100
    resume_upload_minutes: float = 10.0
    rotary_diagnostics: bool = True
    diagnostic_rows: int = 16
    eval_batch: int = 32
    eval_rows: int = 5000
    max_steps: int | None = None
    memory_limit_gb: float = 20.0
    device: str = "cuda"
    data_dir: str = "data"
    output_dir: str = "runs/screen"
    hub_repo: str = "AbstractPhil/captionbert-8192-rotary"
    push: bool = False
    benchmark: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        if not self.arms or any(a not in ARMS for a in self.arms):
            raise ValueError("Unknown or empty arm selection")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be nonempty and unique")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("Arms must be unique")
        if not self.train_chunks or len(set(self.train_chunks)) != len(self.train_chunks):
            raise ValueError("Training chunks must be nonempty and unique")
        if set(self.train_chunks) & {60, 61} or self.val_chunk in self.train_chunks:
            raise ValueError("Held-out chunks cannot be used in training")
        if set(self.train_chunks) & set(MISSING_CHUNKS):
            raise ValueError("Training must use the v2 five-teacher chunk intersection")
        if any(c < 0 or c > 65 for c in (*self.train_chunks, self.val_chunk)):
            raise ValueError("Chunk indices must be in [0, 65]")
        for name in ("epochs", "batch_size", "max_tokens", "bucket_window", "warmup_steps",
                     "log_every", "eval_every", "checkpoint_every", "eval_batch", "eval_rows"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.max_tokens > min(256, self.model.max_len):
            raise ValueError("The screen token ceiling is 256 and must fit the position table")
        if self.temperature <= 0 or self.memory_limit_gb <= 0:
            raise ValueError("Temperature and memory limit must be positive")
        if self.resume_upload_minutes <= 0 or not 1 <= self.diagnostic_rows <= 16:
            raise ValueError("Resume upload interval must be positive; diagnostic_rows must be 1..16")

    def fingerprint(self):
        raw = asdict(self)
        for k in ("output_dir", "data_dir", "push", "hub_repo", "memory_limit_gb", "device",
                  "resume_upload_minutes", "rotary_diagnostics", "diagnostic_rows"):
            raw.pop(k)
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
