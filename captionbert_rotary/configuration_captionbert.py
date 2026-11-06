"""Self-contained configuration shipped beside the safetensors AutoModel."""
from transformers import PretrainedConfig


class CaptionBertConfig(PretrainedConfig):
    model_type = "captionbert_structured_rotary"

    def __init__(self, vocab_size=30522, hidden_size=512, num_hidden_layers=12,
                 num_attention_heads=16, intermediate_size=2048, output_dim=768,
                 max_position_embeddings=8192, hidden_dropout_prob=0.1,
                 rotary_mode="key_all", lookup_layer=3, lookup_head=0,
                 pointer_weight=0.1, pointer_positions="all", key_rho=2.18,
                 use_value_addresses=True, gradient_checkpointing=True,
                 continuation_ids=None, punctuation_ids=None, special_ids=None,
                 phase_seed=1662197431, pad_token_id=0, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.output_dim = int(output_dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.hidden_dropout_prob = float(hidden_dropout_prob)
        self.rotary_mode = rotary_mode
        self.lookup_layer, self.lookup_head = int(lookup_layer), int(lookup_head)
        self.pointer_weight = float(pointer_weight)
        self.pointer_positions = pointer_positions
        self.key_rho = float(key_rho)
        self.use_value_addresses = bool(use_value_addresses)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.continuation_ids = list(continuation_ids or [])
        self.punctuation_ids = list(punctuation_ids if punctuation_ids is not None else [1012, 999, 1029])
        self.special_ids = list(special_ids if special_ids is not None else [0, 100, 101, 102, 103])
        self.phase_seed = int(phase_seed)
        if rotary_mode not in ("rope", "ntk", "hierarchical", "key_end", "key_all"):
            raise ValueError("Unknown rotary_mode")
        if hidden_size % num_attention_heads or (hidden_size // num_attention_heads) % 2:
            raise ValueError("Attention heads must contain an even number of dimensions")
        if rotary_mode in ("hierarchical", "key_end", "key_all") and hidden_size // num_attention_heads != 32:
            raise ValueError("The source-faithful 8x3 band requires 32-dimensional heads")
        if min(vocab_size, hidden_size, num_hidden_layers, intermediate_size, output_dim) < 1:
            raise ValueError("Dimensions must be positive")
        if not 0 <= lookup_layer < num_hidden_layers or not 0 <= lookup_head < num_attention_heads:
            raise ValueError("Lookup layer/head is outside the model")
        if pointer_positions not in ("end", "all") or pointer_weight < 0 or key_rho < 0:
            raise ValueError("Invalid pointer/norm configuration")

