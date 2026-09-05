"""Differentiable hidden-state feedback without intermediate token sampling.

The backbone performs normal causal attention. Latent positions occupy real KV
cache positions, and every feedback step runs the transformer. No vocabulary
projection is needed until the answer is decoded. This is embedding recurrence,
not recurrent reuse of a subset of layers within a token's forward pass.
"""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.cache import KVCache
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import get_total_parameters, load

LORA_KEYS = (
    "self_attn.q_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True)
class AdapterConfig:
    rank: int = 16
    scale: float = 16.0
    num_layers: int = 0
    bridge_rank: int = 64
    seed: int = 42
    compute_dtype: str = "auto"


class FeedbackBridge(nn.Module):
    """Map normalized output states into the backbone's embedding distribution."""

    def __init__(self, dim: int, rank: int, embedding_rms: float):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.up.weight = mx.zeros_like(self.up.weight)
        self.gain = mx.array(embedding_rms, dtype=mx.float32)

    def __call__(self, state: mx.array) -> mx.array:
        x = state.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + 1e-6)
        x = x + self.up(nn.gelu(self.down(x)))
        return (x * self.gain).astype(state.dtype)


class LatentModel(nn.Module):
    def __init__(self, backbone: nn.Module, config: AdapterConfig):
        super().__init__()
        if backbone.model_type not in {"gemma3_text", "qwen3", "gemma4", "gemma4_text"}:
            raise ValueError(f"Unsupported backbone: {backbone.model_type}")
        self.backbone = backbone
        self.config = config
        if not 0 <= config.num_layers <= len(backbone.layers):
            raise ValueError("Adapter layer count must be between zero and the backbone depth")
        dtype = config.compute_dtype
        if dtype == "auto":
            dtype = "float32" if backbone.model_type.startswith("gemma4") else "original"
        if dtype not in {"original", "float32", "bfloat16"}:
            raise ValueError(f"Unsupported compute dtype: {dtype}")
        if dtype != "original":
            # Quantized weight storage stays quantized. Float32 avoids observed
            # nonfinite backward passes through recurrent Gemma 4 activations.
            backbone.set_dtype(getattr(mx, dtype))
        self.compute_dtype = dtype
        self.hidden_size = self.language_model.args.hidden_size
        # Never guess the embedding scale from hidden size: Gemma additionally
        # scales inputs internally. The bridge returns UNscaled embeddings.
        # Use actual dequantized lookup outputs, including for quantized bases.
        # Stratified vocabulary sampling avoids materializing an enormous table.
        vocab = self.language_model.args.vocab_size
        sample_ids = mx.linspace(0, vocab - 1, min(vocab, 1024)).astype(mx.int32)
        emb = self.language_model.model.embed_tokens(sample_ids)
        embedding_rms = mx.sqrt(mx.mean(mx.square(emb.astype(mx.float32)))).item()
        backbone.freeze()
        if config.num_layers > 0:
            linear_to_lora_layers(
                backbone,
                config.num_layers,
                {
                    "rank": config.rank,
                    "scale": config.scale,
                    "dropout": 0.0,
                    "keys": LORA_KEYS,
                },
            )
        self.bridge = FeedbackBridge(self.hidden_size, config.bridge_rank, embedding_rms)

    def expand_lora(self, num_layers: int) -> None:
        """Add zero-output adapters to earlier layers, preserving trained weights."""
        depth = len(self.backbone.layers)
        if not self.config.num_layers < num_layers <= depth:
            raise ValueError("Expanded adapter layer count must increase and fit the backbone")
        pending = []
        for layer in self.backbone.layers[depth - num_layers : depth - self.config.num_layers]:
            replacements = []
            for name, module in layer.named_modules():
                if name not in LORA_KEYS:
                    continue
                if not isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                    raise ValueError(f"Cannot expand dense LoRA over {type(module).__name__}")
                replacements.append(
                    (
                        name,
                        LoRALinear.from_base(
                            module, r=self.config.rank, scale=self.config.scale, dropout=0.0
                        ),
                    )
                )
            pending.append((layer, replacements))
        # Validate and construct every replacement before mutating the model.
        for layer, replacements in pending:
            layer.update_modules(tree_unflatten(replacements))
        self.config = replace(self.config, num_layers=num_layers)

    @property
    def language_model(self):
        if self.backbone.model_type == "gemma4":
            return self.backbone.language_model
        return self.backbone

    def new_cache(self) -> list[KVCache]:
        # Nonrotating caches retain differentiable history during short training
        # sequences. Gemma still applies its own sliding attention masks.
        shared = getattr(self.language_model.args, "num_kv_shared_layers", 0)
        return [KVCache() for _ in range(len(self.backbone.layers) - shared)]

    def hidden(self, ids: mx.array | None, cache=None, embeddings=None) -> mx.array:
        # Gemma's implementation scales input_embeddings in place. Preserve the
        # caller's unscaled vectors for recomputation and activation inspection.
        if embeddings is not None:
            embeddings = embeddings * 1
        core = self.language_model.model
        ple_dim = getattr(core, "hidden_size_per_layer_input", 0)
        if embeddings is not None and ple_dim:
            # Gemma 4's default embeddings-only path recovers a token via
            # nearest-neighbor lookup. Avoid that discrete bottleneck: latent
            # positions retain the continuous projection branch and use zero
            # contribution from the token-indexed per-layer tables. The native
            # projection retains its original 1/sqrt(2) mixing scale.
            ple = mx.zeros(
                (*embeddings.shape[:2], len(core.layers), ple_dim), dtype=embeddings.dtype
            )
            return core(ids, cache=cache, input_embeddings=embeddings, per_layer_inputs=ple)
        return core(ids, cache=cache, input_embeddings=embeddings)

    def logits(self, hidden: mx.array) -> mx.array:
        lm = self.language_model
        if getattr(lm, "tie_word_embeddings", False) or getattr(
            lm.args, "tie_word_embeddings", False
        ):
            logits = lm.model.embed_tokens.as_linear(hidden)
        else:
            logits = lm.lm_head(hidden)
        softcap = getattr(lm, "final_logit_softcapping", None)
        if softcap is not None:
            logits = mx.tanh(logits / softcap) * softcap
        return logits

    def prefill(self, prompt: mx.array, steps: int, ablation: str = "none"):
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        if ablation not in {"none", "zero", "repeat", "shuffle"}:
            raise ValueError(f"Unknown ablation: {ablation}")
        cache = self.new_cache()
        state = self.hidden(prompt, cache=cache)[:, -1:, :]
        initial = state
        for _ in range(steps):
            feedback = self.bridge(initial if ablation == "repeat" else state)
            if ablation == "zero":
                feedback = mx.zeros_like(feedback)
            elif ablation == "shuffle":
                # Reverse features (also works at batch size one); deterministic
                # corruption without changing position count or vector norm.
                feedback = feedback[..., ::-1]
            state = self.hidden(None, cache=cache, embeddings=feedback)
        return state, cache

    def answer_logits(self, prompt: mx.array, continuation: mx.array, steps: int) -> mx.array:
        """Predict every continuation token; inputs never contain the target at its position."""
        state, cache = self.prefill(prompt, steps)
        if continuation.shape[1] > 1:
            rest = self.hidden(continuation[:, :-1], cache=cache)
            state = mx.concatenate([state, rest], axis=1)
        return self.logits(state)

    def save_adapter(self, directory: Path, metadata: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(self.trainable_parameters()))
        mx.save_safetensors(str(directory / "adapter.safetensors"), weights)
        data = {
            "adapter": asdict(self.config),
            "resolved_compute_dtype": self.compute_dtype,
            **metadata,
        }
        (directory / "config.json").write_text(json.dumps(data, indent=2) + "\n")


def load_model(model_path: str, config: AdapterConfig, adapter_path: Path | None = None):
    mx.random.seed(config.seed)
    backbone, tokenizer = load(model_path)
    model = LatentModel(backbone, config)
    if adapter_path:
        weights = mx.load(str(adapter_path / "adapter.safetensors"))
        expected = dict(tree_flatten(model.trainable_parameters()))
        if weights.keys() != expected.keys():
            raise ValueError("Adapter parameter names do not match its configuration")
        for key in weights:
            if weights[key].shape != expected[key].shape:
                raise ValueError(f"Wrong adapter shape for {key}")
        model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model, tokenizer


def parameter_counts(model: LatentModel) -> dict[str, int]:
    return {
        "total": get_total_parameters(model.backbone)
        + sum(p.size for _, p in tree_flatten(model.bridge.parameters())),
        "trainable": sum(p.size for _, p in tree_flatten(model.trainable_parameters())),
    }


def token_loss(model: LatentModel, prompt, continuation, mask, steps: int):
    logits = model.answer_logits(prompt, continuation, steps).astype(mx.float32)
    loss = nn.losses.cross_entropy(logits, continuation, reduction="none")
    return mx.sum(loss * mask) / mx.maximum(mx.sum(mask), 1)
