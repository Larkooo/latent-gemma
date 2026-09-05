import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.gemma4_text import Model, ModelArgs

from latent_gemma.data import Example
from latent_gemma.evaluate import generate
from latent_gemma.model import AdapterConfig, LatentModel, token_loss


@pytest.fixture
def model():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(12)
    args = ModelArgs(
        hidden_size=32,
        num_hidden_layers=4,
        intermediate_size=64,
        num_attention_heads=2,
        head_dim=16,
        global_head_dim=16,
        vocab_size=64,
        vocab_size_per_layer_input=64,
        num_key_value_heads=1,
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=8,
        sliding_window=8,
        sliding_window_pattern=2,
        use_double_wide_mlp=False,
    )
    yield LatentModel(Model(args), AdapterConfig(num_layers=2, rank=4, bridge_rank=8))
    mx.set_default_device(previous)


def close(a, b):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("mode,steps", [("cot", 0), ("hybrid", 0), ("hybrid", 2), ("latent", 2)])
def test_pipelined_decoding_preserves_gemma4_cached_tokens(model, mode, steps):
    class Tokenizer:
        eos_token_id = 100

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def encode(self, text, **kwargs):
            return [2, 3, 4] if text.startswith("prompt") else [5, 6]

        def decode(self, ids):
            return " ".join(map(str, ids))

    example = Example("test", "links", "Question", "", "A", "validation")
    settings = dict(mode=mode, steps=steps, max_tokens=6, hybrid_boundary="reasoning")
    serial = generate(model, Tokenizer(), example, **settings)
    pipelined = generate(model, Tokenizer(), example, **settings, decode_strategy="pipelined")
    for field in (
        "text",
        "prediction",
        "generated_tokens",
        "token_ids",
        "terminated",
        "forced_tokens",
        "transformer_positions",
        "vocabulary_projections",
    ):
        assert pipelined[field] == serial[field]
    assert pipelined["generated_tokens"] == 6
    assert pipelined["prefetched_text_positions"] == 0


def test_original_path_preserves_ple_shared_cache_and_logit_softcap(model):
    ids = mx.array([[2, 3, 4, 5]])
    state, caches = model.prefill(ids, 0)
    close(model.logits(state), model.backbone(ids)[:, -1:])
    assert len(caches) == 2
    assert all(c.offset == 4 for c in caches)


def test_latent_path_never_recovers_token_ids(model, monkeypatch):
    core = model.language_model.model
    original = core._get_per_layer_inputs

    def guarded(ids, input_embeddings=None):
        assert ids is not None, "Latent states must not be discretized to nearest token IDs"
        return original(ids, input_embeddings)

    monkeypatch.setattr(core, "_get_per_layer_inputs", guarded)
    state, caches = model.prefill(mx.array([[2, 3, 4]]), 3)
    mx.eval(state)
    assert all(c.offset == 6 for c in caches)


def test_cache_matches_recomputed_mixed_token_and_latent_ple(model):
    ids = mx.array([[2, 3, 4, 5, 6, 7, 8]])
    core = model.language_model.model
    embeddings = core.embed_tokens(ids)
    ple = core._get_per_layer_inputs(ids)
    hidden = core(None, input_embeddings=embeddings, per_layer_inputs=ple)[:, -1:]
    for _ in range(3):
        embeddings = mx.concatenate([embeddings, model.bridge(hidden)], axis=1)
        ple = mx.concatenate([ple, mx.zeros((1, 1, 4, 8))], axis=1)
        hidden = core(None, input_embeddings=embeddings, per_layer_inputs=ple)[:, -1:]
    actual, _ = model.prefill(ids, 3)
    close(actual, hidden)


def test_feedback_is_trainable_on_gemma4(model):
    prompt = mx.array([[2, 3, 4]])
    target = mx.array([[5, 6]])
    loss, grads = nn.value_and_grad(model, token_loss)(model, prompt, target, mx.ones((1, 2)), 2)
    assert np.isfinite(loss.item())
    assert mx.sum(mx.abs(grads["bridge"]["up"]["weight"])).item() > 0


def test_target_alignment_and_no_future_leakage_with_shared_kv(model):
    prompt = mx.array([[2, 3, 4]])
    target_a = mx.array([[5, 6, 7, 8]])
    target_b = mx.array([[5, 6, 9, 10]])
    a = model.answer_logits(prompt, target_a, 3)
    b = model.answer_logits(prompt, target_b, 3)
    close(a[:, :3], b[:, :3])
    assert not np.allclose(np.array(a[:, 3]), np.array(b[:, 3]))
    full = model.backbone(mx.concatenate([prompt, target_a[:, :-1]], axis=1))
    close(model.answer_logits(prompt, target_a, 0), full[:, 2:])


def test_cached_feedback_gradients_match_full_recomputation(model):
    ids = mx.array([[2, 3, 4, 5, 6, 7, 8]])

    def objective(candidate, cached):
        if cached:
            hidden, _ = candidate.prefill(ids, 3)
        else:
            core = candidate.language_model.model
            embeddings = core.embed_tokens(ids)
            ple = core._get_per_layer_inputs(ids)
            hidden = core(None, input_embeddings=embeddings, per_layer_inputs=ple)[:, -1:]
            for _ in range(3):
                embeddings = mx.concatenate([embeddings, candidate.bridge(hidden)], axis=1)
                ple = mx.concatenate([ple, mx.zeros((1, 1, 4, 8))], axis=1)
                hidden = core(None, input_embeddings=embeddings, per_layer_inputs=ple)[:, -1:]
        return nn.losses.cross_entropy(candidate.logits(hidden), mx.array([[9]])).mean()

    grad = nn.value_and_grad(model, objective)
    cached_loss, cached = grad(model, True)
    full_loss, full = grad(model, False)
    close(cached_loss, full_loss)
    # Gradients amplify float32 rounding across repeated recomputation. Bound
    # relative vector error rather than relative error at near-zero components.
    for a, b in [
        (cached["bridge"]["up"]["weight"], full["bridge"]["up"]["weight"]),
        (cached["bridge"]["gain"], full["bridge"]["gain"]),
    ]:
        a, b = np.array(a), np.array(b)
        assert np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-8) < 1e-4


def test_expansion_preserves_trained_outputs_and_enables_cache_writer_gradients(model, tmp_path):
    # Existing nonzero updates must survive expansion, not just initial adapters.
    projection = model.backbone.layers[-1].self_attn.q_proj
    projection.lora_b = mx.full_like(projection.lora_b, 0.01)
    old_parameters = {k: np.array(v) for k, v in tree_flatten(model.trainable_parameters())}
    assert not any("v_proj" in k for k in old_parameters)
    prompt = mx.array([[2, 3, 4]])
    before = {steps: np.array(model.logits(model.prefill(prompt, steps)[0])) for steps in (0, 2)}
    model.expand_lora(4)
    assert model.config.num_layers == 4
    after_parameters = dict(tree_flatten(model.trainable_parameters()))
    for key, value in old_parameters.items():
        np.testing.assert_array_equal(value, np.array(after_parameters[key]))
    added = after_parameters.keys() - old_parameters.keys()
    assert any("v_proj.lora_b" in key for key in added)
    assert all(key.endswith(("lora_a", "lora_b")) for key in added)
    for key in added:
        if key.endswith("lora_b"):
            assert mx.all(after_parameters[key] == 0).item()
    for steps, expected in before.items():
        close(model.logits(model.prefill(prompt, steps)[0]), expected)
    target = mx.array([[5, 6]])
    loss, grads = nn.value_and_grad(model, token_loss)(model, prompt, target, mx.ones((1, 2)), 2)
    assert np.isfinite(loss.item())
    cache_writer_grad = grads["backbone"]["model"]["layers"][0]["self_attn"]["v_proj"]["lora_b"]
    assert mx.sum(mx.abs(cache_writer_grad)).item() > 0
    model.save_adapter(tmp_path, {"model_path": "tiny"})
    saved_weights = mx.load(str(tmp_path / "adapter.safetensors"))
    assert saved_weights.keys() == after_parameters.keys()


@pytest.mark.parametrize("layers", [1, 2, 5])
def test_invalid_expansion_leaves_existing_model_intact(model, layers):
    names = {key for key, _ in tree_flatten(model.trainable_parameters())}
    with pytest.raises(ValueError, match="increase and fit"):
        model.expand_lora(layers)
    assert model.config.num_layers == 2
    assert names == {key for key, _ in tree_flatten(model.trainable_parameters())}
