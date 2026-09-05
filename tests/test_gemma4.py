import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.gemma4_text import Model, ModelArgs

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
