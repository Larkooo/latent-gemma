import json
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.gemma3_text import Model, ModelArgs

from latent_gemma.model import AdapterConfig, LatentModel, token_loss


@pytest.fixture
def model():
    # CPU float32 isolates semantic equivalence from Metal's shape-dependent
    # matrix multiplication rounding (single-token vs sequence kernels).
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(17)
    args = ModelArgs(
        model_type="gemma3_text",
        hidden_size=32,
        num_hidden_layers=4,
        intermediate_size=64,
        num_attention_heads=2,
        head_dim=16,
        vocab_size=64,
        num_key_value_heads=1,
        sliding_window=8,
        sliding_window_pattern=2,
        query_pre_attn_scalar=16,
    )
    backbone = Model(args)
    yield LatentModel(backbone, AdapterConfig(num_layers=2, rank=4, bridge_rank=8))
    mx.set_default_device(previous_device)


def assert_close(a, b, atol=2e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=2e-5)


def test_zero_steps_matches_original_logits(model):
    ids = mx.array([[2, 3, 4, 5]])
    state, cache = model.prefill(ids, 0)
    assert_close(model.logits(state), model.backbone(ids, cache=model.new_cache())[:, -1:])
    assert all(c.offset == ids.shape[1] for c in cache)


def test_cached_feedback_matches_full_sequence_including_sliding_mask(model):
    ids = mx.array([[2, 3, 4, 5, 6, 7, 8]])
    embeddings = model.backbone.model.embed_tokens(ids)
    hidden = model.hidden(None, embeddings=embeddings)[:, -1:]
    for _ in range(3):
        embeddings = mx.concatenate([embeddings, model.bridge(hidden)], axis=1)
        hidden = model.hidden(None, embeddings=embeddings)[:, -1:]
    actual, cache = model.prefill(ids, 3)
    assert_close(actual, hidden)
    assert all(c.offset == 10 for c in cache)


def test_target_alignment_and_no_future_leakage(model):
    prompt = mx.array([[2, 3, 4]])
    target_a = mx.array([[5, 6, 7, 8]])
    target_b = mx.array([[5, 6, 9, 10]])
    a = model.answer_logits(prompt, target_a, 2)
    b = model.answer_logits(prompt, target_b, 2)
    assert_close(a[:, :3], b[:, :3])
    assert not np.allclose(np.array(a[:, 3]), np.array(b[:, 3]))
    full = model.backbone(mx.concatenate([prompt, target_a[:, :-1]], axis=1))
    assert_close(model.answer_logits(prompt, target_a, 0), full[:, 2:])


def test_nonzero_gradients_through_feedback_and_frozen_base(model):
    prompt = mx.array([[2, 3, 4], [7, 8, 9]])
    target = mx.array([[5, 6], [10, 11]])
    mask = mx.ones(target.shape)
    frozen_before = np.array(model.backbone.model.embed_tokens.weight)
    bridge_before = np.array(model.bridge.up.weight)
    loss, grads = nn.value_and_grad(model, token_loss)(model, prompt, target, mask, 2)
    assert np.isfinite(loss.item())
    assert mx.sum(mx.abs(grads["bridge"]["up"]["weight"])).item() > 0
    optim.SGD(learning_rate=0.01).update(model, grads)
    np.testing.assert_array_equal(frozen_before, np.array(model.backbone.model.embed_tokens.weight))
    assert not np.array_equal(bridge_before, np.array(model.bridge.up.weight))


def test_feedback_ablation_changes_computation(model):
    prompt = mx.array([[2, 3, 4]])
    original, _ = model.prefill(prompt, 3)
    corrupted, cache = model.prefill(prompt, 3, "shuffle")
    assert not np.allclose(np.array(original), np.array(corrupted))
    assert all(c.offset == 6 for c in cache)


def test_masked_padding_has_no_effect_on_loss(model):
    prompt = mx.array([[2, 3, 4]])
    short = token_loss(model, prompt, mx.array([[5, 6]]), mx.ones((1, 2)), 2)
    padded = token_loss(
        model, prompt, mx.array([[5, 6, 0, 0]]), mx.array([[1.0, 1.0, 0.0, 0.0]]), 2
    )
    assert_close(short, padded)


def test_adapter_round_trip(model, tmp_path):
    prompt = mx.array([[2, 3, 4]])
    expected, _ = model.prefill(prompt, 2)
    expected = np.array(expected)
    model.save_adapter(tmp_path, {"model_path": "tiny"})
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["adapter"] == asdict(model.config)
    model.bridge.gain = mx.array(9.0)
    model.load_weights(list(mx.load(str(tmp_path / "adapter.safetensors")).items()), strict=False)
    actual, _ = model.prefill(prompt, 2)
    assert_close(actual, expected)
    assert "backbone.model.embed_tokens.weight" not in dict(
        tree_flatten(model.trainable_parameters())
    )


def test_invalid_steps_fail(model):
    with pytest.raises(ValueError, match="nonnegative"):
        model.prefill(mx.array([[2]]), -1)
