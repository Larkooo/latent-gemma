import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import gemma3_text, gemma4_text

from latent_gemma.curriculum import StageConfig, accumulated_gradient
from latent_gemma.data import (
    Example,
    carried_values,
    encode_example,
    removed_step_result,
    weight_carried_values,
)
from latent_gemma.model import AdapterConfig, LatentModel, clone_cache, token_loss


class CharTokenizer:
    """Character tokens make decoded spans exact and easy to reason about."""

    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "Q:" + messages[0]["content"] + "\n"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i)


EXAMPLE = Example(
    "x",
    "gsm8k",
    "Natalia sold 48 clips in April and half as many in May. How many in total?",
    "In May she sold 48 / 2 = 24 clips.\nIn total she sold 48 + 24 = 72 clips.",
    "72",
    "train",
)


def test_carried_values_and_removed_result_follow_reuse():
    assert carried_values(EXAMPLE, 1) == ["24"]
    assert removed_step_result(EXAMPLE, 1) == "24"
    assert carried_values(EXAMPLE, 0) == []
    assert removed_step_result(EXAMPLE, 0) is None
    # Removing everything leaves only the answer as a reuse target.
    assert carried_values(EXAMPLE, None) == ["72"]
    assert removed_step_result(EXAMPLE, None) == "72"


def test_weighting_marks_first_supervised_mention_only():
    tokenizer = CharTokenizer()
    prompt, ids, mask = encode_example(tokenizer, EXAMPLE, "hybrid", 1, "reasoning")
    weighted = encode_example(
        tokenizer, EXAMPLE, "hybrid", 1, "reasoning", carried_value_weight=3.0
    )[2]
    text = tokenizer.decode(ids)
    start = text.index("24")
    assert [i for i, w in enumerate(weighted) if w == 3.0] == [start, start + 1]
    assert all(w in (0.0, 1.0) for i, w in enumerate(weighted) if i not in (start, start + 1))
    # "2" inside "72" or "24" must not match a value of "2"; the value "72"
    # is matched at its own first supervised mention.
    marked = weight_carried_values(tokenizer, ids, mask, ["2"], 5.0)
    assert marked == mask
    assert encode_example(tokenizer, EXAMPLE, "hybrid", 1, "reasoning")[2] == mask


@pytest.fixture
def small():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(5)
    args = gemma3_text.ModelArgs(
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
    yield LatentModel(gemma3_text.Model(args), AdapterConfig(num_layers=2, rank=4, bridge_rank=8))
    mx.set_default_device(previous)


@pytest.fixture
def gemma4():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(12)
    args = gemma4_text.ModelArgs(
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
    yield LatentModel(gemma4_text.Model(args), AdapterConfig(num_layers=2, rank=4, bridge_rank=8))
    mx.set_default_device(previous)


def test_cloned_cache_branch_never_touches_the_original(small):
    prompt = mx.array([[2, 3, 4, 5]])
    state, cache = small.prefill(prompt, 2)
    before = [(c.keys, c.values, c.offset) for c in cache]
    mx.eval(before)
    branch = clone_cache(cache)
    small.hidden(mx.array([[7, 8, 9]]), cache=branch)
    for (keys, values, offset), c in zip(before, cache, strict=True):
        assert c.offset == offset
        np.testing.assert_array_equal(np.array(c.keys), np.array(keys))
        np.testing.assert_array_equal(np.array(c.values), np.array(values))
    assert all(c.offset == offset + 3 for c in branch)


@pytest.mark.parametrize("fixture", ["small", "gemma4"])
def test_auxiliary_branch_adds_value_loss_without_changing_text_loss(fixture, request):
    model = request.getfixturevalue(fixture)
    prompt = mx.array([[2, 3, 4, 5]])
    continuation = mx.array([[10, 11, 12, 13]])
    mask = mx.array([[0.0, 1.0, 1.0, 1.0]])
    value = mx.array([[20, 21]])
    plain = token_loss(model, prompt, continuation, mask, 2)
    combined = token_loss(model, prompt, continuation, mask, 2, "none", value, 0.5)
    state, cache = model.prefill(prompt, 2)
    logits = model.continuation_logits(state, cache, value).astype(mx.float32)
    expected = mx.mean(nn.losses.cross_entropy(logits, value, reduction="none"))
    mx.eval(plain, combined, expected)
    np.testing.assert_allclose(combined.item(), plain.item() + 0.5 * expected.item(), rtol=1e-5)
    # A zero weight or a missing value reproduces the original objective exactly.
    for extra in ((value, 0.0), (None, 0.5)):
        same = token_loss(model, prompt, continuation, mask, 2, "none", *extra)
        np.testing.assert_allclose(same.item(), plain.item())
    grads = nn.value_and_grad(model, token_loss)(
        model, prompt, continuation, mask, 2, "none", value, 0.5
    )[1]
    flat = dict(tree_flatten(grads))
    assert all(np.isfinite(np.array(g)).all() for g in flat.values())
    assert any(np.abs(np.array(g)).sum() > 0 for k, g in flat.items() if k.startswith("bridge"))


def test_accumulation_uses_value_records_only_when_weighted(small):
    class Tokenizer:
        pad_token_id = 0

    records = [([2, 3], [4, 5, 6], [0.0, 1.0, 1.0], [7, 8]), ([2, 3], [9, 10], [1.0, 1.0], None)]
    loss_off, _, tokens = accumulated_gradient(small, Tokenizer(), records, 2)
    loss_on, _, _ = accumulated_gradient(small, Tokenizer(), records, 2, 1.0)
    assert tokens == 4
    assert loss_on > loss_off


def test_stage_config_rejects_bad_weights():
    StageConfig(carried_value_weight=2.0, value_aux_weight=0.0).validate()
    with pytest.raises(ValueError):
        StageConfig(carried_value_weight=0.0).validate()
    with pytest.raises(ValueError):
        StageConfig(value_aux_weight=-1.0).validate()
