import json
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.gemma3_text import Model, ModelArgs

from latent_gemma.curriculum import (
    StageConfig,
    accumulated_gradient,
    epoch_indices,
    recover_metrics,
    restore_checkpoint,
    save_checkpoint,
)
from latent_gemma.data import Example, remaining_reasoning
from latent_gemma.model import AdapterConfig, LatentModel, token_loss
from latent_gemma.train import make_batch


@pytest.fixture
def tiny():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(9)
    args = ModelArgs(
        model_type="gemma3_text",
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=2,
        head_dim=8,
        vocab_size=32,
        num_key_value_heads=1,
        sliding_window=8,
        sliding_window_pattern=2,
        query_pre_attn_scalar=8,
    )
    yield LatentModel(Model(args), AdapterConfig(num_layers=2, rank=2, bridge_rank=4))
    mx.set_default_device(device)


def test_epoch_sampling_covers_every_example_and_is_repeatable():
    batches = epoch_indices(19, 42, 0, 8)
    assert sorted(i for batch in batches for i in batch) == list(range(19))
    assert list(map(len, batches)) == [8, 8, 3]
    assert batches == epoch_indices(19, 42, 0, 8)
    assert batches != epoch_indices(19, 42, 1, 8)


def test_accumulation_matches_token_weighted_batch(tiny):
    records = [([2, 3], [4, 5], [0.0, 1.0]), ([6, 7], [8, 9, 10], [1.0, 1.0, 1.0])]
    tokenizer = SimpleNamespace(pad_token_id=0)
    loss, grads, tokens = accumulated_gradient(tiny, tokenizer, records, 2)
    expected_loss, expected_grads = nn.value_and_grad(tiny, token_loss)(
        tiny, *make_batch(records, 0), 2
    )
    assert tokens == 4
    np.testing.assert_allclose(loss, expected_loss.item(), atol=1e-5)
    expected = dict(tree_flatten(expected_grads))
    for key, actual in tree_flatten(grads):
        np.testing.assert_allclose(np.array(actual), np.array(expected[key]), atol=2e-5, rtol=2e-4)


def test_optimizer_checkpoint_resumes_exact_next_update(tiny, tmp_path):
    optimizer = optim.AdamW(learning_rate=2e-4, weight_decay=0.0)
    record = [([2, 3], [4, 5], [1.0, 1.0])]
    tokenizer = SimpleNamespace(pad_token_id=0)

    def update():
        _, grads, _ = accumulated_gradient(tiny, tokenizer, record, 2)
        optimizer.update(tiny, grads)
        mx.eval(tiny.parameters(), optimizer.state)

    # The initial checkpoint must also support an uninitialized optimizer.
    save_checkpoint(tiny, optimizer, tmp_path, {"model_path": "tiny"}, {"step": 0})
    restore_checkpoint(tiny, optimizer, tmp_path)
    update()
    save_checkpoint(tiny, optimizer, tmp_path, {"model_path": "tiny"}, {"step": 1}, best=True)
    update()
    expected = {key: np.array(v) for key, v in tree_flatten(tiny.trainable_parameters())}
    state = restore_checkpoint(tiny, optimizer, tmp_path)
    assert state["step"] == 1
    update()
    for key, actual in tree_flatten(tiny.trainable_parameters()):
        np.testing.assert_array_equal(np.array(actual), expected[key])
    assert json.loads((tmp_path / "last.json").read_text())["checkpoint"].endswith("0000001")


@pytest.mark.parametrize("task", ["gsm8k", "arithmetic", "links"])
def test_final_stage_removes_all_reasoning(task):
    example = Example("x", task, "question", "A -> B.\n2 + 3 = 5.", "5", "train")
    assert remaining_reasoning(example, None) == ""


def test_invalid_stage_configuration_fails():
    for field in ("epochs", "batch_size", "validation_size", "checkpoint_every"):
        with pytest.raises(ValueError):
            replace(StageConfig(), **{field: 0}).validate()


def test_checkpoint_recovers_unreferenced_directory_but_protects_live_checkpoint(tiny, tmp_path):
    orphan = tmp_path / "checkpoints/step-0000000"
    orphan.mkdir(parents=True)
    (orphan / "partial").write_text("interrupted before pointer update")
    optimizer = optim.AdamW(learning_rate=2e-4)
    save_checkpoint(tiny, optimizer, tmp_path, {"model_path": "tiny"}, {"step": 0})
    assert restore_checkpoint(tiny, optimizer, tmp_path)["step"] == 0
    with pytest.raises(FileExistsError, match="referenced checkpoint"):
        save_checkpoint(tiny, optimizer, tmp_path, {"model_path": "tiny"}, {"step": 0})


def test_recovery_preserves_interrupted_metrics_without_duplicating_steps(tmp_path):
    original = '{"step": 10}\n{"step": 20}\n{"step":'
    path = tmp_path / "metrics.jsonl"
    path.write_text(original)
    recover_metrics(tmp_path, 10)
    assert path.read_text() == '{"step": 10}\n'
    archives = list((tmp_path / "interrupted-metrics").glob("*.jsonl"))
    assert len(archives) == 1 and archives[0].read_text() == original
    recover_metrics(tmp_path, 10)
    assert len(list((tmp_path / "interrupted-metrics").glob("*.jsonl"))) == 1
