"""Epoch-based curriculum stages with exact, resumable optimizer checkpoints."""

import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .data import encode_example, read_examples, removed_step_result
from .evaluate import generate
from .model import parameter_counts, token_loss
from .provenance import capture, sha256
from .train import make_batch


@dataclass(frozen=True)
class StageConfig:
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 2e-5
    latent_steps: int = 2
    reasoning_steps_to_drop: int | None = 1
    hybrid_boundary: str = "reasoning"
    seed: int = 42
    validation_size: int = 128
    validation_max_tokens: int = 384
    checkpoint_every: int = 50
    log_every: int = 10
    carried_value_weight: float = 1.0
    value_aux_weight: float = 0.0

    def validate(self):
        for name in (
            "epochs",
            "batch_size",
            "validation_size",
            "validation_max_tokens",
            "checkpoint_every",
            "log_every",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.latent_steps < 0 or (
            self.reasoning_steps_to_drop is not None and self.reasoning_steps_to_drop < 0
        ):
            raise ValueError("Latent and removed reasoning counts must be nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.carried_value_weight) or self.carried_value_weight <= 0:
            raise ValueError("carried_value_weight must be finite and positive")
        if not math.isfinite(self.value_aux_weight) or self.value_aux_weight < 0:
            raise ValueError("value_aux_weight must be finite and nonnegative")


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def epoch_indices(size: int, seed: int, epoch: int, batch_size: int):
    """Every example occurs exactly once, including the final partial batch."""
    indices = list(range(size))
    random.Random(seed + epoch).shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, size, batch_size)]


def checkpoint_path(output: Path, pointer: str = "last") -> Path:
    data = json.loads((output / f"{pointer}.json").read_text())
    path = output / data["checkpoint"]
    if sha256(path / "adapter.safetensors") != data["adapter_sha256"]:
        raise ValueError(f"Corrupt {pointer} checkpoint in {output}")
    return path


def accumulated_gradient(model, tokenizer, records, latent_steps, value_aux_weight=0.0):
    """Microbatch one keeps full-vocabulary activations within laptop memory.

    Weight by supervised tokens, so accumulation matches the concatenated token
    objective even when reasoning lengths differ or the last batch is smaller.
    Records may carry a fourth element: token ids of the removed step's result
    for the auxiliary decoding branch.
    """
    value_and_grad = nn.value_and_grad(model, token_loss)
    total_tokens = sum(sum(record[2]) for record in records)
    if total_tokens <= 0:
        raise ValueError("Training batch has no supervised tokens")
    accumulated = None
    total_loss = 0.0
    for record in records:
        batch = make_batch([record], tokenizer.pad_token_id or 0)
        value = None
        if value_aux_weight > 0 and len(record) > 3 and record[3]:
            value = mx.array([record[3]], dtype=mx.int32)
        loss, grads = value_and_grad(model, *batch, latent_steps, "none", value, value_aux_weight)
        weight = sum(record[2]) / total_tokens
        if accumulated is None:
            accumulated = tree_map(lambda g: g * weight, grads)
        else:
            accumulated = tree_map(lambda a, g: a + g * weight, accumulated, grads)
        mx.eval(loss, accumulated)
        total_loss += loss.item() * weight
    return total_loss, accumulated, int(total_tokens)


def save_checkpoint(model, optimizer, output, metadata, state, *, best=False):
    name = f"step-{state['step']:07d}"
    path = output / "checkpoints" / name
    temporary = path.with_name(name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_adapter(temporary, {**metadata, "step": state["step"]})
    mx.save_safetensors(
        str(temporary / "optimizer.safetensors"), dict(tree_flatten(optimizer.state))
    )
    atomic_json(temporary / "training-state.json", state)
    # Recovery only trusts the atomic pointers. A crash after the directory
    # rename but before the pointer update can leave an unreferenced directory.
    if path.exists():
        referenced = {
            checkpoint_path(output, pointer)
            for pointer in ("last", "best")
            if (output / f"{pointer}.json").exists()
        }
        if path in referenced:
            raise FileExistsError(f"Refusing to replace a referenced checkpoint: {path}")
        shutil.rmtree(path)
    os.replace(temporary, path)
    pointer = {
        "checkpoint": str(path.relative_to(output)),
        "adapter_sha256": sha256(path / "adapter.safetensors"),
    }
    atomic_json(output / "last.json", pointer)
    if best:
        atomic_json(output / "best.json", pointer)
    retained = {checkpoint_path(output)}
    if (output / "best.json").exists():
        retained.add(checkpoint_path(output, "best"))
    # Only this stage's superseded checkpoints are removed; metrics stay intact.
    for previous in (output / "checkpoints").glob("step-*"):
        if previous.is_dir() and not previous.name.endswith(".tmp") and previous not in retained:
            shutil.rmtree(previous)
    return path


def restore_checkpoint(model, optimizer, output):
    path = checkpoint_path(output)
    model.load_weights(list(mx.load(str(path / "adapter.safetensors")).items()), strict=False)
    optimizer.state = tree_unflatten(list(mx.load(str(path / "optimizer.safetensors")).items()))
    mx.eval(model.parameters(), optimizer.state)
    return json.loads((path / "training-state.json").read_text())


def recover_metrics(output: Path, step: int):
    """Keep committed steps in the primary log and preserve interrupted work."""
    path = output / "metrics.jsonl"
    if not path.exists():
        return
    committed, interrupted = [], []
    for line in path.read_text().splitlines(keepends=True):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            interrupted.append(line)
            continue
        (committed if row["step"] <= step else interrupted).append(line)
    if interrupted:
        archive = output / "interrupted-metrics"
        archive.mkdir(exist_ok=True)
        shutil.copy2(path, archive / f"attempt-{time.time_ns()}.jsonl")
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(committed))
        os.replace(temporary, path)


def validate_stage(model, tokenizer, examples, records, config, output: Path):
    model.eval()
    losses = []
    correct = truncated = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    for previous in (output, temporary):
        if previous.exists():
            os.replace(
                previous, previous.with_name(f"interrupted-{time.time_ns()}-{previous.name}")
            )
    with temporary.open("x") as stream:
        for example, record in zip(examples, records, strict=True):
            batch = make_batch([record], tokenizer.pad_token_id or 0)
            loss = token_loss(model, *batch, config.latent_steps).item()
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite validation loss")
            losses.append(loss)
            row = generate(
                model,
                tokenizer,
                example,
                "hybrid",
                config.latent_steps,
                max_tokens=config.validation_max_tokens,
                hybrid_boundary=config.hybrid_boundary,
            )
            stream.write(json.dumps({**row, "teacher_forced_loss": loss}, allow_nan=False) + "\n")
            stream.flush()
            correct += int(row["correct"])
            truncated += int(not row["terminated"])
    loss = sum(losses) / len(losses)
    os.replace(temporary, output)
    return {
        "n": len(examples),
        "correct": correct,
        "loss": loss,
        "truncated": truncated,
        "predictions_file": output.name,
        "predictions_sha256": sha256(output),
    }


def train_stage(
    model,
    tokenizer,
    train_path: Path,
    validation_path: Path,
    output: Path,
    model_path: str,
    config: StageConfig,
    source_adapter: str | None = None,
    resume: bool = False,
):
    config.validate()
    examples = read_examples(train_path)
    validation = read_examples(validation_path)
    train_ids, validation_ids = {x.id for x in examples}, {x.id for x in validation}
    if not examples or not validation:
        raise ValueError("Training and validation must be nonempty")
    if len(train_ids) != len(examples) or len(validation_ids) != len(validation):
        raise ValueError("Duplicate examples within a split")
    if train_ids & validation_ids:
        raise ValueError("Training/validation overlap")
    if any(x.split != "train" for x in examples) or any(
        x.split != "validation" for x in validation
    ):
        raise ValueError("Training requires train and validation splits")
    # Validation ordering is independent of model seed and identical across arms.
    validation = sorted(validation, key=lambda x: hashlib.sha256(x.id.encode()).digest())
    validation = validation[: config.validation_size]
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    run = {
        **asdict(config),
        "adapter": asdict(model.config),
        "model_path": str(Path(model_path).resolve()),
        "source_adapter": source_adapter,
        "source_adapter_sha256": sha256(Path(source_adapter) / "adapter.safetensors")
        if source_adapter
        else None,
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(validation_path),
        "validation_ids": [x.id for x in validation],
        "train_examples": len(examples),
        "sampling": "Complete shuffled epochs without replacement; microbatch size one",
        "randomness": "MLX is reseeded with seed + update number before each update",
        "selection": "Validation answer accuracy, then validation loss; final epoch continues next stage",
        "source_sha256": source_hashes,
    }
    if resume:
        if json.loads((output / "run.json").read_text()) != run:
            raise ValueError("Resume settings, data, initialization, or source have changed")
        if (output / "result.json").exists():
            return json.loads((output / "result.json").read_text())
        if not (output / "provenance.json").exists():
            capture(output, model_path, source_adapter)
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "run.json", run)
        capture(output, model_path, source_adapter)

    def encode(example, weight=1.0):
        return encode_example(
            tokenizer,
            example,
            "hybrid",
            config.reasoning_steps_to_drop,
            config.hybrid_boundary,
            carried_value_weight=weight,
        )

    def training_record(example):
        record = encode(example, config.carried_value_weight)
        if config.value_aux_weight <= 0:
            return record
        result = removed_step_result(example, config.reasoning_steps_to_drop)
        value_ids = tokenizer.encode(result, add_special_tokens=False) if result else None
        return (*record, value_ids)

    records = [training_record(x) for x in examples]
    # Validation loss stays the unweighted text objective, comparable across arms.
    val_records = [encode(x) for x in validation]
    optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=0.0)
    metadata = {
        "model_path": run["model_path"],
        "run": {**run, "provenance": json.loads((output / "provenance.json").read_text())},
    }
    state = {
        "step": 0,
        "examples_seen": 0,
        "supervised_tokens": 0,
        "elapsed_s": 0.0,
        "best_score": None,
    }
    if resume:
        if (output / "last.json").exists():
            state = restore_checkpoint(model, optimizer, output)
            # Repair a last/best pointer interruption at an improved epoch.
            best_score = state["best_score"]
            current_best = None
            if (output / "best.json").exists():
                current_best = json.loads(
                    (checkpoint_path(output, "best") / "training-state.json").read_text()
                )["best_score"]
            if best_score is not None and (current_best is None or best_score > current_best):
                atomic_json(output / "best.json", json.loads((output / "last.json").read_text()))
        else:
            # An interruption before the first checkpoint performed no updates.
            save_checkpoint(model, optimizer, output, metadata, state)
        recover_metrics(output, state["step"])
    else:
        mx.random.seed(config.seed)
        save_checkpoint(model, optimizer, output, metadata, state)
    steps_per_epoch = math.ceil(len(records) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    started = time.monotonic()
    prior_elapsed = state["elapsed_s"]
    print(
        json.dumps(
            {
                "status": "training",
                "output": str(output),
                "steps": total_steps,
                "resume_step": state["step"],
                **parameter_counts(model),
            }
        ),
        flush=True,
    )
    with (output / "metrics.jsonl").open("a") as metrics:
        for epoch in range(config.epochs):
            for index, indices in enumerate(
                epoch_indices(len(records), config.seed, epoch, config.batch_size)
            ):
                step = epoch * steps_per_epoch + index + 1
                if step <= state["step"]:
                    continue
                mx.random.seed(config.seed + step)
                model.train()
                loss, grads, tokens = accumulated_gradient(
                    model,
                    tokenizer,
                    [records[i] for i in indices],
                    config.latent_steps,
                    config.value_aux_weight,
                )
                grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
                mx.eval(norm)
                gradient_norm = norm.item()
                if not math.isfinite(loss) or not math.isfinite(gradient_norm):
                    atomic_json(
                        output / "failure.json",
                        {
                            "step": step,
                            "loss": str(loss),
                            "gradient_norm": str(gradient_norm),
                            "updated": False,
                        },
                    )
                    raise FloatingPointError(f"Nonfinite loss/gradient before step {step}")
                optimizer.update(model, grads)
                mx.eval(model.trainable_parameters(), optimizer.state)
                state.update(
                    step=step,
                    examples_seen=state["examples_seen"] + len(indices),
                    supervised_tokens=state["supervised_tokens"] + tokens,
                    elapsed_s=prior_elapsed + time.monotonic() - started,
                )
                row = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": loss,
                    "gradient_norm": gradient_norm,
                    "examples_seen": state["examples_seen"],
                    "elapsed_s": state["elapsed_s"],
                    "peak_memory_bytes": mx.get_peak_memory(),
                }
                end_epoch = index + 1 == steps_per_epoch
                improved = False
                if end_epoch:
                    validation_result = validate_stage(
                        model,
                        tokenizer,
                        validation,
                        val_records,
                        config,
                        output / "validation" / f"epoch-{epoch + 1:03d}.jsonl",
                    )
                    score = [
                        validation_result["correct"] / validation_result["n"],
                        -validation_result["loss"],
                    ]
                    improved = state["best_score"] is None or score > state["best_score"]
                    if improved:
                        state["best_score"] = score
                    row["validation"] = validation_result
                    state["elapsed_s"] = prior_elapsed + time.monotonic() - started
                    row["elapsed_s"] = state["elapsed_s"]
                if step % config.log_every == 0 or step == 1 or end_epoch:
                    metrics.write(json.dumps(row, allow_nan=False) + "\n")
                    metrics.flush()
                    print(json.dumps(row, allow_nan=False), flush=True)
                if step % config.checkpoint_every == 0 or end_epoch:
                    save_checkpoint(model, optimizer, output, metadata, state, best=improved)
                    atomic_json(output / "progress.json", {**state, "total_steps": total_steps})
    result = {
        **state,
        "total_steps": total_steps,
        "best_checkpoint": str(checkpoint_path(output, "best")),
        "last_checkpoint": str(checkpoint_path(output)),
    }
    atomic_json(output / "result.json", result)
    return result
