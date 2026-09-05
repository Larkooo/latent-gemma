"""Seeded LoRA/feedback training with held-out loss and recoverable checkpoints."""

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .data import encode_example, hybrid_boundary_text, read_examples
from .model import parameter_counts, token_loss
from .provenance import capture


def make_batch(records, pad_id: int):
    if not records or len({len(x[0]) for x in records}) != 1:
        raise ValueError("A batch requires equal unpadded prompt lengths")
    length = max(len(x[1]) for x in records)
    prompts = mx.array([x[0] for x in records], dtype=mx.int32)
    targets = mx.array([x[1] + [pad_id] * (length - len(x[1])) for x in records], dtype=mx.int32)
    masks = mx.array([x[2] + [0.0] * (length - len(x[2])) for x in records], dtype=mx.float32)
    return prompts, targets, masks


def encoded_buckets(tokenizer, examples, modes, reasoning_steps_to_drop=0, hybrid_boundary="none"):
    result = {}
    for mode in modes:
        buckets = defaultdict(list)
        for example in examples:
            record = encode_example(
                tokenizer, example, mode, reasoning_steps_to_drop, hybrid_boundary
            )
            buckets[len(record[0])].append(record)
        result[mode] = list(buckets.values())
    return result


def train(
    model,
    tokenizer,
    train_path: Path,
    validation_path: Path,
    output: Path,
    model_path: str,
    steps: int = 500,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    latent_steps: int = 4,
    modes: tuple[str, ...] = ("direct", "cot", "latent"),
    seed: int = 42,
    eval_every: int = 100,
    log_every: int = 10,
    source_adapter: str | None = None,
    reasoning_steps_to_drop: int = 0,
    hybrid_boundary: str = "none",
    train_ablation: str = "none",
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite training run: {output}")
    if not modes or any(mode not in {"direct", "cot", "latent", "hybrid"} for mode in modes):
        raise ValueError("Choose direct, cot, latent, and/or hybrid training modes")
    if reasoning_steps_to_drop < 0:
        raise ValueError("reasoning_steps_to_drop must be nonnegative")
    if train_ablation not in {"none", "zero", "repeat", "shuffle"}:
        raise ValueError(f"Unknown training ablation: {train_ablation}")
    hybrid_boundary_text(hybrid_boundary)
    if steps <= 0 or batch_size <= 0 or eval_every <= 0 or log_every <= 0:
        raise ValueError("Training counts must be positive")
    output.mkdir(parents=True)
    provenance = capture(output, model_path, source_adapter)
    config = {
        "model_path": model_path,
        "source_adapter": source_adapter,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "latent_steps": latent_steps,
        "reasoning_steps_to_drop": reasoning_steps_to_drop,
        "hybrid_boundary": hybrid_boundary,
        "train_ablation": train_ablation,
        "modes": modes,
        "seed": seed,
        "eval_every": eval_every,
        "adapter": asdict(model.config),
        "parameters": parameter_counts(model),
        "provenance": provenance,
    }
    (output / "run.json").write_text(json.dumps(config, indent=2) + "\n")
    rng = random.Random(seed)
    mx.random.seed(seed)
    examples = read_examples(train_path)
    validation = read_examples(validation_path)
    if {x.id for x in examples} & {x.id for x in validation}:
        raise ValueError("Training/validation overlap")
    if any(x.split != "train" for x in examples) or any(
        x.split != "validation" for x in validation
    ):
        raise ValueError("Training requires train/validation splits; never use test examples")
    print(json.dumps({"status": "encoding", **config}), flush=True)
    buckets = encoded_buckets(tokenizer, examples, modes, reasoning_steps_to_drop, hybrid_boundary)
    val_records = {
        mode: [
            encode_example(tokenizer, x, mode, reasoning_steps_to_drop, hybrid_boundary)
            for x in validation[:32]
        ]
        for mode in modes
    }
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)
    loss_and_grad = nn.value_and_grad(model, token_loss)
    best_loss = float("inf")
    started = time.perf_counter()
    with (output / "metrics.jsonl").open("w") as log:
        for step in range(1, steps + 1):
            mode = modes[(step - 1) % len(modes)]
            bucket = rng.choices(buckets[mode], weights=[len(b) for b in buckets[mode]])[0]
            records = rng.choices(bucket, k=batch_size)
            batch = make_batch(records, tokenizer.pad_token_id or 0)
            k = latent_steps if mode in {"latent", "hybrid"} else 0
            model.train()
            loss, grads = loss_and_grad(model, *batch, k, train_ablation)
            grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
            mx.eval(loss, norm)
            value, gradient_norm = loss.item(), norm.item()
            if not math.isfinite(value) or not math.isfinite(gradient_norm):
                failure = {
                    "step": step,
                    "mode": mode,
                    "loss": str(value),
                    "gradient_norm": str(gradient_norm),
                    "updated": False,
                }
                (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
                raise FloatingPointError(f"Nonfinite loss or gradient before update: {failure}")
            optimizer.update(model, grads)
            mx.eval(loss, norm, model.parameters(), optimizer.state)
            if step % log_every == 0 or step == 1:
                row = {
                    "step": step,
                    "mode": mode,
                    "loss": value,
                    "grad_norm": gradient_norm,
                    "elapsed_s": time.perf_counter() - started,
                    "peak_memory_bytes": mx.get_peak_memory(),
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
            if step % eval_every == 0 or step == steps:
                model.eval()
                val_losses = {}
                for val_mode, items in val_records.items():
                    k_val = latent_steps if val_mode in {"latent", "hybrid"} else 0
                    losses = [
                        token_loss(
                            model,
                            *make_batch([r], tokenizer.pad_token_id or 0),
                            k_val,
                            train_ablation,
                        ).item()
                        for r in items
                    ]
                    val_losses[val_mode] = sum(losses) / len(losses)
                # Mode-specific selection avoids choosing latent checkpoints based
                # on improvements in a different output format.
                score = val_losses.get(
                    "latent", val_losses.get("hybrid", sum(val_losses.values()) / len(val_losses))
                )
                metadata = {
                    "model_path": model_path,
                    "run": config,
                    "step": step,
                    "validation_losses": val_losses,
                }
                model.save_adapter(output / "last", metadata)
                if score < best_loss:
                    best_loss = score
                    model.save_adapter(output / "best", metadata)
                row = {"step": step, "validation_loss": val_losses, "best_loss": best_loss}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
    result = {"steps": steps, "best_loss": best_loss, "elapsed_s": time.perf_counter() - started}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
