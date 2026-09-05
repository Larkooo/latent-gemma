"""Greedy answer generation and synchronized per-example measurements."""

import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .data import Example, extract_answer, prompt_text
from .model import LatentModel


def generate(
    model: LatentModel,
    tokenizer,
    example: Example,
    mode: str,
    steps: int = 4,
    max_tokens: int = 96,
    ablation: str = "none",
) -> dict:
    if mode not in {"direct", "cot", "latent", "native", "hybrid"}:
        raise ValueError(f"Unknown mode: {mode}")
    prompt = mx.array(
        [
            tokenizer.encode(
                prompt_text(tokenizer, example, mode == "native"), add_special_tokens=False
            )
        ]
    )
    latent_steps = steps if mode in {"latent", "hybrid"} else 0
    model.eval()
    mx.synchronize()
    started = time.perf_counter()
    state, cache = model.prefill(prompt, latent_steps, ablation)
    forced_count = 0
    if mode not in {"cot", "native", "hybrid"}:
        suffix = tokenizer.encode("\nAnswer: ", add_special_tokens=False)
        forced_count = len(suffix)
        state = model.hidden(mx.array([suffix]), cache=cache)[:, -1:, :]
    tokens = []
    stops = (
        set(tokenizer.eos_token_ids)
        if hasattr(tokenizer, "eos_token_ids")
        else {tokenizer.eos_token_id}
    )
    terminated = False
    for index in range(max_tokens):
        next_id = mx.argmax(model.logits(state[:, -1:, :]), axis=-1).item()
        tokens.append(next_id)
        if next_id in stops:
            terminated = True
            break
        if index + 1 < max_tokens:
            state = model.hidden(mx.array([[next_id]]), cache=cache)
    mx.synchronize()
    latency = time.perf_counter() - started
    text = tokenizer.decode([t for t in tokens if t not in stops])
    prediction = extract_answer(text, example.task, mode)
    return {
        "id": example.id,
        "task": example.task,
        "mode": mode,
        "latent_steps": latent_steps,
        "ablation": ablation,
        "prediction": prediction,
        "answer": example.answer,
        "correct": prediction == example.answer,
        "text": text,
        "latency_s": latency,
        "generated_tokens": len(tokens),
        "prompt_tokens": prompt.shape[1],
        "forced_tokens": forced_count,
        "terminated": terminated,
        "transformer_positions": prompt.shape[1]
        + forced_count
        + latent_steps
        + max(0, len(tokens) - 1),
        "peak_memory_bytes": mx.get_peak_memory(),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")

    def metrics(group):
        n = len(group)
        p = sum(x["correct"] for x in group) / n
        # Wilson score interval, including the all-correct/all-wrong cases.
        z = 1.96
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        radius = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
        return {
            "n": n,
            "correct": sum(x["correct"] for x in group),
            "accuracy": p,
            "accuracy_ci95": [center - radius, center + radius],
            "median_latency_s": statistics.median(x["latency_s"] for x in group),
            "p95_latency_s": float(np.percentile([x["latency_s"] for x in group], 95)),
            "mean_generated_tokens": statistics.mean(x["generated_tokens"] for x in group),
            "mean_transformer_positions": statistics.mean(
                x["transformer_positions"] for x in group
            ),
            "truncated": sum(not x["terminated"] for x in group),
        }

    return {
        "overall": metrics(rows),
        "tasks": {
            task: metrics([x for x in rows if x["task"] == task])
            for task in sorted({x["task"] for x in rows})
        },
    }


def evaluate(
    model,
    tokenizer,
    examples: list[Example],
    output: Path,
    mode: str,
    steps: int,
    max_tokens: int,
    ablation: str = "none",
    metadata: dict | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Warmup is excluded from metrics; it uses a training-independent fixed prompt.
    warmup = Example("warmup", "arithmetic", "Compute (2 + 3) * 4.", "", "20", "warmup")
    generate(
        model, tokenizer, warmup, mode, steps, max_tokens=min(max_tokens, 8), ablation=ablation
    )
    mx.reset_peak_memory()
    rows = []
    with output.open("w") as file:
        for i, example in enumerate(examples):
            row = generate(model, tokenizer, example, mode, steps, max_tokens, ablation)
            rows.append(row)
            file.write(json.dumps(row) + "\n")
            file.flush()
            if (i + 1) % 25 == 0:
                print(
                    json.dumps({"evaluated": i + 1, "correct": sum(r["correct"] for r in rows)}),
                    flush=True,
                )
    result = {
        "metadata": metadata or {},
        "mode": mode,
        "steps": steps,
        "ablation": ablation,
        "predictions_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        **summarize(rows),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result
