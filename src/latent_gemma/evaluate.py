"""Greedy answer generation and synchronized per-example measurements."""

import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .data import Example, extract_answer, hybrid_boundary_text, prompt_text
from .model import LatentModel


def _decode_tokens(model, state, cache, stops, max_tokens, strategy):
    tokens = []
    if strategy == "serial":
        for index in range(max_tokens):
            next_id = mx.argmax(model.logits(state[:, -1:, :]), axis=-1).item()
            tokens.append(next_id)
            if next_id in stops:
                return tokens, True, 0
            if index + 1 < max_tokens:
                state = model.hidden(mx.array([[next_id]]), cache=cache)
        return tokens, False, 0

    # Schedule the next token before reading the current one on the host. This
    # overlaps graph construction and execution, as in MLX LM's generation loop.
    token = mx.argmax(model.logits(state[:, -1:, :]), axis=-1)
    mx.async_eval(token)
    for index in range(max_tokens):
        prefetched = index + 1 < max_tokens
        if prefetched:
            state = model.hidden(token.reshape(1, 1), cache=cache)
            next_token = mx.argmax(model.logits(state[:, -1:, :]), axis=-1)
            mx.async_eval(next_token)
        next_id = token.item()
        tokens.append(next_id)
        if next_id in stops:
            # A request ending before its cap scheduled one unused continuation.
            # Synchronization below includes that work in latency measurements.
            return tokens, True, int(prefetched)
        if prefetched:
            token = next_token
    return tokens, False, 0


def generate(
    model: LatentModel,
    tokenizer,
    example: Example,
    mode: str,
    steps: int = 4,
    max_tokens: int = 96,
    ablation: str = "none",
    hybrid_boundary: str = "none",
    decode_strategy: str = "serial",
) -> dict:
    if mode not in {"direct", "cot", "latent", "native", "hybrid", "plain"}:
        raise ValueError(f"Unknown mode: {mode}")
    if decode_strategy not in {"serial", "pipelined"}:
        raise ValueError(f"Unknown decode strategy: {decode_strategy}")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    model.eval()
    mx.synchronize()
    request_started = time.perf_counter()
    prompt = mx.array(
        [
            tokenizer.encode(
                prompt_text(tokenizer, example, mode == "native", reasoning_prefix=mode != "plain"),
                add_special_tokens=False,
            )
        ]
    )
    latent_steps = steps if mode in {"latent", "hybrid"} else 0
    mx.synchronize()
    started = time.perf_counter()
    state, cache = model.prefill(prompt, latent_steps, ablation)
    forced_count = 0
    suffix_text = ""
    if mode in {"direct", "latent"}:
        suffix_text = "\nAnswer: "
    elif mode == "hybrid":
        suffix_text = hybrid_boundary_text(hybrid_boundary)
    if suffix_text:
        suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
        forced_count = len(suffix)
        state = model.hidden(mx.array([suffix]), cache=cache)[:, -1:, :]
    stops = (
        set(tokenizer.eos_token_ids)
        if hasattr(tokenizer, "eos_token_ids")
        else {tokenizer.eos_token_id}
    )
    tokens, terminated, prefetched = _decode_tokens(
        model, state, cache, stops, max_tokens, decode_strategy
    )
    mx.synchronize()
    latency = time.perf_counter() - started
    text = tokenizer.decode([t for t in tokens if t not in stops])
    prediction = extract_answer(text, example.task, mode)
    end_to_end_latency = time.perf_counter() - request_started
    return {
        "id": example.id,
        "task": example.task,
        "mode": mode,
        "latent_steps": latent_steps,
        "ablation": ablation,
        "hybrid_boundary": hybrid_boundary if mode == "hybrid" else "none",
        "decode_strategy": decode_strategy,
        "prediction": prediction,
        "answer": example.answer,
        "correct": prediction == example.answer,
        "text": text,
        "latency_s": latency,
        "end_to_end_latency_s": end_to_end_latency,
        "generated_tokens": len(tokens),
        "token_ids": tokens,
        "prompt_tokens": prompt.shape[1],
        "forced_tokens": forced_count,
        "terminated": terminated,
        "prefetched_text_positions": prefetched,
        "vocabulary_projections": len(tokens) + prefetched,
        "transformer_positions": prompt.shape[1]
        + forced_count
        + latent_steps
        + max(0, len(tokens) - 1)
        + prefetched,
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
        result = {
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
        # Historical runs lack this measurement. Never synthesize it from model
        # latency or summarize an unrepresentative subset of a mixed run.
        if all("end_to_end_latency_s" in x for x in group):
            durations = [x["end_to_end_latency_s"] for x in group]
            result["median_end_to_end_latency_s"] = statistics.median(durations)
            result["p95_end_to_end_latency_s"] = float(np.percentile(durations, 95))
        for field in ("prefetched_text_positions", "vocabulary_projections"):
            if all(field in x for x in group):
                result[f"mean_{field}"] = statistics.mean(x[field] for x in group)
        return result

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
    hybrid_boundary: str = "none",
    decode_strategy: str = "serial",
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Warmup is excluded from metrics; it uses a training-independent fixed prompt.
    warmup = Example("warmup", "arithmetic", "Compute (2 + 3) * 4.", "", "20", "warmup")
    generate(
        model,
        tokenizer,
        warmup,
        mode,
        steps,
        max_tokens=min(max_tokens, 8),
        ablation=ablation,
        hybrid_boundary=hybrid_boundary,
        decode_strategy=decode_strategy,
    )
    mx.reset_peak_memory()
    rows = []
    with output.open("w") as file:
        for i, example in enumerate(examples):
            row = generate(
                model,
                tokenizer,
                example,
                mode,
                steps,
                max_tokens,
                ablation,
                hybrid_boundary,
                decode_strategy,
            )
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
        "max_tokens": max_tokens,
        "ablation": ablation,
        "hybrid_boundary": hybrid_boundary if mode == "hybrid" else "none",
        "decode_strategy": decode_strategy,
        "timing_scope": {
            "latency_s": "Prompt prefill, latent computation, forced prefix, and generation through stop or token cap; excludes prompt formatting and final decoding.",
            "end_to_end_latency_s": "Warm model request, from prompt formatting/tokenization through final decoding and answer extraction; excludes model loading, warmup, and external serving/network overhead.",
        },
        "predictions_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        **summarize(rows),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result
