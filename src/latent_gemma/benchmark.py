"""Interleaved, repeated comparison of two decoding paths on one checkpoint."""

import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx

from .compare import compare
from .data import Example
from .evaluate import generate, summarize
from .provenance import sha256


@dataclass(frozen=True)
class DecodeCondition:
    mode: str
    steps: int = 0
    max_tokens: int = 96
    ablation: str = "none"
    hybrid_boundary: str = "none"


def benchmark_pair(
    model,
    tokenizer,
    examples: list[Example],
    output: Path,
    baseline: DecodeCondition,
    candidate: DecodeCondition,
    repeats: int = 3,
    seed: int = 42,
    metadata: dict | None = None,
) -> dict:
    if repeats < 2:
        raise ValueError("Repeated benchmarking requires at least two repeats")
    if not examples or len({x.id for x in examples}) != len(examples):
        raise ValueError("Benchmark examples must be nonempty with unique IDs")
    output.mkdir(parents=True, exist_ok=False)
    conditions = {"baseline": baseline, "candidate": candidate}
    warmup = Example("warmup", "arithmetic", "Compute (2 + 3) * 4.", "", "20", "warmup")
    for condition in conditions.values():
        settings = asdict(condition)
        settings["max_tokens"] = min(condition.max_tokens, 8)
        generate(model, tokenizer, warmup, **settings)
    mx.reset_peak_memory()
    # Counterbalance which condition runs first, then randomize the schedule.
    first_sides = [
        "baseline" if i % 2 == 0 else "candidate" for i in range(len(examples) * repeats)
    ]
    random.Random(seed).shuffle(first_sides)
    aggregates = {side: [] for side in conditions}
    semantic_keys = (
        "prediction",
        "answer",
        "correct",
        "text",
        "generated_tokens",
        "prompt_tokens",
        "forced_tokens",
        "terminated",
        "transformer_positions",
    )
    with (output / "measurements.jsonl").open("x") as trace:
        for index, example in enumerate(examples):
            measurements = {side: [] for side in conditions}
            for repeat in range(repeats):
                first = first_sides[index * repeats + repeat]
                order = [first, "candidate" if first == "baseline" else "baseline"]
                for position, side in enumerate(order):
                    row = generate(model, tokenizer, example, **asdict(conditions[side]))
                    trace.write(
                        json.dumps({"repeat": repeat, "order": position, "condition": side, **row})
                        + "\n"
                    )
                    trace.flush()
                    measurements[side].append(row)
            for side, rows in measurements.items():
                first = rows[0]
                if any(any(row[key] != first[key] for key in semantic_keys) for row in rows[1:]):
                    raise ValueError(
                        f"Nondeterministic generated output for {example.id}, {side}; inspect measurements"
                    )
                aggregate = {**first, "measurement_repeats": repeats}
                for field in ("latency_s", "end_to_end_latency_s"):
                    aggregate[field] = statistics.median(row[field] for row in rows)
                aggregate["peak_memory_bytes"] = max(row["peak_memory_bytes"] for row in rows)
                aggregates[side].append(aggregate)
            if (index + 1) % 10 == 0:
                print(
                    json.dumps({"benchmarked_examples": index + 1, "repeats": repeats}), flush=True
                )
    hashes = {}
    for side, rows in aggregates.items():
        path = output / f"{side}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        hashes[side] = sha256(path)
    result = {
        "metadata": metadata or {},
        "seed": seed,
        "repeats": repeats,
        "conditions": {side: asdict(condition) for side, condition in conditions.items()},
        "aggregation": "Median latency per example and condition; each question counts once for accuracy. Both paths use the same loaded checkpoint.",
        "timing_scope": "Warm request from prompt formatting through final decoding and answer extraction; excludes model loading, warmup, and external serving overhead.",
        "predictions_sha256": hashes,
        "measurements_sha256": sha256(output / "measurements.jsonl"),
        "results": {side: summarize(rows) for side, rows in aggregates.items()},
        "comparison": compare(
            aggregates["baseline"], aggregates["candidate"], seed, "end_to_end_latency_s"
        ),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
