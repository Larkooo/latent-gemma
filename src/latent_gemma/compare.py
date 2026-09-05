"""Paired accuracy comparisons with deterministic bootstrap intervals."""

import json
from pathlib import Path

import numpy as np


def read_predictions(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x]
    if not rows or len({x["id"] for x in rows}) != len(rows):
        raise ValueError("Predictions must be nonempty with unique IDs")
    return rows


def compare(
    baseline: list[dict],
    candidate: list[dict],
    seed: int = 42,
    latency_field: str = "latency_s",
) -> dict:
    if latency_field not in {"latency_s", "end_to_end_latency_s"}:
        raise ValueError(f"Unknown latency field: {latency_field}")
    a = {x["id"]: x for x in baseline}
    b = {x["id"]: x for x in candidate}
    if not a or len(a) != len(baseline) or len(b) != len(candidate) or a.keys() != b.keys():
        raise ValueError("Paired evaluation requires identical, nonempty, unique example IDs")
    for key in a:
        if a[key]["answer"] != b[key]["answer"] or a[key]["task"] != b[key]["task"]:
            raise ValueError(f"Different targets for example {key}")
        for row in (a[key], b[key]):
            value = row.get(latency_field)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Missing or invalid {latency_field} for example {key}")
    result = {}
    rng = np.random.default_rng(seed)
    timing_rng = np.random.default_rng(seed)
    tasks = ["overall", *sorted({x["task"] for x in baseline})]
    for task in tasks:
        keys = [k for k in a if task == "overall" or a[k]["task"] == task]
        delta = np.array([int(b[k]["correct"]) - int(a[k]["correct"]) for k in keys])
        bootstrap = np.array(
            [rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)]
        )
        before_latency = np.array([a[k][latency_field] for k in keys])
        after_latency = np.array([b[k][latency_field] for k in keys])
        paired_speedups = before_latency / after_latency
        timing_samples = []
        # Resample paired questions, keeping memory bounded for larger evaluations.
        # These intervals describe question sampling, not between-session drift.
        for start in range(0, 10000, 256):
            indices = timing_rng.integers(0, len(keys), (min(256, 10000 - start), len(keys)))
            ratio = np.median(before_latency[indices], axis=1) / np.median(
                after_latency[indices], axis=1
            )
            paired = np.median(paired_speedups[indices], axis=1)
            timing_samples.append(np.column_stack((ratio, paired)))
        timing_intervals = np.percentile(np.concatenate(timing_samples), [2.5, 97.5], axis=0)
        result[task] = {
            "latency_field": latency_field,
            "n": len(keys),
            "baseline_accuracy": float(np.mean([a[k]["correct"] for k in keys])),
            "candidate_accuracy": float(np.mean([b[k]["correct"] for k in keys])),
            "accuracy_delta": float(delta.mean()),
            "paired_delta_ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "candidate_only_correct": int(np.sum(delta == 1)),
            "baseline_only_correct": int(np.sum(delta == -1)),
            "baseline_median_latency_s": float(np.median(before_latency)),
            "candidate_median_latency_s": float(np.median(after_latency)),
            "ratio_of_median_latencies": float(
                np.median(before_latency) / np.median(after_latency)
            ),
            "ratio_of_median_latencies_ci95": timing_intervals[:, 0].tolist(),
            "median_paired_speedup": float(np.median(paired_speedups)),
            "median_paired_speedup_ci95": timing_intervals[:, 1].tolist(),
            "fraction_questions_faster": float(np.mean(after_latency < before_latency)),
        }
    return result
