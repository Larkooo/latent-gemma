"""Audit campaign artifacts and report seed-level accuracy and paired timings."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from run_campaign import ARMS, digest, verify_plan, write_json

from latent_gemma.data import SCORING_POLICY, answer_matches, extract_answer

SEMANTIC_FIELDS = (
    "id",
    "task",
    "mode",
    "latent_steps",
    "ablation",
    "hybrid_boundary",
    "decode_strategy",
    "prediction",
    "answer",
    "correct",
    "scoring_policy",
    "text",
    "generated_tokens",
    "token_ids",
    "prompt_tokens",
    "forced_tokens",
    "terminated",
    "transformer_positions",
    "prefetched_text_positions",
    "vocabulary_projections",
)


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"Empty artifact: {path}")
    return rows


def index_rows(rows):
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    return indexed


def audit_predictions(path, questions, selected, max_tokens, decode_strategy="serial"):
    rows = index_rows(read_rows(path))
    if set(rows) != set(questions):
        raise ValueError(f"Prediction coverage differs from test split: {path}")
    for identifier, row in rows.items():
        question = questions[identifier]
        prediction = extract_answer(row["text"], row["task"], row["mode"])
        correct = answer_matches(prediction, question["answer"], question["task"])
        if (
            row["answer"] != question["answer"]
            or row["task"] != question["task"]
            or row["prediction"] != prediction
            or row["correct"] != correct
            or row["scoring_policy"] != SCORING_POLICY
        ):
            raise ValueError(f"Prediction or score cannot be reproduced: {identifier}")
        if (
            row["mode"] != "hybrid"
            or row["hybrid_boundary"] != "reasoning"
            or row["latent_steps"] != selected["latent_steps"]
            or row["decode_strategy"] != decode_strategy
            or row["ablation"] != "none"
            or row["generated_tokens"] != len(row["token_ids"])
            or not 0 < row["generated_tokens"] <= max_tokens
        ):
            raise ValueError(f"Unexpected decoding policy: {identifier}")
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    if (
        digest(path) != summary["predictions_sha256"]
        or summary["overall"]["correct"] != sum(row["correct"] for row in rows.values())
        or summary["overall"]["n"] != len(rows)
    ):
        raise ValueError(f"Summary does not reconcile with predictions: {path}")
    return rows


def audit_timing(directory, timing_ids, predictions, repeats):
    result = json.loads((directory / "result.json").read_text())
    trace_path = directory / "measurements.jsonl"
    trace = read_rows(trace_path)
    if digest(trace_path) != result["measurements_sha256"]:
        raise ValueError(f"Timing trace hash mismatch: {directory}")
    if len(trace) != len(timing_ids) * repeats * 2:
        raise ValueError("Incomplete timing trace")
    measurements, orders = defaultdict(list), defaultdict(dict)
    first = Counter()
    for row in trace:
        identifier, side, repeat = row["id"], row["condition"], row["repeat"]
        if identifier not in timing_ids or side not in predictions or repeat not in range(repeats):
            raise ValueError("Unexpected timing request")
        key = identifier, repeat
        if side in orders[key] or row["order"] not in (0, 1):
            raise ValueError("Repeated or invalid timing order")
        orders[key][side] = row["order"]
        if row["order"] == 0:
            first[side] += 1
        if any(row[field] != predictions[side][identifier][field] for field in SEMANTIC_FIELDS):
            raise ValueError(f"Timed generation differs from untimed test: {identifier}, {side}")
        measurements[identifier, side].append(row)
    if any(sorted(order.values()) != [0, 1] for order in orders.values()):
        raise ValueError("Timing requests are not paired")
    if abs(first["baseline"] - first["candidate"]) > 1:
        raise ValueError("Timing order was not counterbalanced")
    aggregates = {}
    for side in predictions:
        path = directory / f"{side}.jsonl"
        if digest(path) != result["predictions_sha256"][side]:
            raise ValueError("Timing aggregate hash mismatch")
        rows = index_rows(read_rows(path))
        if set(rows) != timing_ids:
            raise ValueError("Timing aggregate coverage mismatch")
        for identifier, row in rows.items():
            samples = measurements[identifier, side]
            if len(samples) != repeats or row["measurement_repeats"] != repeats:
                raise ValueError("Missing timing repeats")
            if any(row[field] != predictions[side][identifier][field] for field in SEMANTIC_FIELDS):
                raise ValueError("Timing aggregate changed generated output")
            for field in ("latency_s", "end_to_end_latency_s"):
                durations = [sample[field] for sample in samples]
                if not all(np.isfinite(value) and value > 0 for value in durations):
                    raise ValueError("Invalid latency")
                if row[field] != statistics.median(durations):
                    raise ValueError("Timing aggregate is not the per-question median")
        aggregates[side] = rows
    identifiers = sorted(timing_ids)
    baseline, candidate = [
        np.array(
            [aggregates[side][identifier]["end_to_end_latency_s"] for identifier in identifiers]
        )
        for side in ("baseline", "candidate")
    ]
    return {
        "questions": len(identifiers),
        "repeats": repeats,
        "baseline_mean_s": float(baseline.mean()),
        "candidate_mean_s": float(candidate.mean()),
        "mean_latency_reduction": float(1 - candidate.mean() / baseline.mean()),
        "baseline_median_s": float(np.median(baseline)),
        "candidate_median_s": float(np.median(candidate)),
        "median_latency_reduction": float(1 - np.median(candidate) / np.median(baseline)),
        "baseline_p95_s": float(np.percentile(baseline, 95)),
        "candidate_p95_s": float(np.percentile(candidate, 95)),
        "median_paired_speedup": float(np.median(baseline / candidate)),
        "session_calibration": result["session_calibration"],
        "paired_intervals": result["comparison"]["overall"],
    }


def accuracy_comparison(baseline, candidate, seed=20260905, draws=10000):
    """Crossed bootstrap preserves paired questions and shared training seeds."""
    differences = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    if differences.ndim != 2 or differences.shape[0] < 3 or not differences.shape[1]:
        raise ValueError("Expected at least three seeds with paired questions")
    seeds, questions = differences.shape
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for index in range(draws):
        chosen_seeds = rng.integers(seeds, size=seeds)
        chosen_questions = rng.integers(questions, size=questions)
        samples[index] = differences[np.ix_(chosen_seeds, chosen_questions)].mean()
    return {
        "mean_accuracy_delta": float(differences.mean()),
        "per_seed_accuracy_delta": differences.mean(axis=1).tolist(),
        "crossed_bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "bootstrap_draws": draws,
        "uncertainty_note": "Exploratory interval: only three training seeds; seed and shared question IDs are independently resampled while pairing arms. No multiplicity adjustment.",
    }


def summarize(output):
    plan = json.loads((output / "plan.json").read_text())
    verify_plan(output, plan)
    selections = json.loads((output / "selected-checkpoints.json").read_text())
    questions = index_rows(read_rows(output / "data/test.jsonl"))
    timing_ids = set(index_rows(read_rows(output / "data/timing.jsonl")))
    seeds = list(map(str, plan["seeds"]))
    if set(selections) != set(seeds):
        raise ValueError("Selected checkpoint seeds differ from plan")
    predictions, accuracy, timing, training = {}, {}, {}, {}
    for seed in seeds:
        predictions[seed], accuracy[seed], timing[seed], training[seed] = {}, {}, {}, {}
        if set(selections[seed]) != set(ARMS):
            raise ValueError("Missing comparison arm")
        for arm in ARMS:
            selected = selections[seed][arm]
            if digest(Path(selected["adapter"]) / "adapter.safetensors") != selected["sha256"]:
                raise ValueError("Selected checkpoint changed")
            path = output / "evaluations" / f"seed-{seed}" / f"{arm}.jsonl"
            rows = audit_predictions(
                path, questions, selected, plan["max_tokens"], plan["decode_strategy"]
            )
            predictions[seed][arm] = rows
            accuracy[seed][arm] = {
                "n": len(rows),
                "correct": sum(row["correct"] for row in rows.values()),
                "accuracy": statistics.mean(row["correct"] for row in rows.values()),
                "truncated": sum(not row["terminated"] for row in rows.values()),
                "mean_generated_tokens": statistics.mean(
                    row["generated_tokens"] for row in rows.values()
                ),
                "mean_transformer_positions": statistics.mean(
                    row["transformer_positions"] for row in rows.values()
                ),
            }
        root = output / "runs" / f"seed-{seed}"
        stage_names = ["warmup", *[f"{arm}-stage-{stage}" for arm in ARMS for stage in range(1, 5)]]
        for name in stage_names:
            stage = root / name
            configuration = json.loads((stage / "run.json").read_text())
            result = json.loads((stage / "result.json").read_text())
            expected = plan["split_sizes"]["train"] * configuration["epochs"]
            if result["examples_seen"] != expected or result["step"] != result["total_steps"]:
                raise ValueError(f"Incomplete training budget: {stage}")
            training[seed][name] = {
                "examples_seen": expected,
                "elapsed_s": result["elapsed_s"],
                "supervised_tokens": result["supervised_tokens"],
                "epoch_validation": [
                    row for row in read_rows(stage / "metrics.jsonl") if "validation" in row
                ],
            }
        for baseline in ("pause", "short_text", "cot"):
            directory = output / "timing" / f"seed-{seed}" / f"feedback-vs-{baseline}"
            timing[seed][baseline] = audit_timing(
                directory,
                timing_ids,
                {
                    "baseline": predictions[seed][baseline],
                    "candidate": predictions[seed]["feedback"],
                },
                plan["timing_repeats"],
            )
    identifiers = sorted(questions)
    matrices = {
        arm: [
            [predictions[seed][arm][identifier]["correct"] for identifier in identifiers]
            for seed in seeds
        ]
        for arm in ARMS
    }
    comparisons = {
        arm: accuracy_comparison(matrices[arm], matrices["feedback"])
        for arm in ("pause", "short_text", "cot")
    }
    sources = [
        p
        for directory in ("runs", "evaluations", "timing")
        for p in (output / directory).rglob("*")
        if p.is_file() and p.suffix in {".json", ".jsonl"}
    ]
    result = {
        "plan_sha256": digest(output / "plan.json"),
        "selection_sha256": digest(output / "selected-checkpoints.json"),
        "artifact_sha256": {str(p.relative_to(output)): digest(p) for p in sources},
        "accuracy_by_seed": accuracy,
        "feedback_comparisons": comparisons,
        "paired_timing_by_seed": timing,
        "decode_strategy": plan["decode_strategy"],
        "training": training,
        "limits": [
            plan["claim_boundary"],
            plan["timing_scope"],
            "Three seeds quantify some training variation, not convergence or general reasoning.",
            "Nominal transformer positions are not FLOPs, energy, or a uniform-cost measure.",
            "Faster wrong answers are retained in latency and counted as accuracy failures.",
        ],
    }
    report = [
        "# GSM8K Coconut curriculum comparison",
        "",
        plan["claim_boundary"],
        "",
        "All test questions are scored once per seed and arm. Latencies come from a separate frozen, repeated, paired subset.",
        "",
        "| Seed | Arm | Correct | Accuracy | Truncated | Mean generated tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        for arm in ARMS:
            row = accuracy[seed][arm]
            report.append(
                f"| {seed} | {arm} | {row['correct']}/{row['n']} | {row['accuracy']:.2%} | {row['truncated']} | {row['mean_generated_tokens']:.1f} |"
            )
    report += [
        "",
        "Positive accuracy deltas favor recurrent feedback. Intervals resample both seeds and paired questions; with three seeds they remain exploratory.",
        "",
        "| Control | Mean accuracy delta | 95% interval | Per-seed deltas |",
        "|---|---:|---:|---|",
    ]
    for arm, row in comparisons.items():
        low, high = row["crossed_bootstrap_ci95"]
        deltas = ", ".join(f"{100 * value:+.2f} pp" for value in row["per_seed_accuracy_delta"])
        report.append(
            f"| {arm} | {100 * row['mean_accuracy_delta']:+.2f} pp | [{100 * low:+.2f}, {100 * high:+.2f}] pp | {deltas} |"
        )
    report += [
        "",
        "Latency reduction is `1 - feedback / control`. Mean and median use per-question median request times; neither is the mean of per-question percentage changes.",
        "",
        "| Seed | Control | Control mean (s) | Feedback mean (s) | Mean reduction | Median reduction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        for arm, row in timing[seed].items():
            report.append(
                f"| {seed} | {arm} | {row['baseline_mean_s']:.3f} | {row['candidate_mean_s']:.3f} | {row['mean_latency_reduction']:.2%} | {row['median_latency_reduction']:.2%} |"
            )
    report += [
        "",
        *result["limits"],
        "",
        "Full per-seed timings, calibration probes, training curves, hashes, and uncertainty estimates are in `summary.json`.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    write_json(output / "summary.json", result)
    print(
        json.dumps({"summary": str(output / "summary.json"), "report": str(output / "report.md")})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output.resolve())
