"""Audit logical inference work in the frozen boundary pilot, without timing claims."""

import hashlib
import json
import statistics
from pathlib import Path


def audit() -> dict:
    source = Path(__file__).resolve().parent.parent / "curriculum-boundary-stage1"
    conditions = {}
    hashes = {}
    for condition, filename in (("text_cot", "cot.jsonl"), ("hybrid_k2", "hybrid.jsonl")):
        path = source / filename
        hashes[str(path.relative_to(source.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary_path = path.with_suffix(".summary.json")
        summary = json.loads(summary_path.read_text())
        if hashes[str(path.relative_to(source.parent))] != summary["predictions_sha256"]:
            raise ValueError("Frozen prediction hash differs from its recorded summary")
        decoder = source / summary["report_export"]["source_directory"] / "evaluate.py"
        decoder_hash = hashlib.sha256(decoder.read_bytes()).hexdigest()
        if decoder_hash != summary["metadata"]["provenance"]["source_sha256"]["evaluate.py"]:
            raise ValueError("Archived decoder differs from the recorded inference source")
        hashes[str(decoder.relative_to(source.parent))] = decoder_hash
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        indexed = {row["id"]: row for row in rows}
        if len(indexed) != len(rows) or len(rows) != 100:
            raise ValueError("Expected 100 unique questions in each frozen pilot condition")
        for row in rows:
            for field in (
                "prompt_tokens",
                "forced_tokens",
                "latent_steps",
                "generated_tokens",
                "transformer_positions",
            ):
                if not isinstance(row[field], int) or row[field] < 0:
                    raise ValueError(f"Invalid work counter: {field}")
            if not row["terminated"] or row["generated_tokens"] < 1:
                raise ValueError("This audit expects completed, nonempty pilot outputs")
            expected = (
                row["prompt_tokens"]
                + row["forced_tokens"]
                + row["latent_steps"]
                + row["generated_tokens"]
                - 1
            )
            if row["transformer_positions"] != expected:
                raise ValueError("Recorded positions disagree with serial decoder accounting")
        conditions[condition] = indexed
    baseline, candidate = conditions.values()
    if baseline.keys() != candidate.keys():
        raise ValueError("Paired question IDs differ")
    for key, row in baseline.items():
        other = candidate[key]
        if any(row[field] != other[field] for field in ("task", "answer", "prompt_tokens")):
            raise ValueError("Paired tasks, targets, or prompt lengths differ")
        if row["latent_steps"] != 0 or row["forced_tokens"] != 0:
            raise ValueError("Unexpected text baseline")
        if other["latent_steps"] != 2 or other["forced_tokens"] != 5:
            raise ValueError("Unexpected hybrid recipe")
    results = {}
    for task in ("overall", *sorted({row["task"] for row in baseline.values()})):
        keys = [key for key, row in baseline.items() if task == "overall" or row["task"] == task]
        result = {"n": len(keys)}
        for name, indexed in conditions.items():
            rows = [indexed[key] for key in keys]
            result[name] = {
                "correct": sum(row["correct"] for row in rows),
                **{
                    f"mean_{field}": statistics.mean(row[field] for row in rows)
                    for field in (
                        "prompt_tokens",
                        "generated_tokens",
                        "forced_tokens",
                        "latent_steps",
                        "transformer_positions",
                    )
                },
                # The frozen serial decoder projects once for each sampled token,
                # including EOS. It has no speculative prefetch or latent head.
                "inferred_mean_vocabulary_projections": statistics.mean(
                    row["generated_tokens"] for row in rows
                ),
            }
        before, after = result["text_cot"], result["hybrid_k2"]
        result["generated_token_reduction_fraction"] = (
            1 - after["mean_generated_tokens"] / before["mean_generated_tokens"]
        )
        result["nominal_transformer_position_change_fraction"] = (
            after["mean_transformer_positions"] / before["mean_transformer_positions"] - 1
        )
        results[task] = result
    return {
        "scope": "Post-observation accounting of the 100-question boundary diagnostic validation pilot; same selected adapter in both modes.",
        "input_sha256": hashes,
        "results": results,
        "interpretation": [
            "Generated-token counts include EOS; forced transition tokens are separate.",
            "The five-token transition is processed as one block; it is not five autoregressive generation steps.",
            "Transformer positions are logical input positions, not measured FLOPs, executed layer counts, GPU time, energy, or bytes moved.",
            "MLX lazy execution and Gemma shared KV can prune unused intermediate work.",
            "Vocabulary projection counts are inferred from the archived serial decoder, not hardware counters.",
            "This audit does not establish faster inference or lower total computation.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
