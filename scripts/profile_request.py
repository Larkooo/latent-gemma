"""Time one warm request per condition in the current session, with a load probe.

Reproduces the quiet-session check: the same two adapters and the same
`generate` path as `benchmark_pair`, run alone and alternating, plus a
per-phase breakdown of one request. Absolute numbers depend on concurrent load;
the session calibration probe is recorded alongside them.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from latent_gemma.benchmark import session_calibration
from latent_gemma.data import hybrid_boundary_text, prompt_text, read_examples
from latent_gemma.evaluate import generate
from latent_gemma.model import AdapterConfig, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--baseline-adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--latent-steps", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")

    def load(path):
        saved = json.loads((path / "config.json").read_text())
        return load_model(args.model, AdapterConfig(**saved["adapter"]), path)

    baseline, tokenizer = load(args.baseline_adapter)
    candidate, _ = load(args.adapter)
    saved = json.loads((args.adapter / "config.json").read_text())
    boundary = saved.get("run", {}).get("hybrid_boundary", "none")
    examples = read_examples(args.data)[: args.limit + 1]
    conditions = {
        "baseline": (baseline, dict(mode="cot", steps=0, max_tokens=96, hybrid_boundary="none")),
        "candidate": (
            candidate,
            dict(mode="hybrid", steps=args.latent_steps, max_tokens=96, hybrid_boundary=boundary),
        ),
    }
    for model, settings in conditions.values():
        for _ in range(2):
            generate(model, tokenizer, examples[0], **settings)
    report = {"calibration_before": session_calibration(), "runs": {}}

    def run(side, example):
        model, settings = conditions[side]
        row = generate(model, tokenizer, example, **settings)
        return {
            "end_to_end_latency_s": row["end_to_end_latency_s"],
            "tokens": row["generated_tokens"],
        }

    for side in conditions:
        report["runs"][f"{side}_alone"] = [run(side, e) for e in examples[1:]]
    alternating = {side: [] for side in conditions}
    for example in examples[1:]:
        for side in conditions:
            alternating[side].append(run(side, example))
    for side, rows in alternating.items():
        report["runs"][f"{side}_alternating"] = rows

    example = examples[3]
    breakdown = {}
    for side, (model, settings) in conditions.items():
        mx.synchronize()
        t0 = time.perf_counter()
        ids = tokenizer.encode(prompt_text(tokenizer, example), add_special_tokens=False)
        t1 = time.perf_counter()
        state, cache = model.prefill(mx.array([ids]), settings["steps"])
        mx.eval(state)
        mx.synchronize()
        t2 = time.perf_counter()
        suffix = hybrid_boundary_text(settings["hybrid_boundary"])
        if suffix:
            forced = mx.array([tokenizer.encode(suffix, add_special_tokens=False)])
            state = model.hidden(forced, cache=cache)[:, -1:, :]
            mx.eval(state)
            mx.synchronize()
        t3 = time.perf_counter()
        stops = set(getattr(tokenizer, "eos_token_ids", [tokenizer.eos_token_id]))
        steps = []
        for _ in range(96):
            started = time.perf_counter()
            next_id = mx.argmax(model.logits(state[:, -1:, :]), axis=-1).item()
            if next_id in stops:
                steps.append(time.perf_counter() - started)
                break
            state = model.hidden(mx.array([[next_id]]), cache=cache)
            steps.append(time.perf_counter() - started)
        mx.synchronize()
        t4 = time.perf_counter()
        breakdown[side] = {
            "prompt_tokens": len(ids),
            "template_encode_s": t1 - t0,
            "prefill_and_latent_s": t2 - t1,
            "forced_transition_s": t3 - t2,
            "decode_steps": len(steps),
            "decode_total_s": t4 - t3,
            "decode_step_median_s": statistics.median(steps),
        }
    report["breakdown"] = breakdown
    report["calibration_after"] = session_calibration()
    summary = {
        name: {
            "median_s": statistics.median(r["end_to_end_latency_s"] for r in rows),
            "mean_tokens": statistics.mean(r["tokens"] for r in rows),
        }
        for name, rows in report["runs"].items()
    }
    report["summary"] = summary
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "breakdown": breakdown}, indent=2))


if __name__ == "__main__":
    main()
