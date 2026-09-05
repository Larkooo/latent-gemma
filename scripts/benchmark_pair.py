"""Compare decoding paths with counterbalanced order and repeated measurements."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from latent_gemma.benchmark import DecodeCondition, benchmark_pair
from latent_gemma.data import read_examples
from latent_gemma.model import AdapterConfig, load_model
from latent_gemma.provenance import capture, sha256, validate_adapter_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-mode", choices=["cot", "direct", "hybrid"], default="cot")
    parser.add_argument("--candidate-mode", choices=["latent", "hybrid"], default="hybrid")
    parser.add_argument("--latent-steps", type=int, default=2)
    parser.add_argument(
        "--candidate-ablation", choices=["none", "zero", "shuffle", "repeat"], default="none"
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {args.output}")
    if args.limit <= 0 or args.latent_steps < 0 or args.max_tokens <= 0 or args.repeats < 2:
        raise ValueError("Invalid benchmark counts")
    saved = json.loads((args.adapter / "config.json").read_text())
    config = AdapterConfig(**saved["adapter"])
    validate_adapter_model(saved, args.model)
    model, tokenizer = load_model(args.model, config, args.adapter)
    boundary = saved.get("run", {}).get("hybrid_boundary", "none")
    baseline = DecodeCondition(
        args.baseline_mode, max_tokens=args.max_tokens, hybrid_boundary=boundary
    )
    candidate = DecodeCondition(
        args.candidate_mode, args.latent_steps, args.max_tokens, args.candidate_ablation, boundary
    )
    metadata = {
        "model": args.model,
        "adapter_path": str(args.adapter),
        "adapter_config": asdict(config),
        "resolved_compute_dtype": model.compute_dtype,
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "provenance": capture(
            args.output.with_suffix(".provenance"), args.model, str(args.adapter)
        ),
    }
    result = benchmark_pair(
        model,
        tokenizer,
        read_examples(args.data)[: args.limit],
        args.output,
        baseline,
        candidate,
        args.repeats,
        args.seed,
        metadata,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
