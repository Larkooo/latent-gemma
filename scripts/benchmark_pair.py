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
    parser.add_argument("--baseline-adapter", type=Path)
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
    baseline_model = model
    baseline_saved = saved
    baseline_path = args.baseline_adapter or args.adapter
    if baseline_path.resolve() != args.adapter.resolve():
        baseline_saved = json.loads((baseline_path / "config.json").read_text())
        validate_adapter_model(baseline_saved, args.model)
        baseline_model, _ = load_model(
            args.model, AdapterConfig(**baseline_saved["adapter"]), baseline_path
        )
        if baseline_model.compute_dtype != model.compute_dtype:
            raise ValueError("Paired timing requires matching computation dtypes")
    baseline_boundary = baseline_saved.get("run", {}).get("hybrid_boundary", "none")
    baseline = DecodeCondition(
        args.baseline_mode, max_tokens=args.max_tokens, hybrid_boundary=baseline_boundary
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
        "baseline_adapter_path": str(baseline_path),
        "baseline_adapter_config": baseline_saved["adapter"],
        "baseline_adapter_sha256": sha256(baseline_path / "adapter.safetensors"),
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
        baseline_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
