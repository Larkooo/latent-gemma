"""Repeat the two-versus-three-step comparison on a checkpoint trained with two."""

import argparse
import json
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
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {args.output}")
    saved = json.loads((args.adapter / "config.json").read_text())
    if saved["run"]["latent_steps"] != 2 or saved["run"]["hybrid_boundary"] != "reasoning":
        raise ValueError("This comparison requires the two-step fixed-transition checkpoint")
    validate_adapter_model(saved, args.model)
    examples = read_examples(args.data)
    if len(examples) != 100:
        raise ValueError("Use the complete 100-question validation sample")
    model, tokenizer = load_model(args.model, AdapterConfig(**saved["adapter"]), args.adapter)
    metadata = {
        "model": args.model,
        "adapter_path": str(args.adapter),
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "resolved_compute_dtype": model.compute_dtype,
        "trained_latent_steps": 2,
        "reproduction_script_sha256": sha256(Path(__file__)),
        "provenance": capture(
            args.output.with_suffix(".provenance"), args.model, str(args.adapter)
        ),
    }
    conditions = [
        DecodeCondition("hybrid", steps=count, max_tokens=96, hybrid_boundary="reasoning")
        for count in (2, 3)
    ]
    result = benchmark_pair(
        model,
        tokenizer,
        examples,
        args.output,
        *conditions,
        repeats=3,
        seed=42,
        metadata=metadata,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
