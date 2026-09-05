import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .data import generate_dataset, read_examples
from .evaluate import evaluate
from .model import AdapterConfig, load_model
from .provenance import capture, validate_adapter_model
from .train import train


def main():
    parser = argparse.ArgumentParser(description="Continuous-activation Gemma experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data")
    data.add_argument("--output", type=Path, required=True)
    for key, count in [("train", 6000), ("validation", 400), ("test", 600), ("ood", 400)]:
        data.add_argument(f"--{key}", type=int, default=count)
    data.add_argument("--seed", type=int, default=20260905)
    for name in ("train", "evaluate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--model", required=True)
        cmd.add_argument("--adapter", type=Path)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--latent-steps", type=int, default=4)
        cmd.add_argument("--lora-layers", type=int, default=6)
        cmd.add_argument("--rank", type=int, default=16)
        cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument("--hybrid-boundary", choices=["none", "reasoning"])
        cmd.add_argument(
            "--compute-dtype", choices=["auto", "original", "float32", "bfloat16"], default="auto"
        )
        if name == "train":
            cmd.add_argument("--data", type=Path, required=True)
            cmd.add_argument("--steps", type=int, default=500)
            cmd.add_argument("--batch-size", type=int, default=4)
            cmd.add_argument("--learning-rate", type=float, default=2e-4)
            cmd.add_argument("--modes", nargs="+", default=["direct", "cot", "latent"])
            cmd.add_argument("--eval-every", type=int, default=100)
            cmd.add_argument("--log-every", type=int, default=10)
            cmd.add_argument("--reasoning-steps-to-drop", type=int, default=0)
        else:
            cmd.add_argument("--data", type=Path, required=True)
            cmd.add_argument("--decode-strategy", choices=["serial", "pipelined"], default="serial")
            cmd.add_argument(
                "--mode",
                choices=["direct", "cot", "latent", "native", "hybrid", "plain"],
                required=True,
            )
            cmd.add_argument("--limit", type=int)
            cmd.add_argument("--max-tokens", type=int, default=96)
            cmd.add_argument(
                "--ablation", choices=["none", "zero", "shuffle", "repeat"], default="none"
            )
    args = parser.parse_args()
    if args.command == "data":
        print(
            json.dumps(
                generate_dataset(
                    args.output, args.train, args.validation, args.test, args.ood, args.seed
                ),
                indent=2,
            )
        )
        return
    config = AdapterConfig(
        num_layers=args.lora_layers,
        rank=args.rank,
        seed=args.seed,
        compute_dtype=args.compute_dtype,
    )
    hybrid_boundary = args.hybrid_boundary or "none"
    if args.adapter:
        saved = json.loads((args.adapter / "config.json").read_text())
        config = AdapterConfig(**saved["adapter"])
        validate_adapter_model(saved, args.model)
        if args.hybrid_boundary is None:
            hybrid_boundary = saved.get("run", {}).get("hybrid_boundary", "none")
    model, tokenizer = load_model(args.model, config, args.adapter)
    if args.command == "train":
        result = train(
            model,
            tokenizer,
            args.data / "train.jsonl",
            args.data / "validation.jsonl",
            args.output,
            args.model,
            args.steps,
            args.batch_size,
            args.learning_rate,
            args.latent_steps,
            tuple(args.modes),
            args.seed,
            args.eval_every,
            args.log_every,
            str(args.adapter) if args.adapter else None,
            args.reasoning_steps_to_drop,
            hybrid_boundary,
        )
        print(json.dumps(result))
    else:
        examples = read_examples(args.data)
        if args.limit is not None:
            examples = examples[: args.limit]
        metadata = {
            "model": args.model,
            "adapter_path": str(args.adapter),
            "adapter_config": asdict(config),
            "resolved_compute_dtype": model.compute_dtype,
            "data": str(args.data),
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        }
        metadata["provenance"] = capture(
            args.output.with_suffix(".provenance"),
            args.model,
            str(args.adapter) if args.adapter else None,
        )
        evaluate(
            model,
            tokenizer,
            examples,
            args.output,
            args.mode,
            args.latent_steps,
            args.max_tokens,
            args.ablation,
            metadata,
            hybrid_boundary,
            args.decode_strategy,
        )


if __name__ == "__main__":
    main()
