"""Train one resumable Coconut-style stage, with complete shuffled epochs."""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from latent_gemma.curriculum import StageConfig, train_stage
from latent_gemma.model import AdapterConfig, load_model
from latent_gemma.provenance import validate_adapter_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--feedback-kind", choices=["recurrent", "pause"], default="recurrent")
    parser.add_argument("--lora-layers", type=int, default=35)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--latent-steps", type=int, default=2)
    parser.add_argument("--drop-steps", default="1", help="Number of reasoning steps, or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--memory-gb", type=float, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.memory_gb <= 0:
        parser.error("--memory-gb must be positive")
    dropped = None if args.drop_steps == "all" else int(args.drop_steps)
    config = StageConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_steps=args.latent_steps,
        reasoning_steps_to_drop=dropped,
        seed=args.seed,
        validation_size=args.validation_size,
        validation_max_tokens=args.max_tokens,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
    )
    config.validate()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Use a new output directory or --resume: {args.output}")
    mx.set_memory_limit(int(args.memory_gb * 1024**3))
    mx.set_cache_limit(512 * 1024**2)
    adapter_config = AdapterConfig(
        num_layers=args.lora_layers,
        rank=args.rank,
        seed=args.seed,
        feedback_kind=args.feedback_kind,
    )
    if args.adapter:
        saved = json.loads((args.adapter / "config.json").read_text())
        validate_adapter_model(saved, args.model)
        adapter_config = AdapterConfig(**saved["adapter"])
        if adapter_config.num_layers != args.lora_layers or adapter_config.rank != args.rank:
            raise ValueError("Initialization adapter differs from requested LoRA configuration")
        if adapter_config.feedback_kind == "pause" and args.feedback_kind != "pause":
            raise ValueError("A pause checkpoint cannot initialize recurrent feedback")
    model, tokenizer = load_model(args.model, adapter_config, args.adapter)
    if args.feedback_kind == "pause" and model.config.feedback_kind != "pause":
        mx.random.seed(args.seed)
        model.enable_pause_positions()
    result = train_stage(
        model,
        tokenizer,
        args.data / "train.jsonl",
        args.data / "validation.jsonl",
        args.output,
        args.model,
        config,
        str(args.adapter) if args.adapter else None,
        resume=args.resume,
    )
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
