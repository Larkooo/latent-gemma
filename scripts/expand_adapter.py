"""Extend an existing adapter into earlier transformer layers without retraining."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from latent_gemma.data import Example, prompt_text
from latent_gemma.model import AdapterConfig, load_model, parameter_counts
from latent_gemma.provenance import capture, validate_adapter_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora-layers", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite adapter: {args.output}")
    saved = json.loads((args.adapter / "config.json").read_text())
    config = AdapterConfig(**saved["adapter"])
    validate_adapter_model(saved, args.model)
    model, tokenizer = load_model(args.model, config, args.adapter)
    model.eval()
    probe = Example("expansion-probe", "arithmetic", "Compute (2 + 3) * 4.", "", "20", "probe")
    prompt = mx.array([tokenizer.encode(prompt_text(tokenizer, probe), add_special_tokens=False)])

    def logits(steps):
        state, _ = model.prefill(prompt, steps)
        return np.array(model.logits(state))

    before = {steps: logits(steps) for steps in (0, 2)}
    model.expand_lora(args.lora_layers)
    verification = {}
    for steps, expected in before.items():
        actual = logits(steps)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
        verification[str(steps)] = {
            "max_logit_difference": float(np.max(np.abs(actual - expected)))
        }
    args.output.mkdir(parents=True)
    provenance = capture(args.output, args.model, str(args.adapter))
    run = {
        "model_path": args.model,
        "source_adapter": str(args.adapter),
        "operation": "expand_lora",
        "source_adapter_config": asdict(config),
        "adapter": asdict(model.config),
        "hybrid_boundary": saved.get("run", {}).get("hybrid_boundary", "none"),
        "provenance": provenance,
        "parameters": parameter_counts(model),
        "probe_verification": verification,
    }
    model.save_adapter(args.output, {"model_path": args.model, "run": run})
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
