"""Teacher-forced loss on the values a compressed model must compute silently.

For every question, the numbers first computed in the removed reasoning steps
and later reused in the remaining text are located in the supervised target.
Their token loss and top-1 accuracy show whether latent positions carry that
information, independently of the many easy tokens that dominate mean loss.
"""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from latent_gemma.data import encode_example, read_examples, remaining_reasoning
from latent_gemma.model import AdapterConfig, load_model
from latent_gemma.provenance import validate_adapter_model
from latent_gemma.train import make_batch

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> list[str]:
    return [n.replace(",", "") for n in NUMBER.findall(text)]


def carried_values(example, dropped: int) -> list[str]:
    steps = [line.strip() for line in example.reasoning.splitlines() if line.strip()]
    removed = "\n".join(steps[:dropped])
    kept = remaining_reasoning(example, dropped)
    question = set(numbers(example.question))
    computed = [n for n in numbers(removed) if n not in question]
    reused = set(numbers(kept))
    seen = []
    for n in computed:
        if n in reused and n not in seen:
            seen.append(n)
    return seen


def token_spans(tokenizer, ids: list[int]) -> list[tuple[int, int]]:
    spans, previous = [], 0
    for i in range(1, len(ids) + 1):
        current = len(tokenizer.decode(ids[:i]))
        spans.append((previous, current))
        previous = current
    return spans


def carried_token_indices(tokenizer, ids, mask, values):
    """First-occurrence token indices of every carried value in the supervised text."""
    text = tokenizer.decode(ids)
    spans = token_spans(tokenizer, ids)
    found = {}
    for value in values:
        pattern = re.compile(r"(?<![\d.])" + re.escape(value) + r"(?![\d])")
        for match in pattern.finditer(text):
            start, end = match.span()
            indices = [i for i, (a, b) in enumerate(spans) if a < end and b > start and mask[i]]
            if indices:
                found[value] = indices
                break
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-steps", type=int, required=True)
    parser.add_argument("--drop-steps", type=int, required=True)
    parser.add_argument("--ablation", default="none")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    saved = json.loads((args.adapter / "config.json").read_text())
    validate_adapter_model(saved, args.model)
    model, tokenizer = load_model(args.model, AdapterConfig(**saved["adapter"]), args.adapter)
    model.eval()
    examples = read_examples(args.data)[: args.limit]
    rows = []
    for example in examples:
        prompt, ids, mask = encode_example(
            tokenizer, example, "hybrid", args.drop_steps, "reasoning"
        )
        batch = make_batch([(prompt, ids, mask)], tokenizer.pad_token_id or 0)
        logits = model.answer_logits(batch[0], batch[1], args.latent_steps, args.ablation)
        logits = logits.astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, batch[1], reduction="none")[0]
        top1 = mx.argmax(logits, axis=-1)[0] == batch[1][0]
        mx.eval(losses, top1)
        losses, top1 = losses.tolist(), top1.tolist()
        values = carried_values(example, args.drop_steps)
        found = carried_token_indices(tokenizer, ids, mask, values)
        supervised = [i for i, m in enumerate(mask) if m]
        first = supervised[0]
        carried = sorted({i for indices in found.values() for i in indices})
        rows.append(
            {
                "id": example.id,
                "carried_values": values,
                "carried_found": {v: [ids[i] for i in idx] for v, idx in found.items()},
                "carried_token_loss": sum(losses[i] for i in carried) / len(carried)
                if carried
                else None,
                "carried_token_top1": sum(top1[i] for i in carried) / len(carried)
                if carried
                else None,
                "carried_all_values_top1": all(all(top1[i] for i in idx) for idx in found.values())
                if found
                else None,
                "first_token_loss": losses[first],
                "first_token_top1": top1[first],
                "mean_loss": sum(losses[i] for i in supervised) / len(supervised),
                "supervised_tokens": len(supervised),
                "carried_tokens": len(carried),
            }
        )
    carried_rows = [r for r in rows if r["carried_tokens"]]
    summary = {
        "adapter": str(args.adapter),
        "adapter_sha256": saved.get("adapter_sha256"),
        "latent_steps": args.latent_steps,
        "drop_steps": args.drop_steps,
        "ablation": args.ablation,
        "n": len(rows),
        "n_with_carried_values": len(carried_rows),
        "mean_loss": sum(r["mean_loss"] for r in rows) / len(rows),
        "first_token_loss": sum(r["first_token_loss"] for r in rows) / len(rows),
        "first_token_top1": sum(r["first_token_top1"] for r in rows) / len(rows),
        "carried_token_loss": sum(r["carried_token_loss"] for r in carried_rows)
        / max(1, len(carried_rows)),
        "carried_token_top1": sum(r["carried_token_top1"] for r in carried_rows)
        / max(1, len(carried_rows)),
        "questions_all_carried_values_top1": sum(
            bool(r["carried_all_values_top1"]) for r in carried_rows
        ),
    }
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
