"""Replay the trainer's seeded Python sampler to count data exposure."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from latent_gemma.data import prompt_text, read_examples

root = Path(__file__).resolve().parent.parent
tokenizer = AutoTokenizer.from_pretrained(root / "work/models/gemma-4-e2b-it-4bit", local_files_only=True)
dataset = root / "work/data/diagnostics/train.jsonl"
examples = read_examples(dataset)
buckets = defaultdict(list)
for example in examples:
    length = len(tokenizer.encode(prompt_text(tokenizer, example), add_special_tokens=False))
    buckets[length].append(example)
groups = list(buckets.values())
result = {"train_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "dataset_examples": len(examples), "runs": {}}
for name in ("gemma4-warmup-v2", "gemma4-latent-k4-v2", "gemma4-curriculum-stage1"):
    config = json.loads((root / "work/runs" / name / "run.json").read_text())
    selected = json.loads((root / "work/runs" / name / "best/config.json").read_text())["step"]
    assert config["train_sha256"] == result["train_sha256"]
    rng = random.Random(config["seed"])
    draws = defaultdict(list)
    for step in range(1, selected + 1):
        mode = config["modes"][(step - 1) % len(config["modes"])]
        group = rng.choices(groups, weights=[len(g) for g in groups])[0]
        draws[mode].extend(rng.choices(group, k=config["batch_size"]))
    result["runs"][name] = {
        "selected_step": selected,
        "modes": {
            mode: {
                "draws": len(items), "unique_examples": len({x.id for x in items}),
                "draws_by_task": dict(Counter(x.task for x in items)),
                "unique_by_task": {task: len({x.id for x in items if x.task == task}) for task in sorted({x.task for x in items})},
            }
            for mode, items in draws.items()
        },
    }
(root / "work/runs/training-exposure-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
