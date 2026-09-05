"""Fetch the official GSM8K release; only the training set supplies validation."""

import argparse
import hashlib
import json
import random
import re
import urllib.request
from dataclasses import asdict
from pathlib import Path

from latent_gemma.data import Example, semantic_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--revision", default="3101c7d5072418e28b9008a6636bde82a006892c")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = args.revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Supply an immutable 40-character repository commit SHA")
    manifest = {
        "source": "https://github.com/openai/grade-school-math",
        "revision": revision,
        "seed": args.seed,
        "license": "MIT",
        "raw_sha256": {},
        "sha256": {},
    }
    records = {}
    for split in ("train", "test"):
        url = f"https://raw.githubusercontent.com/openai/grade-school-math/{revision}/grade_school_math/data/{split}.jsonl"
        body = urllib.request.urlopen(url).read()
        manifest["raw_sha256"][split] = hashlib.sha256(body).hexdigest()
        examples = []
        for line in body.splitlines():
            row = json.loads(line)
            reasoning, answer = row["answer"].rsplit("####", 1)
            reasoning = re.sub(r"<<[^>]+>>", "", reasoning).strip()
            examples.append(
                Example(
                    semantic_id("gsm8k", row["question"]),
                    "gsm8k",
                    row["question"],
                    reasoning,
                    answer.strip().replace(",", ""),
                    split,
                )
            )
        random.Random(args.seed).shuffle(examples)
        records[split] = examples
    validation = records["train"][: args.validation_size]
    records["validation"] = [
        Example(x.id, x.task, x.question, x.reasoning, x.answer, "validation") for x in validation
    ]
    records["train"] = records["train"][args.validation_size :]
    for split, rows in records.items():
        body = "".join(json.dumps(asdict(x), sort_keys=True) + "\n" for x in rows)
        (args.output / f"{split}.jsonl").write_text(body)
        manifest["sha256"][split] = hashlib.sha256(body.encode()).hexdigest()
    manifest["sizes"] = {k: len(v) for k, v in records.items()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
