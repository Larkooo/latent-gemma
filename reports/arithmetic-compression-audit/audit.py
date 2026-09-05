"""Audit the preserved arithmetic continuations without decoding latent states."""

import hashlib
import json
import re
from pathlib import Path

reports = Path(__file__).resolve().parent.parent
data_path = reports / "curriculum-boundary-stage1/validation-sample.jsonl"
examples = {row["id"]: row for row in map(json.loads, data_path.read_text().splitlines())}
result = {
    "metric": "Correct final answer and exact remaining annotated equation, ignoring whitespace and final periods.",
    "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    "runs": {},
}


def canonical(text):
    return re.sub(r"\s+", "", text).rstrip(".")


for name, relative in {
    "latent": "curriculum-boundary-stage1/hybrid.jsonl",
    "trained_control": "matched-short-text-control/hybrid.jsonl",
}.items():
    path = reports / relative
    rows = []
    for prediction in map(json.loads, path.read_text().splitlines()):
        example = examples[prediction["id"]]
        if example["task"] != "arithmetic":
            continue
        expected = example["reasoning"].split(". ", 1)[1]
        generated = re.split(r"Answer:\s*", prediction["text"], maxsplit=1, flags=re.IGNORECASE)[0].strip()
        exact = canonical(generated) == canonical(expected)
        rows.append({
            "id": example["id"],
            "expected_continuation": expected,
            "generated_continuation": generated,
            "canonical_remaining_equation": exact,
            "final_correct": prediction["correct"],
            "compressed_correct": exact and prediction["correct"],
        })
    result["runs"][name] = {
        "source": relative,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n": len(rows),
        "compressed_correct": sum(row["compressed_correct"] for row in rows),
        "final_correct": sum(row["final_correct"] for row in rows),
        "correct_noncanonical_continuations": sum(row["final_correct"] and not row["canonical_remaining_equation"] for row in rows),
        "rows": rows,
    }

print(json.dumps(result, indent=2))
