"""Paired analysis of compressed-reasoning predictions against gold and a reference.

Reports accuracy, paired transitions against a reference prediction file, how
the generation handles the final value computed in the removed steps
(recomputed in text, used silently, or absent), and generated-line counts
relative to the expected shortened solution.
"""

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(text):
    return [n.replace(",", "") for n in NUMBER.findall(text)]


def read_rows(path):
    return {
        json.loads(line)["id"]: json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line
    }


def value_handling(example, row, dropped):
    steps = [s.strip() for s in example["reasoning"].splitlines() if s.strip()]
    question = set(numbers(example["question"]))
    computed = [n for n in numbers("\n".join(steps[:dropped])) if n not in question]
    if not computed:
        return None
    value = computed[-1]
    text = row["text"].split("Answer:")[0]
    if re.search(r"=\s*\$?" + re.escape(value) + r"(?![\d])", text):
        return "recomputed"
    if re.search(r"(?<![\d.])" + re.escape(value) + r"(?![\d])", text):
        return "silent"
    return "absent"


def analyze(gold, rows, dropped, reference=None):
    ids = list(rows)
    result = {"n": len(ids), "correct": sum(rows[i]["correct"] for i in ids)}
    result["accuracy"] = result["correct"] / len(ids)
    result["truncated"] = sum(not rows[i]["terminated"] for i in ids)
    result["mean_generated_tokens"] = statistics.mean(rows[i]["generated_tokens"] for i in ids)
    if "teacher_forced_loss" in rows[ids[0]]:
        result["mean_teacher_forced_loss"] = statistics.mean(
            rows[i]["teacher_forced_loss"] for i in ids
        )
    handling = collections.defaultdict(lambda: [0, 0])
    deltas = collections.Counter()
    for i in ids:
        kind = value_handling(gold[i], rows[i], dropped)
        if kind:
            handling[kind][0] += 1
            handling[kind][1] += int(rows[i]["correct"])
        expected = len([s for s in gold[i]["reasoning"].splitlines() if s.strip()]) - dropped
        produced = len([s for s in rows[i]["text"].split("Answer:")[0].splitlines() if s.strip()])
        deltas[max(-3, min(3, produced - expected))] += 1
    result["removed_value_handling"] = {
        k: {"n": v[0], "correct": v[1]} for k, v in sorted(handling.items())
    }
    result["generated_lines_minus_expected"] = dict(sorted(deltas.items()))
    if reference:
        transitions = collections.Counter(
            (bool(reference[i]["correct"]), bool(rows[i]["correct"])) for i in ids
        )
        result["paired_vs_reference"] = {
            "both_correct": transitions[(True, True)],
            "reference_only": transitions[(True, False)],
            "candidate_only": transitions[(False, True)],
            "both_wrong": transitions[(False, False)],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--drop-steps", type=int, default=1)
    args = parser.parse_args()
    gold = read_rows(args.gold)
    reference = read_rows(args.reference) if args.reference else None
    report = {}
    for path in args.predictions:
        report[str(path)] = analyze(gold, read_rows(path), args.drop_steps, reference)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
