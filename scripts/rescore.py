"""Re-score preserved generations after a parser fix without replacing originals."""

import argparse
import json
import shutil
from pathlib import Path

from latent_gemma import data
from latent_gemma.compare import read_predictions
from latent_gemma.evaluate import summarize
from latent_gemma.provenance import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if any(
        path.exists()
        for path in (
            args.output,
            args.output.with_suffix(".summary.json"),
            args.output.with_suffix(".scoring.py"),
        )
    ):
        raise FileExistsError(args.output)
    original = json.loads(args.source.with_suffix(".summary.json").read_text())
    rows = read_predictions(args.source)
    changed_predictions = 0
    changed_correctness = 0
    for row in rows:
        prediction = data.extract_answer(row["text"], row["task"], row["mode"])
        correct = data.answer_matches(prediction, row["answer"], row["task"])
        changed_predictions += prediction != row["prediction"]
        changed_correctness += correct != row["correct"]
        row.update(prediction=prediction, correct=correct, scoring_policy=data.SCORING_POLICY)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x) + "\n" for x in rows))
    shutil.copy2(data.__file__, args.output.with_suffix(".scoring.py"))
    result = {
        **original,
        "scoring_policy": data.SCORING_POLICY,
        "rescoring": {
            "source": str(args.source),
            "source_sha256": sha256(args.source),
            "source_summary_sha256": sha256(args.source.with_suffix(".summary.json")),
            "parser_sha256": sha256(Path(data.__file__)),
            "scoring_source": str(args.output.with_suffix(".scoring.py")),
            "changed_predictions": changed_predictions,
            "changed_correctness": changed_correctness,
        },
        "predictions_sha256": sha256(args.output),
        **summarize(rows),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
