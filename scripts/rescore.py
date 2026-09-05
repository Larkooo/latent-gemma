"""Re-score preserved generations after a parser fix without replacing originals."""

import argparse
import json
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
    if args.output.exists():
        raise FileExistsError(args.output)
    original = json.loads(args.source.with_suffix(".summary.json").read_text())
    rows = read_predictions(args.source)
    for row in rows:
        row["prediction"] = data.extract_answer(row["text"], row["task"], row["mode"])
        row["correct"] = row["prediction"] == row["answer"]
    args.output.write_text("".join(json.dumps(x) + "\n" for x in rows))
    result = {
        "metadata": original["metadata"],
        "rescoring": {
            "source": str(args.source),
            "source_sha256": sha256(args.source),
            "parser_sha256": sha256(Path(data.__file__)),
        },
        "mode": original["mode"],
        "steps": original["steps"],
        "ablation": original["ablation"],
        "predictions_sha256": sha256(args.output),
        **summarize(rows),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
