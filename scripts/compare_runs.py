import argparse
import json
from pathlib import Path

from latent_gemma.compare import compare, read_predictions
from latent_gemma.provenance import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--latency-field", choices=["latency_s", "end_to_end_latency_s"], default="latency_s"
    )
    args = parser.parse_args()
    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_sha256": sha256(args.baseline),
        "candidate_sha256": sha256(args.candidate),
        "comparison": compare(
            read_predictions(args.baseline),
            read_predictions(args.candidate),
            latency_field=args.latency_field,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
