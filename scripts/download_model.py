"""Download pinned model files without modifying global caches."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", help="Commit SHA; resolved and recorded when omitted")
    args = parser.parse_args()
    revision = HfApi().model_info(args.repo, revision=args.revision or "main").sha
    snapshot_download(
        args.repo,
        revision=revision,
        local_dir=args.output,
        cache_dir=args.output.parent / ".hf-cache",
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "README.md"],
    )
    (args.output / "source.json").write_text(
        json.dumps({"repo": args.repo, "revision": revision}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
