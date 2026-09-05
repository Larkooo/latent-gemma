"""Record the exact local inputs used by an experiment."""

import hashlib
import json
import platform
import shutil
import sys
from importlib.metadata import version
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adapter_model(saved: dict, model_path: str) -> None:
    """Allow relocated pinned checkpoints while rejecting a different backbone."""
    model = Path(model_path)
    recorded = saved.get("run", {}).get("provenance", {})
    expected_source = recorded.get("model_source")
    source_path = model / "source.json"
    if expected_source and expected_source.get("revision") and source_path.exists():
        actual_source = json.loads(source_path.read_text())
        if any(actual_source.get(key) != expected_source.get(key) for key in ("repo", "revision")):
            raise ValueError(
                "Adapter backbone repository/revision differs from the requested model"
            )
        if sha256(model / "config.json") != recorded.get("model_config_sha256"):
            raise ValueError("Adapter backbone configuration differs from the requested model")
    elif Path(saved["model_path"]).resolve() != model.resolve():
        raise ValueError("Unpinned adapter backbone path differs from the requested model")


def capture(directory: Path, model_path: str, adapter_path: str | None = None) -> dict:
    source = Path(__file__).parent
    snapshot = directory / "source"
    snapshot.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for path in sorted(source.glob("*.py")):
        hashes[path.name] = sha256(path)
        shutil.copy2(path, snapshot / path.name)
    model = Path(model_path)
    source_info = model / "source.json"
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {p: version(p) for p in ["mlx", "mlx-lm", "numpy", "transformers"]},
        "source_sha256": hashes,
        "model_config_sha256": sha256(model / "config.json"),
        "model_source": json.loads(source_info.read_text()) if source_info.exists() else None,
        "adapter_sha256": sha256(Path(adapter_path) / "adapter.safetensors")
        if adapter_path
        else None,
    }
    (directory / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
