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
