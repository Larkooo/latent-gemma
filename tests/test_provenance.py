import json

import pytest

from latent_gemma.provenance import sha256, validate_adapter_model


@pytest.fixture
def checkpoint(tmp_path):
    model = tmp_path / "relocated"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "gemma4"}')
    source = {"repo": "example/gemma", "revision": "pinned-revision"}
    (model / "source.json").write_text(json.dumps(source))
    saved = {
        "model_path": "/old/location",
        "run": {
            "provenance": {
                "model_source": source,
                "model_config_sha256": sha256(model / "config.json"),
            }
        },
    }
    return model, saved


def test_pinned_backbone_can_move(checkpoint):
    model, saved = checkpoint
    validate_adapter_model(saved, str(model))


def test_different_revision_rejected_even_at_original_path(checkpoint):
    model, saved = checkpoint
    saved["model_path"] = str(model)
    (model / "source.json").write_text('{"repo": "example/gemma", "revision": "different"}')
    with pytest.raises(ValueError, match="revision"):
        validate_adapter_model(saved, str(model))


def test_modified_model_config_rejected(checkpoint):
    model, saved = checkpoint
    (model / "config.json").write_text('{"model_type": "gemma3"}')
    with pytest.raises(ValueError, match="configuration"):
        validate_adapter_model(saved, str(model))


def test_unpinned_backbone_cannot_silently_move(checkpoint):
    model, saved = checkpoint
    saved.pop("run")
    with pytest.raises(ValueError, match="Unpinned"):
        validate_adapter_model(saved, str(model))
