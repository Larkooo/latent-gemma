import inspect
import json
from types import SimpleNamespace

import pytest

from latent_gemma import cli


@pytest.mark.parametrize("command", ["train", "evaluate"])
def test_decode_option_reaches_evaluation_without_changing_training(tmp_path, monkeypatch, command):
    observed = {}
    train_signature = inspect.signature(cli.train)
    evaluate_signature = inspect.signature(cli.evaluate)

    def train(*args, **kwargs):
        train_signature.bind(*args, **kwargs)
        observed["command"] = "train"
        return {}

    def evaluate(*args, **kwargs):
        arguments = evaluate_signature.bind(*args, **kwargs).arguments
        observed.update(command="evaluate", decode_strategy=arguments["decode_strategy"])

    monkeypatch.setattr(
        cli, "load_model", lambda *args: (SimpleNamespace(compute_dtype="float32"), None)
    )
    monkeypatch.setattr(cli, "capture", lambda *args: {})
    monkeypatch.setattr(cli, "train", train)
    monkeypatch.setattr(cli, "evaluate", evaluate)
    data = tmp_path / "questions.jsonl"
    data.write_text(
        json.dumps(
            {
                "id": "question",
                "task": "links",
                "question": "Question",
                "reasoning": "",
                "answer": "A",
                "split": "validation",
            }
        )
        + "\n"
    )
    argv = [
        "latent-gemma",
        command,
        "--model",
        "unused",
        "--data",
        str(data),
        "--output",
        str(tmp_path / "output"),
    ]
    if command == "evaluate":
        argv += ["--mode", "hybrid", "--decode-strategy", "pipelined"]
    monkeypatch.setattr("sys.argv", argv)
    cli.main()
    assert observed["command"] == command
    if command == "evaluate":
        assert observed["decode_strategy"] == "pipelined"
