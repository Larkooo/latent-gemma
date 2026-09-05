import json

import pytest

from latent_gemma.benchmark import DecodeCondition, benchmark_pair
from latent_gemma.data import Example


def fake_generate(model, tokenizer, example, mode, **kwargs):
    return {
        "id": example.id,
        "task": example.task,
        "answer": example.answer,
        "correct": True,
        "prediction": example.answer,
        "text": example.answer,
        "latency_s": 1.0,
        "end_to_end_latency_s": 2.0 if mode == "cot" else 1.0,
        "generated_tokens": 2,
        "prompt_tokens": 3,
        "forced_tokens": 0,
        "terminated": True,
        "transformer_positions": 4,
        "peak_memory_bytes": 0,
    }


@pytest.mark.parametrize("separate_models", [False, True])
def test_repeats_reduce_timing_noise_without_multiplying_accuracy_samples(
    tmp_path, monkeypatch, separate_models
):
    calls = []
    candidate_model = object()
    baseline_model = object() if separate_models else candidate_model

    def generate(*args, **kwargs):
        assert args[0] is (baseline_model if kwargs["mode"] == "cot" else candidate_model)
        row = fake_generate(*args, **kwargs)
        calls.append(row["id"])
        return row

    monkeypatch.setattr("latent_gemma.benchmark.generate", generate)
    monkeypatch.setattr("latent_gemma.benchmark.mx.reset_peak_memory", lambda: None)
    examples = [Example(str(i), "links", "Question", "", "A", "validation") for i in range(2)]
    result = benchmark_pair(
        candidate_model,
        None,
        examples,
        tmp_path / "run",
        DecodeCondition("cot"),
        DecodeCondition("hybrid", 2),
        repeats=4,
        baseline_model=baseline_model,
    )
    assert calls.count("warmup") == 2
    assert len(calls) == 18
    assert result["comparison"]["overall"]["n"] == 2
    assert result["comparison"]["overall"]["ratio_of_median_latencies"] == 2
    assert result["shared_model"] is (not separate_models)
    rows = [
        json.loads(line) for line in (tmp_path / "run/measurements.jsonl").read_text().splitlines()
    ]
    first = [row["condition"] for row in rows if row["order"] == 0]
    assert first.count("baseline") == first.count("candidate") == 4
    assert len((tmp_path / "run/baseline.jsonl").read_text().splitlines()) == 2


def test_different_outputs_across_repeats_are_not_silently_aggregated(tmp_path, monkeypatch):
    count = 0

    def generate(*args, **kwargs):
        nonlocal count
        row = fake_generate(*args, **kwargs)
        count += 1
        row["text"] = str(count)
        return row

    monkeypatch.setattr("latent_gemma.benchmark.generate", generate)
    monkeypatch.setattr("latent_gemma.benchmark.mx.reset_peak_memory", lambda: None)
    examples = [Example("test", "links", "Question", "", "A", "validation")]
    with pytest.raises(ValueError, match="Nondeterministic generated output"):
        benchmark_pair(
            None,
            None,
            examples,
            tmp_path / "run",
            DecodeCondition("cot"),
            DecodeCondition("hybrid", 2),
        )
    assert (tmp_path / "run/measurements.jsonl").exists()
    assert not (tmp_path / "run/result.json").exists()
