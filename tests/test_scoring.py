import json
import runpy
from pathlib import Path

import pytest

from latent_gemma.data import SCORING_POLICY, answer_matches
from latent_gemma.evaluate import summarize


@pytest.mark.parametrize(
    "prediction,expected,match",
    [
        ("4.00", "4", True),
        ("4", "4.000", True),
        ("1,000.00", "1000", True),
        ("-0", "0.00", True),
        ("4.000000000000001", "4", False),
        ("NaN", "NaN", False),
        ("Infinity", "Infinity", False),
        ("unknown", "4", False),
        (None, "4", False),
    ],
)
def test_numeric_values_use_exact_decimal_equality(prediction, expected, match):
    assert answer_matches(prediction, expected, "gsm8k") is match


def test_link_answers_remain_exact_labels():
    assert answer_matches("A", "A", "links")
    assert not answer_matches("a", "A", "links")


def test_rescoring_preserves_raw_records_and_evaluation_settings(tmp_path, monkeypatch):
    source = tmp_path / "original.jsonl"
    output = tmp_path / "rescored.jsonl"
    rows = [
        {
            "id": str(i),
            "task": "gsm8k",
            "mode": "cot",
            "prediction": value,
            "answer": "4",
            "correct": False,
            "text": f"Answer: ${value}",
            "latency_s": 1.0,
            "end_to_end_latency_s": 1.1,
            "generated_tokens": 4,
            "transformer_positions": 8,
            "terminated": True,
        }
        for i, value in enumerate(("4.00", "4.01"))
    ]
    body = "".join(json.dumps(row) + "\n" for row in rows)
    source.write_text(body)
    source.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "metadata": {"source": "test"},
                "mode": "cot",
                "steps": 0,
                "ablation": "none",
                "max_tokens": 96,
                "hybrid_boundary": "none",
                "timing_scope": {"end_to_end_latency_s": "original timing scope"},
                **summarize(rows),
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["rescore", str(source), str(output)])
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/rescore.py"), run_name="__main__"
    )
    changed = [json.loads(line) for line in output.read_text().splitlines()]
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert source.read_text() == body
    assert [row["correct"] for row in changed] == [True, False]
    assert [row["prediction"] for row in changed] == ["4.00", "4.01"]
    assert summary["rescoring"]["changed_correctness"] == 1
    assert summary["rescoring"]["changed_predictions"] == 0
    assert summary["max_tokens"] == 96
    assert summary["timing_scope"]["end_to_end_latency_s"] == "original timing scope"
    assert summary["scoring_policy"] == SCORING_POLICY
    assert output.with_suffix(".scoring.py").is_file()
