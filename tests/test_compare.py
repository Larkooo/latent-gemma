import pytest

from latent_gemma.compare import compare


def row(identifier, correct, latency=1.0):
    return {
        "id": identifier,
        "correct": correct,
        "latency_s": latency,
        "task": "links",
        "answer": "A",
    }


def test_paired_comparison_uses_ids_not_order():
    a = [row("a", True), row("b", False)]
    b = [row("b", True, 0.5), row("a", True, 0.5)]
    result = compare(a, b)["overall"]
    assert result["accuracy_delta"] == 0.5
    assert result["ratio_of_median_latencies"] == 2
    assert result["candidate_only_correct"] == 1


def test_mismatched_samples_rejected():
    with pytest.raises(ValueError, match="identical"):
        compare([row("a", True)], [row("b", True)])


def test_different_expected_answers_rejected():
    a = row("a", True)
    b = {**a, "answer": "B"}
    with pytest.raises(ValueError, match="targets"):
        compare([a], [b])
