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


def test_selected_latency_field_is_used_for_both_runs():
    a = {**row("a", True, 1.0), "end_to_end_latency_s": 2.0}
    b = {**row("a", True, 0.25), "end_to_end_latency_s": 1.0}
    result = compare([a], [b], latency_field="end_to_end_latency_s")["overall"]
    assert result["latency_field"] == "end_to_end_latency_s"
    assert result["ratio_of_median_latencies"] == 2


@pytest.mark.parametrize("duration", [None, -1.0, 0.0, float("nan"), float("inf")])
def test_invalid_latency_rejected(duration):
    a = {**row("a", True), "end_to_end_latency_s": 2.0}
    b = {**row("a", True), "end_to_end_latency_s": duration}
    with pytest.raises(ValueError, match="end_to_end_latency_s"):
        compare([a], [b], latency_field="end_to_end_latency_s")


def test_old_measurement_cannot_substitute_for_end_to_end_latency():
    a = row("a", True)
    b = {**a, "end_to_end_latency_s": 1.2}
    with pytest.raises(ValueError, match="end_to_end_latency_s"):
        compare([a], [b], latency_field="end_to_end_latency_s")
