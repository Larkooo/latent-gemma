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


def test_latency_bootstrap_preserves_question_pairing():
    a = [row(str(i), True, duration) for i, duration in enumerate([1.0, 10.0, 100.0, 1000.0])]
    b = [{**r, "latency_s": r["latency_s"] / 2} for r in reversed(a)]
    result = compare(a, b)["overall"]
    assert result["ratio_of_median_latencies_ci95"] == [2.0, 2.0]
    assert result["median_paired_speedup"] == 2.0
    assert result["median_paired_speedup_ci95"] == [2.0, 2.0]
    assert result["fraction_questions_faster"] == 1.0


def test_identical_timings_have_no_speedup():
    rows = [row("a", True, 0.25), row("b", False, 25)]
    result = compare(rows, rows)["overall"]
    assert result["ratio_of_median_latencies_ci95"] == [1.0, 1.0]
    assert result["median_paired_speedup_ci95"] == [1.0, 1.0]
    assert result["fraction_questions_faster"] == 0.0


def test_latency_intervals_cover_variability_and_are_reproducible():
    a = [row(str(i), True) for i in range(4)]
    b = [row(str(i), True, duration) for i, duration in enumerate([0.5, 0.5, 2.0, 2.0])]
    result = compare(a, b)
    assert result == compare(a, b)
    lower, upper = result["overall"]["median_paired_speedup_ci95"]
    assert lower < 1.0 < upper
    assert result["overall"]["fraction_questions_faster"] == 0.5


def test_different_scoring_policies_require_rescoring():
    a = row("a", False)
    b = {**row("a", True), "scoring_policy": "numeric-equivalence-v1"}
    with pytest.raises(ValueError, match="Scoring policies differ"):
        compare([a], [b])
