import importlib
import json
from pathlib import Path

import pytest

from latent_gemma.data import SCORING_POLICY


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_campaign")


@pytest.fixture
def report(campaign):
    return importlib.import_module("summarize_campaign")


def test_curriculum_controls_match_removed_reasoning_and_position_budget(campaign):
    for stage in range(1, 5):
        feedback, pause, short, cot = [campaign.stage_settings(arm, stage) for arm in campaign.ARMS]
        assert feedback["latent_steps"] == pause["latent_steps"] == 2 * min(stage, 3)
        assert feedback["drop_steps"] == pause["drop_steps"] == short["drop_steps"]
        assert short["latent_steps"] == cot["latent_steps"] == 0
        assert cot["drop_steps"] == "0"
    assert campaign.stage_settings("feedback", 4)["drop_steps"] == "all"


def test_partial_evaluation_is_preserved_for_restart(campaign, tmp_path):
    directory = tmp_path / "timing/seed-42/feedback-vs-pause"
    directory.mkdir(parents=True)
    (directory / "measurements.jsonl").write_text("partial measurements\n")
    campaign.archive_partial(tmp_path, [directory])
    assert not directory.exists()
    archived = list((tmp_path / "interrupted-attempts").rglob("measurements.jsonl"))
    assert len(archived) == 1 and archived[0].read_text() == "partial measurements\n"


def test_accuracy_comparison_keeps_seed_and_question_pairing(report):
    baseline = [[1, 0, 1, 0]] * 3
    identical = report.accuracy_comparison(baseline, baseline, draws=50)
    assert identical["mean_accuracy_delta"] == 0
    assert identical["crossed_bootstrap_ci95"] == [0, 0]
    improved = report.accuracy_comparison(baseline, [[1, 1, 1, 0]] * 3, draws=50)
    assert improved["per_seed_accuracy_delta"] == [0.25] * 3
    assert improved["mean_accuracy_delta"] == 0.25


def test_prediction_audit_rejects_wrong_scores_and_missing_questions(report, tmp_path):
    path = tmp_path / "predictions.jsonl"
    row = {
        "id": "a",
        "task": "gsm8k",
        "text": "Answer: 5",
        "prediction": "5",
        "answer": "5",
        "correct": True,
        "scoring_policy": SCORING_POLICY,
        "mode": "hybrid",
        "hybrid_boundary": "reasoning",
        "latent_steps": 6,
        "decode_strategy": "serial",
        "ablation": "none",
        "generated_tokens": 3,
        "token_ids": [1, 2, 3],
    }
    questions = {"a": {"answer": "5", "task": "gsm8k"}}

    def save():
        path.write_text(json.dumps(row) + "\n")
        path.with_suffix(".summary.json").write_text(
            json.dumps(
                {
                    "predictions_sha256": report.digest(path),
                    "overall": {"correct": 1, "n": 1},
                }
            )
        )

    save()
    assert report.audit_predictions(path, questions, {"latent_steps": 6}, 384)["a"]["correct"]
    row["correct"] = False
    save()
    with pytest.raises(ValueError, match="score cannot be reproduced"):
        report.audit_predictions(path, questions, {"latent_steps": 6}, 384)
    with pytest.raises(ValueError, match="coverage"):
        report.audit_predictions(path, {**questions, "b": questions["a"]}, {"latent_steps": 6}, 384)


def test_complete_report_audits_repeats_training_and_scores(report, campaign, tmp_path):
    plan = {
        "runtime": campaign.runtime(),
        "sha256": {},
        "model_sha256": {},
        "seeds": [42, 43, 44],
        "split_sizes": {"train": 2},
        "max_tokens": 384,
        "timing_repeats": 2,
        "decode_strategy": "serial",
        "claim_boundary": "Test fixture",
        "timing_scope": "Test timings",
    }
    campaign.write_json(tmp_path / "plan.json", plan)
    data = tmp_path / "data"
    data.mkdir()
    questions = [{"id": identifier, "task": "gsm8k", "answer": "5"} for identifier in ("a", "b")]
    for split in ("test", "timing"):
        (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in questions))
    selections = {}
    for seed in map(str, plan["seeds"]):
        selections[seed] = {}
        root = tmp_path / "runs" / f"seed-{seed}"
        names = [
            "warmup",
            *[f"{arm}-stage-{stage}" for arm in campaign.ARMS for stage in range(1, 5)],
        ]
        for name in names:
            stage = root / name
            stage.mkdir(parents=True)
            campaign.write_json(stage / "run.json", {"epochs": 1})
            campaign.write_json(
                stage / "result.json",
                {
                    "examples_seen": 2,
                    "step": 1,
                    "total_steps": 1,
                    "elapsed_s": 1.0,
                    "supervised_tokens": 6,
                },
            )
            (stage / "metrics.jsonl").write_text('{"step":1,"validation":{"correct":2}}\n')
        predictions = {}
        for arm in campaign.ARMS:
            checkpoint = root / f"{arm}-stage-4"
            (checkpoint / "adapter.safetensors").write_bytes(b"fixture")
            steps = 6 if arm in ("feedback", "pause") else 0
            selections[seed][arm] = {
                "adapter": str(checkpoint),
                "sha256": campaign.digest(checkpoint / "adapter.safetensors"),
                "latent_steps": steps,
            }
            rows = [
                {
                    **question,
                    "mode": "hybrid",
                    "latent_steps": steps,
                    "ablation": "none",
                    "hybrid_boundary": "reasoning",
                    "decode_strategy": "serial",
                    "prediction": "5",
                    "correct": True,
                    "scoring_policy": SCORING_POLICY,
                    "text": "Answer: 5",
                    "generated_tokens": 3,
                    "token_ids": [1, 2, 3],
                    "prompt_tokens": 5,
                    "forced_tokens": 5,
                    "terminated": True,
                    "transformer_positions": 12 + steps,
                    "prefetched_text_positions": 0,
                    "vocabulary_projections": 3,
                    "latency_s": 1.0,
                    "end_to_end_latency_s": 1.1,
                }
                for question in questions
            ]
            predictions[arm] = rows
            path = tmp_path / "evaluations" / f"seed-{seed}" / f"{arm}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            campaign.write_json(
                path.with_suffix(".summary.json"),
                {
                    "predictions_sha256": campaign.digest(path),
                    "overall": {"correct": 2, "n": 2},
                },
            )
        for baseline in ("pause", "short_text", "cot"):
            directory = tmp_path / "timing" / f"seed-{seed}" / f"feedback-vs-{baseline}"
            directory.mkdir(parents=True)
            trace, hashes = [], {}
            for side, arm in (("baseline", baseline), ("candidate", "feedback")):
                rows = predictions[arm]
                path = directory / f"{side}.jsonl"
                path.write_text(
                    "".join(json.dumps({**row, "measurement_repeats": 2}) + "\n" for row in rows)
                )
                hashes[side] = campaign.digest(path)
                for row in rows:
                    for repeat in range(2):
                        trace.append(
                            {
                                **row,
                                "repeat": repeat,
                                "condition": side,
                                "order": repeat if side == "baseline" else 1 - repeat,
                            }
                        )
            path = directory / "measurements.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in trace))
            campaign.write_json(
                directory / "result.json",
                {
                    "measurements_sha256": campaign.digest(path),
                    "predictions_sha256": hashes,
                    "session_calibration": {},
                    "comparison": {"overall": {}},
                },
            )
    campaign.write_json(tmp_path / "selected-checkpoints.json", selections)
    report.summarize(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["feedback_comparisons"]["pause"]["mean_accuracy_delta"] == 0
    assert summary["paired_timing_by_seed"]["42"]["cot"]["mean_latency_reduction"] == 0
    assert (tmp_path / "report.md").exists()
    corrupt = tmp_path / "timing/seed-42/feedback-vs-pause/measurements.jsonl"
    corrupt.write_text(corrupt.read_text().replace('"order": 0', '"order": 1'))
    with pytest.raises(ValueError, match="trace hash mismatch"):
        report.summarize(tmp_path)
