"""Compare two and three latent steps after the active test split finishes."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "outputs/latent-gemma"
RUNS = ROOT / "work/runs"
OUTPUT = RUNS / "latent-step-2-vs-3"
MODEL = ROOT / "work/models/gemma-4-e2b-it-4bit"
ADAPTER = RUNS / "gemma4-curriculum-boundary-stage1/best"
DATA = ROOT / "work/data/diagnostics/validation.jsonl"
WRAPPER_PID = 74779
TEST_PID = 74782
TEST_RESULT = RUNS / "frozen-boundary-comparison/test/result.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main():
    wrapper_command = command(WRAPPER_PID)
    test_command = command(TEST_PID)
    if not wrapper_command.endswith("work/frozen_boundary_comparison.py"):
        raise RuntimeError("The expected evaluation coordinator is not running")
    if (
        "scripts/benchmark_pair.py" not in test_command
        or str(TEST_RESULT.parent) not in test_command
    ):
        raise RuntimeError("The expected test split is no longer the active worker")

    paths = [
        *sorted((REPO / "src/latent_gemma").glob("*.py")),
        DATA,
        ADAPTER / "config.json",
        ADAPTER / "adapter.safetensors",
        MODEL / "config.json",
        MODEL / "source.json",
        Path(__file__).resolve(),
    ]
    pins = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    plan = {
        "purpose": "Inference sensitivity to an extra latent step on the same checkpoint; no retraining.",
        "prediction_before_run": "Three steps will probably be slower and will not improve accuracy on the checkpoint trained with two.",
        "trained_latent_steps": 2,
        "baseline_latent_steps": 2,
        "candidate_latent_steps": 3,
        "questions": 100,
        "split": "First 100 diagnostic validation questions, previously used for development and overlapping checkpoint selection.",
        "repeats": 3,
        "total_timed_requests": 600,
        "max_tokens": 96,
        "decode_strategy": "serial",
        "seed": 42,
        "source_sha256": pins,
        "interpretation": "This does not compare independently trained two-step and three-step models or establish a globally optimal step count.",
    }
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT / "plan.json", plan)
    shutil.copy2(__file__, OUTPUT / "run.py")
    state = {
        "coordinator_pid": WRAPPER_PID,
        "coordinator_command": wrapper_command,
        "test_pid": TEST_PID,
        "test_command": test_command,
        "reason": "Finish the current test split, run the requested validation comparison, then continue OOD without overlapping timed GPU workloads.",
    }
    paused = False
    try:
        os.kill(WRAPPER_PID, signal.SIGSTOP)
        paused = True
        state["coordinator_paused_unix_s"] = time.time()
        write_json(OUTPUT / "scheduling.json", state)
        if command(TEST_PID) != test_command:
            raise RuntimeError("Active worker changed while pausing the coordinator")
        print(
            json.dumps(
                {"waiting_for": str(TEST_RESULT), "current_test_continues": True}
            ),
            flush=True,
        )
        while True:
            if TEST_RESULT.exists():
                try:
                    completed = json.loads(TEST_RESULT.read_text())
                except json.JSONDecodeError:
                    time.sleep(1)
                    continue
                if completed["comparison"]["overall"]["n"] != 600:
                    raise RuntimeError("The independent test result is incomplete")
                state["completed_test_result_sha256"] = digest(TEST_RESULT)
                break
            if command(TEST_PID) != test_command:
                raise RuntimeError(
                    "Test worker exited before writing its completed result"
                )
            time.sleep(2)

        for relative, expected in pins.items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f"Frozen comparison input changed: {relative}")
        state["comparison_started_unix_s"] = time.time()
        write_json(OUTPUT / "scheduling.json", state)
        sys.path.insert(0, str(REPO / "src"))
        from latent_gemma.benchmark import DecodeCondition, benchmark_pair
        from latent_gemma.data import read_examples
        from latent_gemma.model import AdapterConfig, load_model
        from latent_gemma.provenance import capture, validate_adapter_model

        saved = json.loads((ADAPTER / "config.json").read_text())
        if saved["run"]["latent_steps"] != 2:
            raise RuntimeError("Unexpected training step count")
        validate_adapter_model(saved, str(MODEL))
        model, tokenizer = load_model(
            str(MODEL), AdapterConfig(**saved["adapter"]), ADAPTER
        )
        examples = read_examples(DATA)[:100]
        if len(examples) != 100:
            raise RuntimeError("Missing validation questions")
        metadata = {
            "model": str(MODEL),
            "adapter_path": str(ADAPTER),
            "data": str(DATA),
            "data_sha256": digest(DATA),
            "resolved_compute_dtype": model.compute_dtype,
            "trained_latent_steps": 2,
            "plan_sha256": digest(OUTPUT / "plan.json"),
            "provenance": capture(OUTPUT / "provenance", str(MODEL), str(ADAPTER)),
        }
        conditions = [
            DecodeCondition(
                "hybrid",
                steps=count,
                max_tokens=96,
                hybrid_boundary=saved["run"]["hybrid_boundary"],
                decode_strategy="serial",
            )
            for count in (2, 3)
        ]
        print(
            json.dumps(
                {"starting": "Two versus three steps", "questions": 100, "repeats": 3}
            ),
            flush=True,
        )
        result = benchmark_pair(
            model,
            tokenizer,
            examples,
            OUTPUT / "paired",
            *conditions,
            repeats=3,
            seed=42,
            metadata=metadata,
        )
        print(
            json.dumps({"completed": True, "comparison": result["comparison"]}),
            flush=True,
        )
        state["comparison_completed_unix_s"] = time.time()
    finally:
        if paused and command(WRAPPER_PID) == wrapper_command:
            os.kill(WRAPPER_PID, signal.SIGCONT)
            state["coordinator_resumed_unix_s"] = time.time()
            print(json.dumps({"resumed_coordinator": WRAPPER_PID}), flush=True)
        write_json(OUTPUT / "scheduling.json", state)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    main()
