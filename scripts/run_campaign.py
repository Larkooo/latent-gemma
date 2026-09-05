"""Freeze and run a sequential, resumable multi-seed curriculum comparison."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

ARMS = ("feedback", "pause", "short_text", "cot")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def runtime():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("mlx", "mlx-lm", "numpy", "transformers")},
    }


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def stage_settings(arm, stage):
    if arm not in ARMS or stage not in range(1, 5):
        raise ValueError("Unknown arm or curriculum stage")
    return {
        "feedback_kind": "pause" if arm == "pause" else "recurrent",
        "latent_steps": 2 * min(stage, 3) if arm in {"feedback", "pause"} else 0,
        "drop_steps": "0" if arm == "cot" else str(stage) if stage < 4 else "all",
    }


def create(args):
    output = args.output.resolve()
    repository = Path(__file__).resolve().parent.parent
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 3:
        raise ValueError("Use at least three distinct training seeds")
    if (
        min(
            args.warmup_epochs,
            args.stage_epochs,
            args.final_epochs,
            args.batch_size,
            args.validation_size,
            args.max_tokens,
            args.timing_questions,
        )
        <= 0
    ):
        raise ValueError("All campaign counts must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("src", "scripts"):
        shutil.copytree(
            repository / directory,
            output / "code" / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    (output / "data").mkdir()
    splits = {}
    for split in ("train", "validation", "test"):
        source = args.data / f"{split}.jsonl"
        shutil.copy2(source, output / "data" / source.name)
        rows = [json.loads(line) for line in source.read_text().splitlines() if line]
        if not rows or any(row["split"] != split for row in rows):
            raise ValueError(f"Invalid {split} split")
        identifiers = {row["id"] for row in rows}
        if len(identifiers) != len(rows):
            raise ValueError(f"Duplicate {split} IDs")
        splits[split] = identifiers
    if any(
        splits[a] & splits[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Overlapping dataset splits")
    manifest = args.data / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, output / "data/manifest.json")
    test_lines = (output / "data/test.jsonl").read_text().splitlines()
    # Freeze the latency subset before any predictions; selection ignores answers.
    random.Random(20260905).shuffle(test_lines)
    timing = test_lines[: args.timing_questions]
    (output / "data/timing.jsonl").write_text("\n".join(timing) + "\n")
    model = Path(args.model).resolve()
    model_files = [path for path in sorted(model.iterdir()) if path.is_file()]
    if not any(path.suffix == ".safetensors" for path in model_files):
        raise ValueError("Local model weights are required")
    frozen_files = [p for d in ("code", "data") for p in (output / d).rglob("*") if p.is_file()]
    plan = {
        "created_unix_s": time.time(),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "repository_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository, text=True
        ),
        "model": str(model),
        "model_sha256": {str(p): digest(p) for p in model_files},
        "python": sys.executable,
        "runtime": runtime(),
        "seeds": args.seeds,
        "arms": list(ARMS),
        "warmup_epochs": args.warmup_epochs,
        "stage_epochs": args.stage_epochs,
        "final_epochs": args.final_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_layers": 35,
        "rank": 16,
        "memory_gb": 16,
        "validation_size": args.validation_size,
        "max_tokens": args.max_tokens,
        "timing_questions": len(timing),
        "timing_repeats": 3,
        "decode_strategy": "pipelined",
        "split_sizes": {name: len(ids) for name, ids in splits.items()},
        "sha256": {str(p.relative_to(output)): digest(p) for p in frozen_files},
        "curriculum": {arm: [stage_settings(arm, i) for i in range(1, 5)] for arm in ARMS},
        "claim_boundary": "Gemma/LoRA adaptation of Coconut on official GSM8K, not a replication of its GPT-2 augmented-data results.",
        "selection": "Same fixed validation subset for every seed and arm. Stage transitions use last weights. Final-stage best validation accuracy selects compressed checkpoints; CoT selects its best across all stages.",
        "comparisons": "Feedback versus trained pause, matched shortened-text curriculum, and continued full-text CoT. All arms receive identical examples and update budgets after their common per-seed warmup; training FLOPs differ.",
        "evaluation": "All test questions scored for every arm after all training finishes. Separate frozen random subset for repeated paired latency. No test-based selection or cap changes.",
        "timing_scope": "Sequential local jobs with no overlapping campaign training; other desktop workload may remain. Session probes are descriptive, not a correction factor.",
    }
    write_json(output / "plan.json", plan)
    write_json(
        output / "status.json", {"status": "prepared", "plan_sha256": digest(output / "plan.json")}
    )
    print(json.dumps({"campaign": str(output), "plan": plan}, indent=2))


def resolve_checkpoint(stage, pointer="last"):
    data = json.loads((stage / f"{pointer}.json").read_text())
    checkpoint = stage / data["checkpoint"]
    if digest(checkpoint / "adapter.safetensors") != data["adapter_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
    return checkpoint


def verify_plan(output, plan):
    if runtime() != plan["runtime"]:
        raise ValueError(
            "Python, package versions, or operating system changed since plan creation"
        )
    for relative, expected in plan["sha256"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    for name, expected in plan["model_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Base model changed: {name}")


def archive_partial(output, paths):
    existing = [path for path in paths if path.exists()]
    if existing:
        archive = output / "interrupted-attempts" / str(time.time_ns())
        for path in existing:
            target = archive / path.relative_to(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), target)


def run_command(output, plan, label, arguments, expected, restart_paths=()):
    if expected.exists():
        return
    archive_partial(output, restart_paths)
    log = output / "logs" / f"{label}.log"
    log.parent.mkdir(exist_ok=True)
    environment = {
        **os.environ,
        "PYTHONPATH": str(output / "code/src"),
        "TOKENIZERS_PARALLELISM": "false",
    }
    command = [plan["python"], *map(str, arguments)]
    with log.open("a") as stream:
        stream.write(json.dumps({"started_unix_s": time.time(), "command": command}) + "\n")
        stream.flush()
        child = subprocess.Popen(
            command,
            cwd=output / "code",
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            pass_fds=(plan["_lock_fd"],),
        )
        try:
            while child.poll() is None:
                write_json(
                    output / "status.json",
                    {
                        "status": "running",
                        "job": label,
                        "worker_pid": child.pid,
                        "coordinator_pid": os.getpid(),
                        "updated_unix_s": time.time(),
                        "log": str(log),
                        "expected_result": str(expected),
                    },
                )
                time.sleep(15)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    if child.returncode != 0 or not expected.exists():
        raise RuntimeError(f"Job {label} failed with exit {child.returncode}; inspect {log}")


def train(output, plan, seed, name, settings, epochs, adapter):
    destination = output / "runs" / f"seed-{seed}" / name
    if destination.exists() and not (destination / "run.json").exists():
        archive_partial(output, [destination])
    arguments = [
        "scripts/train_stage.py",
        "--model",
        plan["model"],
        "--data",
        output / "data",
        "--output",
        destination,
        "--seed",
        seed,
        "--epochs",
        epochs,
        "--batch-size",
        plan["batch_size"],
        "--learning-rate",
        plan["learning_rate"],
        "--validation-size",
        plan["validation_size"],
        "--max-tokens",
        plan["max_tokens"],
        "--lora-layers",
        plan["lora_layers"],
        "--rank",
        plan["rank"],
        "--memory-gb",
        plan["memory_gb"],
        "--latent-steps",
        settings["latent_steps"],
        "--drop-steps",
        settings["drop_steps"],
        "--feedback-kind",
        settings["feedback_kind"],
    ]
    if adapter is not None:
        arguments += ["--adapter", adapter]
    if destination.exists():
        arguments.append("--resume")
    run_command(output, plan, f"seed-{seed}-{name}", arguments, destination / "result.json")
    return destination


def collect_selections(output, plan):
    selections = {}
    for seed in plan["seeds"]:
        root = output / "runs" / f"seed-{seed}"
        selections[str(seed)] = {}
        for arm in ARMS:
            stages = [root / f"{arm}-stage-{i}" for i in range(1, 5)]
            eligible = [root / "warmup", *stages] if arm == "cot" else [stages[-1]]

            def score(stage):
                return json.loads(
                    (resolve_checkpoint(stage, "best") / "training-state.json").read_text()
                )["best_score"]

            selected = max(eligible, key=score)
            path = resolve_checkpoint(selected, "best")
            selections[str(seed)][arm] = {
                "adapter": str(path),
                "sha256": digest(path / "adapter.safetensors"),
                "validation_score": score(selected),
                "latent_steps": 6 if arm in {"feedback", "pause"} else 0,
            }
    return selections


def run(output):
    output = output.resolve()
    plan = json.loads((output / "plan.json").read_text())
    # Children inherit the shared lock, so a killed coordinator cannot start a
    # second campaign while its GPU worker is still alive.
    with (
        (output.parent / ".latent-gemma-gpu.lock").open("a") as gpu_lock,
        (output / ".lock").open("a") as lock,
    ):
        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_plan(output, plan)
        plan["_lock_fd"] = gpu_lock.fileno()
        try:
            for seed in plan["seeds"]:
                warmup = train(
                    output,
                    plan,
                    seed,
                    "warmup",
                    {
                        "feedback_kind": "recurrent",
                        "latent_steps": 0,
                        "drop_steps": "0",
                    },
                    plan["warmup_epochs"],
                    None,
                )
                for arm in ARMS:
                    adapter = resolve_checkpoint(warmup)
                    for stage, settings in enumerate(plan["curriculum"][arm], 1):
                        epochs = plan["final_epochs"] if stage == 4 else plan["stage_epochs"]
                        finished = train(
                            output, plan, seed, f"{arm}-stage-{stage}", settings, epochs, adapter
                        )
                        adapter = resolve_checkpoint(finished)
            selections = collect_selections(output, plan)
            frozen = output / "selected-checkpoints.json"
            if frozen.exists() and json.loads(frozen.read_text()) != selections:
                raise ValueError("Selected checkpoints changed after test evaluation began")
            write_json(frozen, selections)
            for seed, arms in selections.items():
                for arm, selected in arms.items():
                    destination = output / "evaluations" / f"seed-{seed}" / f"{arm}.jsonl"
                    run_command(
                        output,
                        plan,
                        f"test-seed-{seed}-{arm}",
                        [
                            "-m",
                            "latent_gemma.cli",
                            "evaluate",
                            "--model",
                            plan["model"],
                            "--adapter",
                            selected["adapter"],
                            "--data",
                            output / "data/test.jsonl",
                            "--output",
                            destination,
                            "--mode",
                            "hybrid",
                            "--latent-steps",
                            selected["latent_steps"],
                            "--max-tokens",
                            plan["max_tokens"],
                            "--decode-strategy",
                            plan["decode_strategy"],
                        ],
                        destination.with_suffix(".summary.json"),
                        restart_paths=[
                            destination,
                            destination.with_suffix(".provenance"),
                        ],
                    )
                for baseline in ("pause", "short_text", "cot"):
                    destination = output / "timing" / f"seed-{seed}" / f"feedback-vs-{baseline}"
                    run_command(
                        output,
                        plan,
                        f"timing-seed-{seed}-feedback-vs-{baseline}",
                        [
                            "scripts/benchmark_pair.py",
                            "--model",
                            plan["model"],
                            "--adapter",
                            arms["feedback"]["adapter"],
                            "--baseline-adapter",
                            arms[baseline]["adapter"],
                            "--data",
                            output / "data/timing.jsonl",
                            "--output",
                            destination,
                            "--baseline-mode",
                            "hybrid",
                            "--candidate-mode",
                            "hybrid",
                            "--baseline-decode",
                            plan["decode_strategy"],
                            "--candidate-decode",
                            plan["decode_strategy"],
                            "--latent-steps",
                            6,
                            "--baseline-latent-steps",
                            arms[baseline]["latent_steps"],
                            "--max-tokens",
                            plan["max_tokens"],
                            "--repeats",
                            plan["timing_repeats"],
                            "--limit",
                            plan["timing_questions"],
                            "--seed",
                            seed,
                        ],
                        destination / "result.json",
                        restart_paths=[
                            destination,
                            destination.with_suffix(".provenance"),
                        ],
                    )
            run_command(
                output,
                plan,
                "summary",
                [
                    "scripts/summarize_campaign.py",
                    output,
                ],
                output / "summary.json",
            )
            write_json(
                output / "status.json",
                {
                    "status": "complete",
                    "completed_unix_s": time.time(),
                    "summary": str(output / "summary.json"),
                },
            )
        except BaseException as error:
            status = json.loads((output / "status.json").read_text())
            write_json(
                output / "status.json",
                {**status, "status": "stopped", "error": str(error), "updated_unix_s": time.time()},
            )
            raise


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Stopped by signal {signum}; rerun to resume")

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("create")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    prepare.add_argument("--warmup-epochs", type=int, default=3)
    prepare.add_argument("--stage-epochs", type=int, default=1)
    prepare.add_argument("--final-epochs", type=int, default=3)
    prepare.add_argument("--batch-size", type=int, default=8)
    prepare.add_argument("--learning-rate", type=float, default=2e-5)
    prepare.add_argument("--validation-size", type=int, default=128)
    prepare.add_argument("--max-tokens", type=int, default=384)
    prepare.add_argument("--timing-questions", type=int, default=128)
    execute = commands.add_parser("run")
    execute.add_argument("output", type=Path)
    status = commands.add_parser("status")
    status.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        create(args)
    elif args.command == "run":
        run(args.output)
    else:
        print((args.output / "status.json").read_text())


if __name__ == "__main__":
    main()
