# Latent Gemma

Research implementation of continuous-activation reasoning with Gemma. A learned
bridge feeds transformer hidden states back as input embeddings for a configurable
number of steps, then the model generates an answer. Latent steps do not sample
tokens or project states into vocabulary logits.

**Status: experiments in progress.** Correctness checks pass on small Gemma 3 and
Gemma 4 architectures. Accuracy and speed improvements have not been established.
This project does not reproduce Astra's undisclosed architecture.

## Scope

- MLX backend for Apple Silicon, with LoRA adaptation of existing model weights.
- Gemma 3 text and Gemma 4 text inference/training interfaces.
- Gemma 4 per-layer embeddings: continuous projection with zero token-table
  contribution on latent positions, so hidden states are not discretized by
  the upstream nearest-token fallback. Normal token positions are unchanged.
- Direct-answer, explicit chain-of-thought, and continuous-state experiments.
- Deterministic diagnostic datasets, public GSM8K data preparation, separate
  training/validation/test splits, prediction logs, synchronized GPU timing,
  uncertainty estimates, and activation ablations.

This is embedding recurrence through the full transformer, rather than recurrent
reuse of an internal subset of layers. Each latent position still costs computation.
See the [experiment protocol](docs/protocol.md).

## Install

Python 3.12+ and Apple Silicon are required for the current backend.

```sh
uv venv --python 3.12
uv pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

## Reproduce a diagnostic experiment

Use a directory outside this checkout for model weights and experiment runs.

```sh
.venv/bin/python scripts/download_model.py \
  mlx-community/gemma-3-270m-it-bf16 ../work/models/gemma270m \
  --revision c806ef3a4ed971bd75aaee3346e0fef808512f03
.venv/bin/latent-gemma data --output ../work/data/diagnostics
.venv/bin/latent-gemma train \
  --model ../work/models/gemma270m --data ../work/data/diagnostics \
  --output ../work/runs/warmup --modes direct cot --steps 600 \
  --lora-layers 18 --rank 32 --batch-size 8
.venv/bin/latent-gemma train \
  --model ../work/models/gemma270m --data ../work/data/diagnostics \
  --adapter ../work/runs/warmup/best --output ../work/runs/latent \
  --modes direct cot latent --latent-steps 4 --steps 600 --batch-size 8
.venv/bin/latent-gemma evaluate \
  --model ../work/models/gemma270m --adapter ../work/runs/latent/best \
  --data ../work/data/diagnostics/validation.jsonl --mode latent \
  --latent-steps 4 --max-tokens 16 --output ../work/runs/latent-validation.jsonl
```

Evaluate `--mode direct` and `--mode cot` as matched controls. Use
`--ablation zero`, `--ablation shuffle`, and `--ablation repeat` with latent mode.
Fix configuration choices on validation data before evaluating the test set.
An increase in output token efficiency alone is not evidence of lower latency.

```sh
.venv/bin/python scripts/prepare_gsm8k.py ../work/data/gsm8k
```

GSM8K validation examples come only from the original training split. Original
test examples remain test examples. Neither synthetic training nor small GSM8K
samples establish general-purpose model quality.

## Artifacts and licensing

Runs store adapter-only checkpoints, configuration, losses, and (for new runs)
source snapshots and dependency provenance. Base model weights are not included.
Gemma 3 checkpoints retain their Gemma terms; Gemma 4 is Apache-2.0. The code
license below does not relicense model weights or datasets.

Method references: [Coconut](https://github.com/facebookresearch/coconut),
[recurrent-depth research](https://arxiv.org/abs/2502.05171), and
[MLX LM](https://github.com/ml-explore/mlx-lm).

